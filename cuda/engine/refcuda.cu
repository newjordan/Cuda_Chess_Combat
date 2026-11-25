// =============================================================================
// refcuda.cu — extern "C" GPU chess referee API for Rust/Python FFI.
//
// Exposes cuda-dojo's GPU chess primitives as a stable C ABI. Each entry
// point dispatches actual chess work to GPU kernels (movegen / make_unmake /
// in_check / coord3d run on the device). Host code is just orchestration:
// FEN parsing, host↔device copies, kernel launches.
//
// Built as `librefcuda.so` for FFI loading by Rust (cust / libloading) or
// Python (ctypes / pyo3 + cust). The chess rules are in CUDA C++ once and
// shared by all callers.
// =============================================================================
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "engine_types.h"
#include "movegen.cuh"
#include "make_unmake.cuh"
#include "attacks.cuh"
#include "coord3d.cuh"
#include "search.cuh"
#include "uci.h"
#include "tt.cuh"
#include "zobrist.cuh"

using namespace engine;

// Stub for the only uci.cu symbol search.cu uses. We never run a UCI
// command loop in this .so so external stop signals don't apply.
namespace uci {
    bool stop_requested() { return false; }
}

// =============================================================================
// GPU kernels — chess work runs HERE, not on host.
// =============================================================================
namespace {

__global__ void k_legal_moves(const Position* s, Move* out, int* out_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    Move pseudo[MAX_MOVES];
    int n = generate_moves(s, pseudo);
    int legal = 0;
    for (int i = 0; i < n; ++i) {
        Position c = *s;
        make_move(&c, pseudo[i]);
        if (in_check(&c, 1 - c.side)) continue;
        out[legal++] = pseudo[i];
    }
    *out_count = legal;
}

__global__ void k_make_move(const Position* s_in, Move mv, Position* s_out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    *s_out = *s_in;
    make_move(s_out, mv);
}

__global__ void k_in_check(const Position* s, int* out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    *out = in_check(s, s->side) ? 1 : 0;
}

__global__ void k_legal_count(const Position* s, int* out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    Move buf[MAX_MOVES];
    int n = generate_moves(s, buf);
    int legal = 0;
    for (int i = 0; i < n; ++i) {
        Position c = *s;
        make_move(&c, buf[i]);
        if (!in_check(&c, 1 - c.side)) legal++;
    }
    *out = legal;
}

__global__ void k_coord3d(const Position* s, float* out_xyz, int* out_octant) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    Coord3D c = compute_coord3d(s);
    out_xyz[0] = c.x;
    out_xyz[1] = c.y;
    out_xyz[2] = c.z;
    *out_octant = c.octant_id;
}

// ----- Batched kernels -----------------------------------------------------
// Each block handles one position from the input array. Single thread per
// block (the inner work is serial today; SIMT-parallelizing per-position is
// a follow-up). The win vs single-position kernels is one launch +
// one synchronize for the whole batch instead of N each.

__global__ void k_legal_moves_batched(const Position* s, int n,
                                      Move* out, int* out_counts) {
    int idx = blockIdx.x;
    if (idx >= n || threadIdx.x != 0) return;
    Move pseudo[MAX_MOVES];
    int m = generate_moves(&s[idx], pseudo);
    Move* my_out = out + idx * MAX_MOVES;
    int legal = 0;
    for (int i = 0; i < m; ++i) {
        Position c = s[idx];
        make_move(&c, pseudo[i]);
        if (in_check(&c, 1 - c.side)) continue;
        my_out[legal++] = pseudo[i];
    }
    out_counts[idx] = legal;
}

__global__ void k_make_move_batched(const Position* s_in, int n,
                                    const Move* mvs, Position* s_out) {
    int idx = blockIdx.x;
    if (idx >= n || threadIdx.x != 0) return;
    s_out[idx] = s_in[idx];
    make_move(&s_out[idx], mvs[idx]);
}

__global__ void k_in_check_batched(const Position* s, int n, int* out) {
    int idx = blockIdx.x;
    if (idx >= n || threadIdx.x != 0) return;
    out[idx] = in_check(&s[idx], s[idx].side) ? 1 : 0;
}

static int check_cuda(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return 0;
    std::fprintf(stderr, "refcuda: %s: %s\n", what, cudaGetErrorString(err));
    return -1;
}

// Frees device memory on every exit path, including a failed cudaMalloc.
template <typename T>
struct DevMem {
    T* p = nullptr;
    ~DevMem() { if (p) cudaFree(p); }
    int alloc(size_t count) {
        cudaError_t err = cudaMalloc(&p, sizeof(T) * count);
        if (err != cudaSuccess) {
            p = nullptr;
            return check_cuda(err, "cudaMalloc");
        }
        return 0;
    }
};

} // anonymous namespace

// =============================================================================
// extern "C" API — stable ABI for FFI.
// =============================================================================
extern "C" {

/// Allocate a Position on the host. Caller must free with refc_position_free.
/// Returns NULL on alloc failure.
Position* refc_position_new(void) {
    Position* p = (Position*)std::calloc(1, sizeof(Position));
    return p;
}

void refc_position_free(Position* p) {
    if (p) std::free(p);
}

/// Parse a FEN string into a Position (host work; non-chess parsing).
/// Returns 0 on success, non-zero on parse failure.
int refc_parse_fen(const char* fen, Position* out) {
    if (!fen || !out) return -1;
    return parse_fen(fen, out);
}

/// Compute legal moves for `pos` ON GPU. Writes up to `max_n` moves into
/// `moves_out` and returns the count actually written. The caller's
/// `moves_out` is host memory of length ≥ max_n.
int refc_legal_moves(const Position* pos, int32_t* moves_out, int max_n) {
    if (!pos || !moves_out || max_n <= 0) return -1;

    DevMem<Position> d_pos;
    DevMem<Move> d_moves;
    DevMem<int> d_count;
    if (d_pos.alloc(1) || d_moves.alloc(MAX_MOVES) || d_count.alloc(1)) return -1;

    if (check_cuda(cudaMemcpy(d_pos.p, pos, sizeof(Position), cudaMemcpyHostToDevice),
                   "H2D position")) return -1;
    k_legal_moves<<<1, 1>>>(d_pos.p, d_moves.p, d_count.p);
    if (check_cuda(cudaGetLastError(), "k_legal_moves launch")) return -1;
    if (check_cuda(cudaDeviceSynchronize(), "k_legal_moves sync")) return -1;

    int count = 0;
    if (check_cuda(cudaMemcpy(&count, d_count.p, sizeof(int), cudaMemcpyDeviceToHost),
                   "D2H count")) return -1;
    if (count < 0) count = 0;
    if (count > MAX_MOVES) count = MAX_MOVES;
    if (count > max_n) count = max_n;
    if (count > 0 &&
        check_cuda(cudaMemcpy(moves_out, d_moves.p, sizeof(int32_t) * (size_t)count,
                              cudaMemcpyDeviceToHost),
                   "D2H moves")) {
        return -1;
    }
    return count;
}

/// Apply move `mv` to `pos`, writing result into `out` (must be a different
/// Position). Move execution runs ON GPU.
int refc_make_move(const Position* pos, int32_t mv, Position* out) {
    if (!pos || !out) return -1;

    DevMem<Position> d_in;
    DevMem<Position> d_out;
    if (d_in.alloc(1) || d_out.alloc(1)) return -1;

    if (check_cuda(cudaMemcpy(d_in.p, pos, sizeof(Position), cudaMemcpyHostToDevice),
                   "H2D position")) return -1;
    k_make_move<<<1, 1>>>(d_in.p, mv, d_out.p);
    if (check_cuda(cudaGetLastError(), "k_make_move launch")) return -1;
    if (check_cuda(cudaDeviceSynchronize(), "k_make_move sync")) return -1;
    if (check_cuda(cudaMemcpy(out, d_out.p, sizeof(Position), cudaMemcpyDeviceToHost),
                   "D2H position")) return -1;
    return 0;
}

/// Side-to-move's king under attack (in_check on GPU).
int refc_is_check(const Position* pos) {
    if (!pos) return 0;
    DevMem<Position> d_pos;
    DevMem<int> d_out;
    if (d_pos.alloc(1) || d_out.alloc(1)) return 0;

    if (check_cuda(cudaMemcpy(d_pos.p, pos, sizeof(Position), cudaMemcpyHostToDevice),
                   "H2D position")) return 0;
    k_in_check<<<1, 1>>>(d_pos.p, d_out.p);
    if (check_cuda(cudaGetLastError(), "k_in_check launch")) return 0;
    if (check_cuda(cudaDeviceSynchronize(), "k_in_check sync")) return 0;
    int v = 0;
    if (check_cuda(cudaMemcpy(&v, d_out.p, sizeof(int), cudaMemcpyDeviceToHost),
                   "D2H check")) return 0;
    return v;
}

/// Number of legal moves at this position (computed on GPU).
int refc_legal_count(const Position* pos) {
    if (!pos) return 0;
    DevMem<Position> d_pos;
    DevMem<int> d_out;
    if (d_pos.alloc(1) || d_out.alloc(1)) return -1;

    if (check_cuda(cudaMemcpy(d_pos.p, pos, sizeof(Position), cudaMemcpyHostToDevice),
                   "H2D position")) return -1;
    k_legal_count<<<1, 1>>>(d_pos.p, d_out.p);
    if (check_cuda(cudaGetLastError(), "k_legal_count launch")) return -1;
    if (check_cuda(cudaDeviceSynchronize(), "k_legal_count sync")) return -1;
    int v = 0;
    if (check_cuda(cudaMemcpy(&v, d_out.p, sizeof(int), cudaMemcpyDeviceToHost),
                   "D2H count")) return -1;
    return v;
}

/// is_checkmate: in check AND zero legal moves.
int refc_is_checkmate(const Position* pos) {
    return refc_is_check(pos) && refc_legal_count(pos) == 0;
}

/// is_stalemate: NOT in check AND zero legal moves.
int refc_is_stalemate(const Position* pos) {
    return !refc_is_check(pos) && refc_legal_count(pos) == 0;
}

/// 3D coord (x,y,z) + octant id. Returns 0 on success.
int refc_coord3d(const Position* pos, float* xyz_out, int* octant_out) {
    if (!pos || !xyz_out || !octant_out) return -1;
    DevMem<Position> d_pos;
    DevMem<float> d_xyz;
    DevMem<int> d_oct;
    if (d_pos.alloc(1) || d_xyz.alloc(3) || d_oct.alloc(1)) return -1;

    if (check_cuda(cudaMemcpy(d_pos.p, pos, sizeof(Position), cudaMemcpyHostToDevice),
                   "H2D position")) return -1;
    k_coord3d<<<1, 1>>>(d_pos.p, d_xyz.p, d_oct.p);
    if (check_cuda(cudaGetLastError(), "k_coord3d launch")) return -1;
    if (check_cuda(cudaDeviceSynchronize(), "k_coord3d sync")) return -1;
    if (check_cuda(cudaMemcpy(xyz_out, d_xyz.p, sizeof(float) * 3, cudaMemcpyDeviceToHost),
                   "D2H coord")) return -1;
    if (check_cuda(cudaMemcpy(octant_out, d_oct.p, sizeof(int), cudaMemcpyDeviceToHost),
                   "D2H octant")) return -1;
    return 0;
}

/// UCI string for a move (host-only; format conversion is not chess work).
/// `uci_out` must hold at least 6 bytes (longest UCI = "e7e8q\0").
void refc_move_to_uci(int32_t mv, char* uci_out) {
    if (!uci_out) return;
    move_to_uci(mv, uci_out);
}

/// Sizeof exposed so FFI callers can allocate Position correctly.
int refc_position_size(void) {
    return (int)sizeof(Position);
}

// ============================================================================
// Batched FFI — process N positions in a single launch.
//
// Each batched call replaces N single-position calls and pays:
//   - 1 cudaMalloc per buffer (vs N)
//   - 1 H→D copy of size N×T (vs N copies of size T)
//   - 1 kernel launch (vs N)
//   - 1 cudaDeviceSynchronize (vs N)
//   - 1 D→H copy back (vs N)
//
// Output layout for legal_moves_batched:
//   moves_out[i * MAX_MOVES + 0 .. counts_out[i]-1] = legal moves for pos[i]
//   moves_out is host-side, length ≥ n * MAX_MOVES int32_t.
//   counts_out is host-side, length ≥ n.
// ============================================================================

/// Compute legal moves for N positions in one launch.
int refc_legal_moves_batched(const Position* positions, int n,
                             int32_t* moves_out, int* counts_out) {
    if (!positions || !moves_out || !counts_out || n <= 0) return -1;

    DevMem<Position> d_pos;
    DevMem<Move> d_moves;
    DevMem<int> d_counts;
    if (d_pos.alloc((size_t)n) || d_moves.alloc((size_t)MAX_MOVES * (size_t)n) ||
        d_counts.alloc((size_t)n)) return -1;

    if (check_cuda(cudaMemcpy(d_pos.p, positions, sizeof(Position) * (size_t)n,
                              cudaMemcpyHostToDevice),
                   "H2D positions")) return -1;
    k_legal_moves_batched<<<n, 1>>>(d_pos.p, n, d_moves.p, d_counts.p);
    if (check_cuda(cudaGetLastError(), "k_legal_moves_batched launch")) return -1;
    if (check_cuda(cudaDeviceSynchronize(), "k_legal_moves_batched sync")) return -1;
    if (check_cuda(cudaMemcpy(counts_out, d_counts.p, sizeof(int) * (size_t)n,
                              cudaMemcpyDeviceToHost),
                   "D2H counts")) return -1;
    if (check_cuda(cudaMemcpy(moves_out, d_moves.p,
                              sizeof(int32_t) * MAX_MOVES * (size_t)n,
                              cudaMemcpyDeviceToHost),
                   "D2H moves")) return -1;
    return 0;
}

/// Apply N moves to N positions, output N successor positions.
int refc_make_move_batched(const Position* positions, int n,
                           const int32_t* moves, Position* out) {
    if (!positions || !moves || !out || n <= 0) return -1;

    DevMem<Position> d_in;
    DevMem<Position> d_out;
    DevMem<Move> d_mvs;
    if (d_in.alloc((size_t)n) || d_out.alloc((size_t)n) || d_mvs.alloc((size_t)n)) return -1;

    if (check_cuda(cudaMemcpy(d_in.p, positions, sizeof(Position) * (size_t)n,
                              cudaMemcpyHostToDevice),
                   "H2D positions")) return -1;
    if (check_cuda(cudaMemcpy(d_mvs.p, moves, sizeof(Move) * (size_t)n,
                              cudaMemcpyHostToDevice),
                   "H2D moves")) return -1;
    k_make_move_batched<<<n, 1>>>(d_in.p, n, d_mvs.p, d_out.p);
    if (check_cuda(cudaGetLastError(), "k_make_move_batched launch")) return -1;
    if (check_cuda(cudaDeviceSynchronize(), "k_make_move_batched sync")) return -1;
    if (check_cuda(cudaMemcpy(out, d_out.p, sizeof(Position) * (size_t)n,
                              cudaMemcpyDeviceToHost),
                   "D2H positions")) return -1;
    return 0;
}

/// In-check predicate for N positions.
int refc_is_check_batched(const Position* positions, int n, int* out) {
    if (!positions || !out || n <= 0) return -1;

    DevMem<Position> d_pos;
    DevMem<int> d_out;
    if (d_pos.alloc((size_t)n) || d_out.alloc((size_t)n)) return -1;

    if (check_cuda(cudaMemcpy(d_pos.p, positions, sizeof(Position) * (size_t)n,
                              cudaMemcpyHostToDevice),
                   "H2D positions")) return -1;
    k_in_check_batched<<<n, 1>>>(d_pos.p, n, d_out.p);
    if (check_cuda(cudaGetLastError(), "k_in_check_batched launch")) return -1;
    if (check_cuda(cudaDeviceSynchronize(), "k_in_check_batched sync")) return -1;
    if (check_cuda(cudaMemcpy(out, d_out.p, sizeof(int) * (size_t)n,
                              cudaMemcpyDeviceToHost),
                   "D2H checks")) return -1;
    return 0;
}

// =============================================================================
// Search — alpha-beta + PeSTO via cuda-dojo's existing GPU search kernels.
// Calls search_root which runs iterative deepening with TT/zobrist on GPU.
// =============================================================================

static bool g_search_initialised = false;

/// Initialize search subsystem (TT, zobrist, root buffers). Idempotent.
void refc_search_init(void) {
    if (!g_search_initialised) {
        if (!zobrist_initialized()) init_zobrist();
        tt_init();
        search_init();
        g_search_initialised = true;
    }
}

/// Reset between games (clears the host-side stop flag mirror).
void refc_search_new_game(void) {
    if (!g_search_initialised) refc_search_init();
    search_new_game();
}

/// Free search-side device memory.
void refc_search_shutdown(void) {
    if (g_search_initialised) {
        search_shutdown();
        g_search_initialised = false;
    }
}

/// Search `pos` to fixed depth (or movetime). Writes UCI bestmove (≤6 bytes
/// incl. NUL) into `uci_out`, score (cp from STM POV) into `score_out`.
/// Returns 0 on success, non-zero on error.
///
/// One of (depth, movetime_ms) should be > 0. depth takes precedence.
int refc_search(const Position* pos, int depth, int movetime_ms,
                char* uci_out, int* score_out)
{
    if (!pos || !uci_out) return -1;
    if (!g_search_initialised) refc_search_init();

    uci::SearchLimits limits;
    if (depth > 0) limits.depth = depth;
    else if (movetime_ms > 0) limits.movetime = movetime_ms;
    else limits.depth = 5;  // sensible default

    int last_score = 0;
    auto cb = [&](const uci::SearchInfo& info) {
        last_score = info.score_cp;
    };

    uci::SearchResult res = search_root(*pos, limits, cb);

    // Copy bestmove UCI (max 6 chars incl. NUL).
    std::strncpy(uci_out, res.bestmove.c_str(), 6);
    uci_out[5] = '\0';
    if (score_out) *score_out = last_score;
    return 0;
}

/// Convenience: search to depth, then return the encoded Move + score.
/// Maps the UCI string back to a Move by enumerating legal_moves and
/// matching by UCI text. Slightly slower than direct search but lets
/// callers stay in the typed Move/Position world.
int refc_search_best_move(const Position* pos, int depth, int movetime_ms,
                            int32_t* move_out, int* score_out)
{
    if (!pos || !move_out) return -1;

    char uci[6] = {0};
    int score = 0;
    if (refc_search(pos, depth, movetime_ms, uci, &score) != 0) return -1;
    if (score_out) *score_out = score;

    // Match UCI to a legal move.
    int32_t legal[MAX_MOVES];
    int n = refc_legal_moves(pos, legal, MAX_MOVES);
    for (int i = 0; i < n; ++i) {
        char buf[8] = {0};
        move_to_uci(legal[i], buf);
        if (std::strncmp(buf, uci, 6) == 0) {
            *move_out = legal[i];
            return 0;
        }
    }
    // Fell through — search returned a non-legal UCI? Use first legal.
    if (n > 0) {
        *move_out = legal[0];
        return 1;  // soft failure
    }
    return -1;
}

} // extern "C"
