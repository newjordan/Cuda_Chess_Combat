// =============================================================================
// refcpu.cpp — host implementation of the librefcuda C ABI.
//
// Same entry points as refcuda.cu, but legal-move generation, make/unmake,
// check detection, and coord3d run on the CPU. Search entry points return
// "not available" because alpha-beta lives in the CUDA search kernels.
// =============================================================================
#include "engine_types.h"
#include "movegen.cuh"
#include "make_unmake.cuh"
#include "attacks.cuh"
#include "coord3d.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstring>

using namespace engine;

static int legal_into(const Position* pos, int32_t* moves_out, int max_n) {
    Move pseudo[MAX_MOVES];
    int n = generate_moves(pos, pseudo);
    int written = 0;
    for (int i = 0; i < n; ++i) {
        Position child = *pos;
        make_move(&child, pseudo[i]);
        if (in_check(&child, 1 - child.side)) continue;
        if (written < max_n) moves_out[written] = pseudo[i];
        written++;
    }
    return written > max_n ? max_n : written;
}

extern "C" {

Position* refc_position_new(void) {
    return static_cast<Position*>(std::calloc(1, sizeof(Position)));
}

void refc_position_free(Position* p) {
    std::free(p);
}

int refc_parse_fen(const char* fen, Position* out) {
    if (!fen || !out) return -1;
    return parse_fen(fen, out);
}

int refc_legal_moves(const Position* pos, int32_t* moves_out, int max_n) {
    if (!pos || !moves_out || max_n <= 0) return -1;
    return legal_into(pos, moves_out, max_n);
}

int refc_make_move(const Position* pos, int32_t mv, Position* out) {
    if (!pos || !out) return -1;
    *out = *pos;
    make_move(out, mv);
    return 0;
}

int refc_is_check(const Position* pos) {
    if (!pos) return 0;
    if (pos->side != WHITE_SIDE && pos->side != BLACK_SIDE) return 0;
    if (pos->kingPos[pos->side] < 0) return 0;
    return in_check(pos, pos->side) ? 1 : 0;
}

int refc_legal_count(const Position* pos) {
    if (!pos) return 0;
    int32_t buf[MAX_MOVES];
    return legal_into(pos, buf, MAX_MOVES);
}

int refc_is_checkmate(const Position* pos) {
    if (!pos) return 0;
    return refc_is_check(pos) && refc_legal_count(pos) == 0;
}

int refc_is_stalemate(const Position* pos) {
    if (!pos) return 0;
    return !refc_is_check(pos) && refc_legal_count(pos) == 0;
}

int refc_coord3d(const Position* pos, float* xyz_out, int* octant_out) {
    if (!pos || !xyz_out || !octant_out) return -1;
    Coord3D c = compute_coord3d(pos);
    xyz_out[0] = c.x;
    xyz_out[1] = c.y;
    xyz_out[2] = c.z;
    *octant_out = c.octant_id;
    return 0;
}

void refc_move_to_uci(int32_t mv, char* uci_out) {
    if (!uci_out) return;
    move_to_uci(mv, uci_out);
}

int refc_position_size(void) {
    return static_cast<int>(sizeof(Position));
}

int refc_legal_moves_batched(const Position* positions, int n,
                             int32_t* moves_out, int* counts_out) {
    if (!positions || !moves_out || !counts_out || n <= 0) return -1;
    for (int i = 0; i < n; ++i) {
        counts_out[i] = legal_into(&positions[i],
                                   moves_out + static_cast<size_t>(i) * MAX_MOVES,
                                   MAX_MOVES);
    }
    return 0;
}

int refc_make_move_batched(const Position* positions, int n,
                           const int32_t* moves, Position* out) {
    if (!positions || !moves || !out || n <= 0) return -1;
    for (int i = 0; i < n; ++i) {
        if (refc_make_move(&positions[i], moves[i], &out[i]) != 0) return -1;
    }
    return 0;
}

int refc_is_check_batched(const Position* positions, int n, int* out) {
    if (!positions || !out || n <= 0) return -1;
    for (int i = 0; i < n; ++i) out[i] = refc_is_check(&positions[i]);
    return 0;
}

void refc_search_init(void) {}
void refc_search_new_game(void) {}
void refc_search_shutdown(void) {}

int refc_search_best_move(const Position* /*pos*/, int /*depth*/, int /*movetime_ms*/,
                          int32_t* /*move_out*/, int* /*score_out*/) {
    return -1;
}

} // extern "C"
