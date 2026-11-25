// =============================================================================
// cpu_rules_test — host-only rules suite.
//
// Perft totals are the published chessprogramming.org "Perft Results"
// counts. Move generation, make/unmake, and Zobrist updates are the same
// sources the CUDA build compiles; this binary just does not launch kernels.
// =============================================================================
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../movegen.cuh"
#include "../make_unmake.cuh"
#include "../attacks.cuh"
#include "../zobrist.cuh"

using namespace engine;

static int g_failures = 0;

static void expect(bool ok, const char* msg) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s\n", msg);
    g_failures++;
}

struct PerftCase {
    const char* name;
    const char* fen;
    int depth;
    long long expected;
};

// https://www.chessprogramming.org/Perft_Results
static const PerftCase PERFT[] = {
    {"startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, 20},
    {"startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2, 400},
    {"startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3, 8902},
    {"startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281},
    {"startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609},

    {"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1, 48},
    {"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2, 2039},
    {"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862},
    {"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},

    {"pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 1, 14},
    {"pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 2, 191},
    {"pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2812},
    {"pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43238},
    {"pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624},

    {"pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 1, 6},
    {"pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 2, 264},
    {"pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9467},
    {"pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333},

    {"pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 1, 44},
    {"pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 2, 1486},
    {"pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379},
    {"pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487},

    {"pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 1, 46},
    {"pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 2, 2079},
    {"pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3, 89890},
    {"pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3894594},
};

static int legal_moves(const Position* s, Move* out) {
    Move pseudo[MAX_MOVES];
    int n = generate_moves(s, pseudo);
    int m = 0;
    for (int i = 0; i < n; i++) {
        Position child = *s;
        make_move(&child, pseudo[i]);
        if (in_check(&child, 1 - child.side)) continue;
        out[m++] = pseudo[i];
    }
    return m;
}

static bool has_uci(const Position* s, const char* want) {
    Move moves[MAX_MOVES];
    int n = legal_moves(s, moves);
    for (int i = 0; i < n; i++) {
        char buf[8] = {0};
        move_to_uci(moves[i], buf);
        if (std::strcmp(buf, want) == 0) return true;
    }
    return false;
}

static std::string fen_of(const Position& s) {
    char buf[128];
    if (position_to_fen(&s, buf, (int)sizeof(buf)) != 0) return {};
    return buf;
}

static Position must_parse(const char* fen) {
    Position p{};
    if (parse_fen(fen, &p) != 0) {
        std::fprintf(stderr, "FAIL parse '%s'\n", fen);
        g_failures++;
    }
    return p;
}

static int unmake_tree(Position* s, int depth, long long* checks) {
    if (depth == 0) return 0;
    Move moves[MAX_MOVES];
    int n = legal_moves(s, moves);
    int failures = 0;
    for (int i = 0; i < n; i++) {
        Position before = *s;
        Position work = *s;
        Undo undo{};
        make_move(&work, moves[i], &undo);
        unmake_move(&work, moves[i], &undo);
        (*checks)++;
        if (std::memcmp(&before, &work, sizeof(Position)) != 0) failures++;
        Position child = before;
        make_move(&child, moves[i]);
        failures += unmake_tree(&child, depth - 1, checks);
    }
    return failures;
}

static int zobrist_tree(const Position& s, uint64_t hash, int depth, long long* checks) {
    if (depth == 0) return 0;
    Move moves[MAX_MOVES];
    int n = legal_moves(&s, moves);
    int failures = 0;
    for (int i = 0; i < n; i++) {
        uint64_t next = zobrist_update(hash, s, moves[i]);
        Position child = s;
        make_move(&child, moves[i]);
        (*checks)++;
        if (zobrist_full(child) != next) failures++;
        failures += zobrist_tree(child, next, depth - 1, checks);
    }
    return failures;
}

int main() {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();

    std::printf("sizeof_position=%zu\n", sizeof(Position));

    long long perft_nodes = 0;
    int perft_pass = 0;
    const int perft_n = (int)(sizeof(PERFT) / sizeof(PERFT[0]));
    for (int i = 0; i < perft_n; i++) {
        Position p = must_parse(PERFT[i].fen);
        long long got = perft(&p, PERFT[i].depth);
        bool ok = got == PERFT[i].expected;
        std::printf("perft %-8s d=%d got=%lld expected=%lld %s\n",
                    PERFT[i].name, PERFT[i].depth, got, PERFT[i].expected,
                    ok ? "ok" : "FAIL");
        if (!ok) g_failures++;
        else perft_pass++;
        perft_nodes += got;
    }

    const char* start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    Position p = must_parse(start);
    expect(fen_of(p) == start, "startpos fen roundtrip");
    expect(has_uci(&p, "e2e4") && has_uci(&p, "g1f3") && !has_uci(&p, "e2e5"),
           "startpos move list");
    expect(!in_check(&p, WHITE_SIDE), "startpos not in check");

    expect(parse_fen("not a fen", &p) != 0, "reject garbage fen");
    expect(parse_fen("8/8/8/8/8/8/8/8 w - - 0 1", &p) != 0, "reject missing kings");
    expect(parse_fen("9/8/8/8/8/8/8/8 w - - 0 1", &p) != 0, "reject rank overflow");
    expect(parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e9 0 1", &p) != 0,
           "reject ep off board");
    expect(parse_fen(nullptr, &p) != 0, "reject null fen");
    // A placement longer than 64 squares must fail instead of writing past board[].
    expect(parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNRp w KQkq - 0 1", &p) != 0,
           "reject extra piece");

    Position after = must_parse(start);
    Move moves[MAX_MOVES];
    int n = legal_moves(&after, moves);
    bool played = false;
    for (int i = 0; i < n; i++) {
        char buf[8] = {0};
        move_to_uci(moves[i], buf);
        if (std::strcmp(buf, "e2e4") == 0) {
            make_move(&after, moves[i]);
            played = true;
            break;
        }
    }
    expect(played, "found e2e4");
    expect(fen_of(after) ==
               "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
           "fen after e2e4");

    Position ep = must_parse("rnbqkbnr/pp1ppppp/8/2ppP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3");
    expect(has_uci(&ep, "e5d6"), "en passant e5d6 is legal");

    // Pinned en passant: exd6 opens the e-file onto the white king.
    // Black king sits on a8 so the FEN is legal; the rook on e8 pins exd6.
    Position pin = must_parse("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    expect(!has_uci(&pin, "e5d6"), "pinned en passant is illegal");
    expect(has_uci(&pin, "e5e6"), "push that keeps the pin closed is legal");
    Move pin_moves[MAX_MOVES];
    expect(legal_moves(&pin, pin_moves) == 6, "pinned ep position has 6 legal moves");

    Position castle = must_parse("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    expect(has_uci(&castle, "e1g1") && has_uci(&castle, "e1c1"), "both white castles");
    Position no_castle = must_parse("r3k2r/8/8/8/5q2/8/8/R3K2R w KQkq - 0 1");
    expect(!has_uci(&no_castle, "e1g1"), "kingside castle through check");
    expect(!has_uci(&no_castle, "e1c1"), "queenside castle onto attacked square");

    Position rook_cap = must_parse("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    n = legal_moves(&rook_cap, moves);
    played = false;
    for (int i = 0; i < n; i++) {
        char buf[8] = {0};
        move_to_uci(moves[i], buf);
        if (std::strcmp(buf, "a1a8") == 0) {
            make_move(&rook_cap, moves[i]);
            played = true;
            break;
        }
    }
    expect(played, "a1a8 capture");
    expect(fen_of(rook_cap) == "R3k2r/8/8/8/8/8/8/4K2R b Kk - 0 1",
           "capturing a rook clears that castle right");

    Position promo = must_parse("8/P7/8/8/8/8/8/k1K5 w - - 0 1");
    expect(has_uci(&promo, "a7a8q") && has_uci(&promo, "a7a8n") &&
               has_uci(&promo, "a7a8r") && has_uci(&promo, "a7a8b"),
           "underpromotions");

    Position mate = must_parse("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    Move mate_moves[MAX_MOVES];
    expect(in_check(&mate, WHITE_SIDE) && legal_moves(&mate, mate_moves) == 0,
           "fool's mate is checkmate");

    Position stale = must_parse("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    Move stale_moves[MAX_MOVES];
    expect(!in_check(&stale, BLACK_SIDE) && legal_moves(&stale, stale_moves) == 0,
           "stalemate");

    // Stray ep token with no victim pawn must not create a capture.
    Position stray = must_parse("8/8/8/8/3P4/8/k7/4K3 w - e3 0 1");
    expect(!has_uci(&stray, "d4e3"), "ep without a victim pawn");

    long long unmake_checks = 0;
    Position uroot = must_parse(start);
    int unmake_fail = unmake_tree(&uroot, 3, &unmake_checks);
    Position ukiwi = must_parse("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    unmake_fail += unmake_tree(&ukiwi, 2, &unmake_checks);
    Position upromo = must_parse("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
    unmake_fail += unmake_tree(&upromo, 2, &unmake_checks);
    expect(unmake_fail == 0, "make/unmake restores the position");
    std::printf("unmake_checks=%lld unmake_mismatches=%d\n", unmake_checks, unmake_fail);

    init_zobrist();
    long long zobrist_checks = 0;
    Position zroot = must_parse(start);
    int zfail = zobrist_tree(zroot, zobrist_full(zroot), 3, &zobrist_checks);
    Position zkiwi = must_parse("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    zfail += zobrist_tree(zkiwi, zobrist_full(zkiwi), 2, &zobrist_checks);
    Position zpromo = must_parse("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
    zfail += zobrist_tree(zpromo, zobrist_full(zpromo), 2, &zobrist_checks);
    expect(zfail == 0, "zobrist incremental matches full hash");
    std::printf("zobrist_checks=%lld zobrist_mismatches=%d\n", zobrist_checks, zfail);

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();
    std::printf("perft_cases=%d perft_passed=%d perft_nodes=%lld elapsed_ms=%lld failures=%d\n",
                perft_n, perft_pass, perft_nodes, (long long)ms, g_failures);
    return g_failures == 0 ? 0 : 1;
}
