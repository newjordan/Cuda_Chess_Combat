// =============================================================================
// cpu_inspect — print a position and its legal moves. No GPU required.
//
//   cpu_inspect
//   cpu_inspect "FEN"
//   cpu_inspect --play e2e4 e7e5 g1f3
//   cpu_inspect "FEN" --play d5
// =============================================================================
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "movegen.cuh"
#include "make_unmake.cuh"
#include "attacks.cuh"

using namespace engine;

static const char* START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

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

static char piece_glyph(int piece) {
    switch (piece) {
        case WPAWN: return 'P';
        case WKNIGHT: return 'N';
        case WBISHOP: return 'B';
        case WROOK: return 'R';
        case WQUEEN: return 'Q';
        case WKING: return 'K';
        case BPAWN: return 'p';
        case BKNIGHT: return 'n';
        case BBISHOP: return 'b';
        case BROOK: return 'r';
        case BQUEEN: return 'q';
        case BKING: return 'k';
        default: return '.';
    }
}

static const char* piece_group(int piece) {
    switch (piece_type(piece)) {
        case 1: return "pawns  ";
        case 2: return "knights";
        case 3: return "bishops";
        case 4: return "rooks  ";
        case 5: return "queens ";
        case 6: return "king   ";
        default: return "other  ";
    }
}

static void print_board(const Position& s) {
    std::fputs("    a   b   c   d   e   f   g   h\n", stdout);
    std::fputs("  +---+---+---+---+---+---+---+---+\n", stdout);
    for (int r = 0; r < 8; r++) {
        std::printf("%d |", 8 - r);
        for (int f = 0; f < 8; f++) {
            std::printf(" %c |", piece_glyph(s.board[make_sq(r, f)]));
        }
        std::printf(" %d\n", 8 - r);
        std::fputs("  +---+---+---+---+---+---+---+---+\n", stdout);
    }
    std::fputs("    a   b   c   d   e   f   g   h\n", stdout);

    char fen[128];
    if (position_to_fen(&s, fen, (int)sizeof(fen)) != 0) {
        std::snprintf(fen, sizeof(fen), "(fen unavailable)");
    }
    const bool check = s.kingPos[s.side] >= 0 && in_check(&s, s.side);
    std::printf("\n%s to move    %s\n",
                s.side == BLACK_SIDE ? "Black" : "White",
                check ? "In check." : "Not in check.");
    std::printf("%s\n", fen);
}

static void print_moves(const Position& s) {
    Move moves[MAX_MOVES];
    int n = legal_moves(&s, moves);
    std::printf("\n%d legal move%s.\n", n, n == 1 ? "" : "s");
    if (n == 0) return;

    // Stable groups in piece order, moves already come out in generator order.
    for (int group = 1; group <= 6; group++) {
        std::string line;
        int count = 0;
        for (int i = 0; i < n; i++) {
            int piece = s.board[move_from(moves[i])];
            if (piece_type(piece) != group) continue;
            char buf[8] = {0};
            move_to_uci(moves[i], buf);
            if (!line.empty()) line.push_back(' ');
            line += buf;
            count++;
        }
        if (count == 0) continue;
        const char* label = piece_group(group);
        std::printf("  %s  ", label);
        int col = 11;
        std::string word;
        for (char ch : line) {
            if (ch == ' ') {
                if (col + 1 + (int)word.size() > 78) {
                    std::printf("\n           ");
                    col = 11;
                } else if (col != 11) {
                    std::fputc(' ', stdout);
                    col++;
                }
                std::fputs(word.c_str(), stdout);
                col += (int)word.size();
                word.clear();
            } else {
                word.push_back(ch);
            }
        }
        if (!word.empty()) {
            if (col + 1 + (int)word.size() > 78 && col != 11) {
                std::printf("\n           ");
            } else if (col != 11) {
                std::fputc(' ', stdout);
            }
            std::fputs(word.c_str(), stdout);
        }
        std::fputc('\n', stdout);
    }
}

static bool apply_uci(Position* s, const std::string& uci) {
    Move moves[MAX_MOVES];
    int n = legal_moves(s, moves);
    for (int i = 0; i < n; i++) {
        char buf[8] = {0};
        move_to_uci(moves[i], buf);
        if (uci == buf) {
            make_move(s, moves[i]);
            return true;
        }
    }
    return false;
}

static void usage() {
    std::fputs(
        "usage: cpu_inspect [FEN] [--play UCI...]\n"
        "\n"
        "Print the board, FEN, check status, and legal moves.\n"
        "With no arguments, start from the standard initial position.\n"
        "Uppercase pieces are white. Square a8 is the top-left corner.\n",
        stderr);
}

int main(int argc, char** argv) {
    const char* fen = START_FEN;
    std::vector<std::string> play;
    bool playing = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage();
            return 0;
        }
        if (std::strcmp(argv[i], "--play") == 0) {
            playing = true;
            continue;
        }
        if (playing) play.emplace_back(argv[i]);
        else if (fen == START_FEN) fen = argv[i];
        else {
            usage();
            return 2;
        }
    }

    Position pos{};
    if (parse_fen(fen, &pos) != 0) {
        std::fprintf(stderr, "cpu_inspect: invalid FEN: %s\n", fen);
        return 1;
    }
    for (const std::string& uci : play) {
        if (!apply_uci(&pos, uci)) {
            std::fprintf(stderr, "cpu_inspect: illegal move %s\n", uci.c_str());
            print_board(pos);
            print_moves(pos);
            return 1;
        }
    }
    print_board(pos);
    print_moves(pos);
    return 0;
}
