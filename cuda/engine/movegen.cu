// =============================================================================
// movegen.cu
//
// Pseudo-legal movegen extracted verbatim from gpu_fighter.cu's
// d_generate_moves / h_generate_moves (which were already byte-identical
// except for compiler annotations). Single __host__ __device__ body.
//
// Plus host helpers: parse_fen, move_to_uci, perft (matches the perft
// implementation gpu_fighter.cu was verified against).
// =============================================================================
#include "movegen.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace engine {

__host__ __device__ static inline int abs_i(int x) { return x < 0 ? -x : x; }

// Never write past MAX_MOVES. A single queen or promoting pawn can emit more
// moves than the old "leave 8 slots" guard allowed, which overflowed the
// caller buffer.
__host__ __device__ static inline int push_move(Move* moves, int count, Move m) {
    if (count < MAX_MOVES) {
        moves[count] = m;
        return count + 1;
    }
    return count;
}

// -------- pseudo-legal movegen, identical to verified gpu_fighter logic --------
__host__ __device__ int generate_moves(const Position* s, Move* moves) {
    const int KN[8] = {-17, -15, -10, -6, 6, 10, 15, 17};
    const int KG[8] = { -9,  -8,  -7, -1, 1,  7,  8,  9};

    int count = 0;
    int side = s->side, opp = 1 - side;

    for (int sq = 0; sq < 64; sq++) {
        int piece = s->board[sq];
        if (piece == EMPTY) continue;
        if (side == WHITE_SIDE && !is_white(piece)) continue;
        if (side == BLACK_SIDE && !is_black(piece)) continue;

        int r = sq_rank(sq), f = sq_file(sq);
        int ptype = piece_type(piece);

        if (ptype == 1) { // pawn
            int dir       = (side == WHITE_SIDE) ? -1 : 1;
            int startRank = (side == WHITE_SIDE) ?  6 : 1;
            int promoRank = (side == WHITE_SIDE) ?  0 : 7;
            int nr = r + dir;

            // single push
            if (on_board(nr, f) && s->board[make_sq(nr, f)] == EMPTY) {
                if (nr == promoRank) {
                    int pb = (side == WHITE_SIDE) ? WQUEEN : BQUEEN;
                    count = push_move(moves, count, make_move(sq, make_sq(nr,f), pb,   FLAG_PROMO));
                    count = push_move(moves, count, make_move(sq, make_sq(nr,f), pb-1, FLAG_PROMO));
                    count = push_move(moves, count, make_move(sq, make_sq(nr,f), pb-2, FLAG_PROMO));
                    count = push_move(moves, count, make_move(sq, make_sq(nr,f), pb-3, FLAG_PROMO));
                } else {
                    count = push_move(moves, count, make_move(sq, make_sq(nr,f), 0, FLAG_NONE));
                }
                if (r == startRank) {
                    int nr2 = r + 2*dir;
                    if (on_board(nr2, f) && s->board[make_sq(nr2,f)] == EMPTY)
                        count = push_move(moves, count, make_move(sq, make_sq(nr2,f), 0, FLAG_DOUBLE));
                }
            }

            // pawn captures + ep
            for (int df = -1; df <= 1; df += 2) {
                int nf = f + df;
                if (!on_board(nr, nf)) continue;
                int t  = make_sq(nr, nf);
                int tp = s->board[t];
                if (tp != EMPTY && piece_color(tp) == opp) {
                    if (nr == promoRank) {
                        int pb = (side == WHITE_SIDE) ? WQUEEN : BQUEEN;
                        count = push_move(moves, count, make_move(sq, t, pb,   FLAG_PROMO));
                        count = push_move(moves, count, make_move(sq, t, pb-1, FLAG_PROMO));
                        count = push_move(moves, count, make_move(sq, t, pb-2, FLAG_PROMO));
                        count = push_move(moves, count, make_move(sq, t, pb-3, FLAG_PROMO));
                    } else {
                        count = push_move(moves, count, make_move(sq, t, 0, FLAG_CAPTURE));
                    }
                }
                // En passant is only legal when the captured pawn is actually
                // sitting behind the target square. A stray ep token must not
                // invent a capture.
                if (t == s->ep && s->ep >= 0) {
                    int cs = (side == WHITE_SIDE) ? t + 8 : t - 8;
                    int victim = (side == WHITE_SIDE) ? BPAWN : WPAWN;
                    if (cs >= 0 && cs < 64 && s->board[cs] == victim)
                        count = push_move(moves, count, make_move(sq, t, 0, FLAG_EP));
                }
            }
        }
        else if (ptype == 2) { // knight
            for (int i = 0; i < 8; i++) {
                int t = sq + KN[i];
                if (t < 0 || t >= 64) continue;
                int dr = abs_i(sq_rank(t) - r), df2 = abs_i(sq_file(t) - f);
                if (!((dr==2 && df2==1) || (dr==1 && df2==2))) continue;
                int tp = s->board[t];
                if (tp == EMPTY) count = push_move(moves, count, make_move(sq, t, 0, FLAG_NONE));
                else if (piece_color(tp) == opp) count = push_move(moves, count, make_move(sq, t, 0, FLAG_CAPTURE));
            }
        }

        if (ptype == 3 || ptype == 5) { // bishop / queen diagonals
            int ddr[] = {-1,-1,1,1}, ddf[] = {-1,1,-1,1};
            for (int d = 0; d < 4; d++) {
                int cr = r + ddr[d], cf = f + ddf[d];
                while (on_board(cr, cf)) {
                    int t = make_sq(cr, cf);
                    int tp = s->board[t];
                    if (tp == EMPTY) count = push_move(moves, count, make_move(sq, t, 0, FLAG_NONE));
                    else { if (piece_color(tp)==opp) count = push_move(moves, count, make_move(sq, t, 0, FLAG_CAPTURE)); break; }
                    cr += ddr[d]; cf += ddf[d];
                }
            }
        }

        if (ptype == 4 || ptype == 5) { // rook / queen orthogonals
            int sdr[] = {-1,1,0,0}, sdf[] = {0,0,-1,1};
            for (int d = 0; d < 4; d++) {
                int cr = r + sdr[d], cf = f + sdf[d];
                while (on_board(cr, cf)) {
                    int t = make_sq(cr, cf);
                    int tp = s->board[t];
                    if (tp == EMPTY) count = push_move(moves, count, make_move(sq, t, 0, FLAG_NONE));
                    else { if (piece_color(tp)==opp) count = push_move(moves, count, make_move(sq, t, 0, FLAG_CAPTURE)); break; }
                    cr += sdr[d]; cf += sdf[d];
                }
            }
        }

        if (ptype == 6) { // king
            for (int i = 0; i < 8; i++) {
                int t = sq + KG[i];
                if (t < 0 || t >= 64) continue;
                int dr = abs_i(sq_rank(t)-r), df2 = abs_i(sq_file(t)-f);
                if (dr > 1 || df2 > 1) continue;
                int tp = s->board[t];
                if (tp == EMPTY) count = push_move(moves, count, make_move(sq, t, 0, FLAG_NONE));
                else if (piece_color(tp) == opp) count = push_move(moves, count, make_move(sq, t, 0, FLAG_CAPTURE));
            }
            // castling
            if (side == WHITE_SIDE && r == 7 && f == 4) {
                if ((s->castle & CASTLE_WK) && s->board[61]==EMPTY && s->board[62]==EMPTY && s->board[63]==WROOK
                    && !is_square_attacked(s,60,BLACK_SIDE) && !is_square_attacked(s,61,BLACK_SIDE) && !is_square_attacked(s,62,BLACK_SIDE))
                    count = push_move(moves, count, make_move(60,62,0,FLAG_CASTLE_K));
                if ((s->castle & CASTLE_WQ) && s->board[59]==EMPTY && s->board[58]==EMPTY && s->board[57]==EMPTY && s->board[56]==WROOK
                    && !is_square_attacked(s,60,BLACK_SIDE) && !is_square_attacked(s,59,BLACK_SIDE) && !is_square_attacked(s,58,BLACK_SIDE))
                    count = push_move(moves, count, make_move(60,58,0,FLAG_CASTLE_Q));
            }
            if (side == BLACK_SIDE && r == 0 && f == 4) {
                if ((s->castle & CASTLE_BK) && s->board[5]==EMPTY && s->board[6]==EMPTY && s->board[7]==BROOK
                    && !is_square_attacked(s,4,WHITE_SIDE) && !is_square_attacked(s,5,WHITE_SIDE) && !is_square_attacked(s,6,WHITE_SIDE))
                    count = push_move(moves, count, make_move(4,6,0,FLAG_CASTLE_K));
                if ((s->castle & CASTLE_BQ) && s->board[3]==EMPTY && s->board[2]==EMPTY && s->board[1]==EMPTY && s->board[0]==BROOK
                    && !is_square_attacked(s,4,WHITE_SIDE) && !is_square_attacked(s,3,WHITE_SIDE) && !is_square_attacked(s,2,WHITE_SIDE))
                    count = push_move(moves, count, make_move(4,2,0,FLAG_CASTLE_Q));
            }
        }
    }
    return count;
}

// ---------- host: FEN parser ----------
static void clear_position(Position* s) {
    memset(s, 0, sizeof(Position));
    for (int i = 0; i < 64; i++) s->board[i] = EMPTY;
    s->ep = -1;
    s->kingPos[0] = -1;
    s->kingPos[1] = -1;
    s->fullmove = 1;
}

static int piece_from_fen_char(char c) {
    switch (c) {
        case 'P': return WPAWN;
        case 'N': return WKNIGHT;
        case 'B': return WBISHOP;
        case 'R': return WROOK;
        case 'Q': return WQUEEN;
        case 'K': return WKING;
        case 'p': return BPAWN;
        case 'n': return BKNIGHT;
        case 'b': return BBISHOP;
        case 'r': return BROOK;
        case 'q': return BQUEEN;
        case 'k': return BKING;
        default:  return -1;
    }
}

int parse_fen(const char* fen, Position* s) {
    if (!s) return -1;
    clear_position(s);
    if (!fen) return -1;

    Position tmp;
    clear_position(&tmp);

    int i = 0;
    int sq = 0;
    int rank_squares = 0;
    int ranks = 1;
    bool any_placement = false;
    while (fen[i] && fen[i] != ' ') {
        char c = fen[i++];
        any_placement = true;
        if (c == '/') {
            if (rank_squares != 8 || ranks >= 8) return -1;
            rank_squares = 0;
            ranks++;
            continue;
        }
        if (c >= '1' && c <= '8') {
            int n = c - '0';
            if (rank_squares + n > 8 || sq + n > 64) return -1;
            rank_squares += n;
            sq += n;
            continue;
        }
        int piece = piece_from_fen_char(c);
        if (piece < 0 || rank_squares >= 8 || sq >= 64) return -1;
        if (piece == WKING) {
            if (tmp.kingPos[WHITE_SIDE] >= 0) return -1;
            tmp.kingPos[WHITE_SIDE] = (int8_t)sq;
        } else if (piece == BKING) {
            if (tmp.kingPos[BLACK_SIDE] >= 0) return -1;
            tmp.kingPos[BLACK_SIDE] = (int8_t)sq;
        }
        tmp.board[sq++] = (int8_t)piece;
        rank_squares++;
    }
    if (!any_placement || ranks != 8 || rank_squares != 8 || sq != 64) return -1;
    if (tmp.kingPos[WHITE_SIDE] < 0 || tmp.kingPos[BLACK_SIDE] < 0) return -1;

    while (fen[i] == ' ') i++;
    if (fen[i] != 'w' && fen[i] != 'b') return -1;
    tmp.side = (fen[i] == 'b') ? BLACK_SIDE : WHITE_SIDE;
    i++;
    if (fen[i] != '\0' && fen[i] != ' ') return -1;

    while (fen[i] == ' ') i++;
    if (fen[i] == '\0') {
        *s = tmp;
        return 0;
    }
    if (fen[i] == '-') {
        i++;
    } else {
        while (fen[i] && fen[i] != ' ') {
            switch (fen[i]) {
                case 'K': tmp.castle |= CASTLE_WK; break;
                case 'Q': tmp.castle |= CASTLE_WQ; break;
                case 'k': tmp.castle |= CASTLE_BK; break;
                case 'q': tmp.castle |= CASTLE_BQ; break;
                default: return -1;
            }
            i++;
        }
    }

    while (fen[i] == ' ') i++;
    if (fen[i] == '\0') {
        *s = tmp;
        return 0;
    }
    if (fen[i] == '-') {
        tmp.ep = -1;
        i++;
    } else if (fen[i] >= 'a' && fen[i] <= 'h' &&
               fen[i + 1] >= '1' && fen[i + 1] <= '8') {
        int file = fen[i] - 'a';
        int rank_num = fen[i + 1] - '0';
        int rank = 8 - rank_num; // a8 is square 0
        i += 2;
        if (!on_board(rank, file)) return -1;
        tmp.ep = (int8_t)make_sq(rank, file);
    } else {
        return -1;
    }

    while (fen[i] == ' ') i++;
    if (fen[i] >= '0' && fen[i] <= '9') {
        int hm = 0;
        while (fen[i] >= '0' && fen[i] <= '9') {
            hm = hm * 10 + (fen[i] - '0');
            if (hm > 32767) return -1;
            i++;
        }
        tmp.halfmove = (int16_t)hm;
    }
    while (fen[i] == ' ') i++;
    if (fen[i] >= '0' && fen[i] <= '9') {
        int fm = 0;
        while (fen[i] >= '0' && fen[i] <= '9') {
            fm = fm * 10 + (fen[i] - '0');
            if (fm > 32767) return -1;
            i++;
        }
        tmp.fullmove = (int16_t)fm;
    }

    *s = tmp;
    return 0;
}

static char piece_to_fen_char(int piece) {
    switch (piece) {
        case WPAWN:   return 'P';
        case WKNIGHT: return 'N';
        case WBISHOP: return 'B';
        case WROOK:   return 'R';
        case WQUEEN:  return 'Q';
        case WKING:   return 'K';
        case BPAWN:   return 'p';
        case BKNIGHT: return 'n';
        case BBISHOP: return 'b';
        case BROOK:   return 'r';
        case BQUEEN:  return 'q';
        case BKING:   return 'k';
        default:      return '?';
    }
}

static int fen_putc(char* out, int cap, int n, char c) {
    if (n < 0 || n + 1 >= cap) return -1;
    out[n] = c;
    return n + 1;
}

int position_to_fen(const Position* s, char* out, int cap) {
    if (!s || !out || cap < 16) return -1;
    int n = 0;
    for (int r = 0; r < 8; r++) {
        if (r) {
            n = fen_putc(out, cap, n, '/');
            if (n < 0) return -1;
        }
        int empty = 0;
        for (int f = 0; f < 8; f++) {
            int piece = s->board[make_sq(r, f)];
            if (piece == EMPTY) {
                empty++;
                continue;
            }
            if (empty) {
                n = fen_putc(out, cap, n, (char)('0' + empty));
                if (n < 0) return -1;
                empty = 0;
            }
            n = fen_putc(out, cap, n, piece_to_fen_char(piece));
            if (n < 0) return -1;
        }
        if (empty) {
            n = fen_putc(out, cap, n, (char)('0' + empty));
            if (n < 0) return -1;
        }
    }

    n = fen_putc(out, cap, n, ' ');
    n = fen_putc(out, cap, n, s->side == BLACK_SIDE ? 'b' : 'w');
    n = fen_putc(out, cap, n, ' ');
    if (n < 0) return -1;

    int castle_n = 0;
    if (s->castle & CASTLE_WK) { n = fen_putc(out, cap, n, 'K'); castle_n++; }
    if (s->castle & CASTLE_WQ) { n = fen_putc(out, cap, n, 'Q'); castle_n++; }
    if (s->castle & CASTLE_BK) { n = fen_putc(out, cap, n, 'k'); castle_n++; }
    if (s->castle & CASTLE_BQ) { n = fen_putc(out, cap, n, 'q'); castle_n++; }
    if (castle_n == 0) n = fen_putc(out, cap, n, '-');
    n = fen_putc(out, cap, n, ' ');
    if (n < 0) return -1;

    if (s->ep >= 0 && s->ep < 64) {
        n = fen_putc(out, cap, n, (char)('a' + sq_file(s->ep)));
        n = fen_putc(out, cap, n, (char)('0' + (8 - sq_rank(s->ep))));
    } else {
        n = fen_putc(out, cap, n, '-');
    }
    n = fen_putc(out, cap, n, ' ');
    if (n < 0) return -1;

    char clocks[32];
    int clocks_n = snprintf(clocks, sizeof(clocks), "%d %d",
                            (int)s->halfmove, (int)s->fullmove);
    if (clocks_n < 0) return -1;
    for (int k = 0; k < clocks_n; k++) {
        n = fen_putc(out, cap, n, clocks[k]);
        if (n < 0) return -1;
    }
    if (n < 0 || n >= cap) return -1;
    out[n] = '\0';
    return 0;
}

// ---------- host: UCI encode (extracted verbatim from gpu_fighter.cu) ----------
void move_to_uci(Move move, char* uci) {
    int from = move_from(move), to = move_to(move);
    int flags = move_flags(move), promo = move_promo(move);
    uci[0] = 'a' + sq_file(from);
    uci[1] = '0' + (8 - sq_rank(from));
    uci[2] = 'a' + sq_file(to);
    uci[3] = '0' + (8 - sq_rank(to));
    uci[4] = '\0';
    if (flags == FLAG_PROMO) {
        int pt = piece_type(promo);
        switch (pt) {
            case 5: uci[4] = 'q'; break;
            case 4: uci[4] = 'r'; break;
            case 3: uci[4] = 'b'; break;
            case 2: uci[4] = 'n'; break;
        }
        uci[5] = '\0';
    }
}

// ---------- host perft (matches gpu_fighter.cu h_perft) ----------
long long perft(Position* s, int depth) {
    if (depth == 0) return 1LL;
    Move moves[MAX_MOVES];
    int n = generate_moves(s, moves);
    long long total = 0;
    for (int i = 0; i < n; i++) {
        Position child = *s;
        make_move(&child, moves[i]);
        // Legality: own king must not be in check after the move.
        if (in_check(&child, 1 - child.side)) continue;
        if (depth == 1) total += 1;
        else total += perft(&child, depth - 1);
    }
    return total;
}

} // namespace engine
