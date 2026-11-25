"""gpu_state.py — host-side BOOKKEEPING over GPU position state.

This module owns the ancillary state the dojo_ref ABI does not expose
directly (castling rights, en-passant target, halfmove/fullmove
counters, repetition history). All chess COMPUTE — legality, terminal
detection, make-move on a board representation — still runs on the
GPU via dojo_ref.Position.

The job here is just:
  - Track piece map / castling / ep / clocks alongside dojo_ref.Position
  - Emit a FEN string for the agent (fighters take FEN on stdin)
  - Emit a board-key for threefold-repetition bookkeeping
  - Detect insufficient-material (a state-format predicate, identical
    to the arbiter's host-side check)

It does NOT enumerate legal moves, it does NOT decide check / mate /
stalemate, it does NOT make moves on the GPU board. Those are
GPU-only: pos.legal_moves(), pos.is_checkmate(), pos.is_stalemate(),
pos.make_move(uci).

The piece-map & castling-rights logic mirrors arbiter chess-engine.js
exactly so that both arbiters consume the SAME FEN at every ply.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


FILES = "abcdefgh"


def _sq_uci_to_idx(uci_sq: str) -> int:
    """Host bookkeeping index: a1 -> 0, h8 -> 63.

    This is not the engine mailbox. engine_types.h stores a8 at index 0.
    HostState only exchanges FEN and UCI with the engine, never raw indices.
    """
    return (ord(uci_sq[1]) - ord("1")) * 8 + (ord(uci_sq[0]) - ord("a"))


def _initial_piecemap() -> dict[int, str]:
    m: dict[int, str] = {}
    back = ["R", "N", "B", "Q", "K", "B", "N", "R"]
    for f in range(8):
        m[f] = back[f]              # rank 1, white back
        m[8 + f] = "P"              # rank 2, white pawns
        m[48 + f] = "p"             # rank 7, black pawns
        m[56 + f] = back[f].lower() # rank 8, black back
    return m


@dataclass
class HostState:
    """Side-channel state the GPU ABI doesn't expose. dojo_ref.Position
    remains the authority on legality/terminal; this is bookkeeping for
    FEN formatting and arbiter-equivalent draw checks."""

    piecemap: dict[int, str] = field(default_factory=_initial_piecemap)
    side: str = "w"
    castling: str = "KQkq"
    ep: str = "-"
    halfmove: int = 0
    fullmove: int = 1
    history: Counter = field(default_factory=Counter)

    # Track castling-right losses via flags so we don't have to keep
    # reparsing 'castling'.
    wK: bool = True
    wQ: bool = True
    bK: bool = True
    bQ: bool = True

    # ---- FEN emission (matches arbiter boardToFen) ---------------------

    def fen(self) -> str:
        """Build a FEN identical to arbiter's boardToFen output."""
        rows = []
        for r1 in range(8, 0, -1):  # rank 8 .. 1 (top to bottom in FEN)
            row = ""
            empty = 0
            for f in range(8):
                sq = (r1 - 1) * 8 + f
                p = self.piecemap.get(sq)
                if p is None:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += p
            if empty:
                row += str(empty)
            rows.append(row)
        placement = "/".join(rows)
        cr = ""
        if self.wK: cr += "K"
        if self.wQ: cr += "Q"
        if self.bK: cr += "k"
        if self.bQ: cr += "q"
        if not cr: cr = "-"
        return f"{placement} {self.side} {cr} {self.ep} {self.halfmove} {self.fullmove}"

    # ---- Threefold key (arbiter getBoardKey) ---------------------------

    def board_key(self) -> str:
        """Position+side+castling+ep, matching arbiter's getBoardKey.
        Used only for threefold-repetition counting; not a chess decision."""
        flat = []
        for r1 in range(8, 0, -1):
            for f in range(8):
                sq = (r1 - 1) * 8 + f
                flat.append(self.piecemap.get(sq, "."))
        cr = ""
        if self.wK: cr += "K"
        if self.wQ: cr += "Q"
        if self.bK: cr += "k"
        if self.bQ: cr += "q"
        if not cr: cr = "-"
        return "".join(flat) + self.side + cr + self.ep

    # ---- Insufficient material (arbiter insufficientMaterial) ----------

    def insufficient_material(self) -> bool:
        """True when checkmate is impossible for either side.

        K vs K, K+N vs K, K+B vs K, and K+B vs K+B with the bishops on the
        same color. A queen, rook, pawn, or knight alongside anything else
        is not an automatic draw — the previous check treated Q+N vs K as
        insufficient because it only looked for any bishop or knight.
        """
        pieces = list(self.piecemap.values())
        if sum(1 for p in pieces if p.lower() == "k") != 2:
            return False
        others = [p for p in pieces if p.lower() != "k"]
        if not others:
            return True
        if any(p.lower() in ("p", "r", "q") for p in others):
            return False
        if len(others) == 1 and others[0].lower() in ("b", "n"):
            return True
        if len(others) == 2 and all(p.lower() == "b" for p in others):
            bishops = [sq for sq, p in self.piecemap.items() if p.lower() == "b"]
            c1 = ((bishops[0] >> 3) + (bishops[0] & 7)) % 2
            c2 = ((bishops[1] >> 3) + (bishops[1] & 7)) % 2
            return c1 == c2
        return False

    # ---- Apply a UCI move to host-side state ---------------------------

    def apply_uci(self, uci: str) -> None:
        """Update host bookkeeping after a UCI move that the referee has
        already validated and applied. This mirrors the move on the
        side-channel state; it does not decide legality.
        """
        if not isinstance(uci, str) or len(uci) < 4 or len(uci) > 5:
            raise ValueError(f"invalid UCI move: {uci!r}")
        if (uci[0] not in FILES or uci[2] not in FILES or
                uci[1] not in "12345678" or uci[3] not in "12345678"):
            raise ValueError(f"invalid UCI move: {uci!r}")
        promo = ""
        if len(uci) == 5:
            promo = uci[4]
            if promo not in "qrbn":
                raise ValueError(f"invalid promotion piece: {uci!r}")
        f_sq = _sq_uci_to_idx(uci[0:2])
        t_sq = _sq_uci_to_idx(uci[2:4])
        piece = self.piecemap.get(f_sq)
        if not piece:
            raise ValueError(f"no piece on {uci[0:2]}")
        mover_white = self.side == "w"

        is_pawn = piece.lower() == "p"
        # Capture flag: target square occupied OR pawn-diagonal to empty
        # (en passant). Halfmove resets on capture or pawn move.
        is_capture = t_sq in self.piecemap
        if is_pawn and (t_sq & 7) != (f_sq & 7) and not is_capture:
            is_capture = True

        # ---- Castling: king moves 2 files ----
        is_castle = piece in ("K", "k") and abs((t_sq & 7) - (f_sq & 7)) == 2
        if is_castle:
            rank_off = f_sq & ~7
            df = (t_sq & 7) - (f_sq & 7)
            if df > 0:
                rook_from, rook_to = rank_off + 7, rank_off + 5
            else:
                rook_from, rook_to = rank_off + 0, rank_off + 3
            rook = self.piecemap.get(rook_from)
            if rook is None:
                raise ValueError(f"castle without a rook for {uci}")
            del self.piecemap[f_sq]
            self.piecemap[t_sq] = piece
            del self.piecemap[rook_from]
            self.piecemap[rook_to] = rook
        else:
            # ---- En passant capture ----
            if is_pawn and (t_sq & 7) != (f_sq & 7) and t_sq not in self.piecemap:
                ep_target = (f_sq & ~7) | (t_sq & 7)
                self.piecemap.pop(ep_target, None)
            # ---- Normal move ----
            if f_sq in self.piecemap:
                del self.piecemap[f_sq]
            if promo:
                self.piecemap[t_sq] = promo.upper() if mover_white else promo.lower()
            else:
                self.piecemap[t_sq] = piece

        # ---- Update castling rights ----
        # King move loses both rights for that side.
        if piece == "K":
            self.wK = False; self.wQ = False
        elif piece == "k":
            self.bK = False; self.bQ = False
        # Rook move/capture from corners.
        for sq in (f_sq, t_sq):
            if sq == _sq_uci_to_idx("a1"): self.wQ = False
            elif sq == _sq_uci_to_idx("h1"): self.wK = False
            elif sq == _sq_uci_to_idx("a8"): self.bQ = False
            elif sq == _sq_uci_to_idx("h8"): self.bK = False

        # ---- Update en-passant target ----
        # Set ONLY if pawn moved 2 squares; else clear.
        if is_pawn and abs((t_sq >> 3) - (f_sq >> 3)) == 2:
            mid_rank = (f_sq >> 3) + ((t_sq >> 3) - (f_sq >> 3)) // 2
            ep_idx = mid_rank * 8 + (f_sq & 7)
            self.ep = FILES[ep_idx & 7] + str((ep_idx >> 3) + 1)
        else:
            self.ep = "-"

        # ---- Halfmove clock ----
        if is_capture or is_pawn:
            self.halfmove = 0
        else:
            self.halfmove += 1

        # ---- Fullmove counter ----
        if self.side == "b":
            self.fullmove += 1

        # ---- Side to move ----
        self.side = "b" if self.side == "w" else "w"

    def record_position_for_threefold(self) -> int:
        """Increment the repetition counter for the current position
        and return the new count."""
        k = self.board_key()
        self.history[k] += 1
        return self.history[k]
