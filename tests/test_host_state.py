"""Bookkeeping tests for HostState. No GPU and no engine binary."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "combat_shipping" / "live_arbiter"))

from gpu_state import HostState  # noqa: E402


START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class HostStateTest(unittest.TestCase):
    def test_start_fen(self):
        self.assertEqual(HostState().fen(), START)

    def test_opening_and_castle(self):
        hs = HostState()
        for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"):
            hs.apply_uci(uci)
        self.assertEqual(
            hs.fen(),
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4",
        )
        self.assertFalse(hs.wK)
        self.assertFalse(hs.wQ)
        self.assertTrue(hs.bK)
        self.assertTrue(hs.bQ)

    def test_en_passant_capture(self):
        hs = HostState()
        for uci in ("e2e4", "c7c5", "e4e5", "d7d5"):
            hs.apply_uci(uci)
        self.assertEqual(
            hs.fen(),
            "rnbqkbnr/pp2pppp/8/2ppP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        )
        hs.apply_uci("e5d6")
        self.assertEqual(
            hs.fen(),
            "rnbqkbnr/pp2pppp/3P4/2p5/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3",
        )
        self.assertNotIn(35, hs.piecemap)  # d5, the captured pawn (a1 = 0)

    def test_promotion(self):
        hs = HostState(
            piecemap={0: "K", 7: "k", 48: "P"},
            wK=False, wQ=False, bK=False, bQ=False,
            castling="-",
        )
        hs.apply_uci("a7a8q")
        self.assertEqual(
            hs.fen(),
            "Q7/8/8/8/8/8/8/K6k b - - 0 1",
        )

    def test_insufficient_material(self):
        def board(*pieces):
            return HostState(piecemap={i: p for i, p in pieces})

        self.assertTrue(board((0, "K"), (63, "k")).insufficient_material())
        self.assertTrue(board((0, "K"), (1, "N"), (63, "k")).insufficient_material())
        self.assertTrue(board((0, "K"), (2, "B"), (63, "k")).insufficient_material())
        self.assertFalse(board((0, "K"), (1, "Q"), (63, "k")).insufficient_material())
        # Queen plus knight is not a draw. The old predicate returned True
        # whenever any piece was a bishop or knight.
        self.assertFalse(board((0, "K"), (1, "Q"), (2, "N"), (63, "k")).insufficient_material())
        # Same-color bishops, nothing else.
        self.assertTrue(board((0, "K"), (2, "B"), (63, "k"), (61, "b")).insufficient_material())
        # Opposite-color bishops (c1 dark, h1 light) can mate.
        self.assertFalse(board((0, "K"), (2, "B"), (63, "k"), (7, "b")).insufficient_material())
        # Queen plus two same-color bishops is still a win.
        self.assertFalse(
            board((0, "K"), (3, "Q"), (2, "B"), (63, "k"), (61, "b")).insufficient_material()
        )

    def test_threefold_counter(self):
        hs = HostState()
        self.assertEqual(hs.record_position_for_threefold(), 1)
        self.assertEqual(hs.record_position_for_threefold(), 2)
        self.assertEqual(hs.record_position_for_threefold(), 3)

    def test_rejects_bad_uci(self):
        hs = HostState()
        with self.assertRaises(ValueError):
            hs.apply_uci("e2")
        with self.assertRaises(ValueError):
            hs.apply_uci("e2e9")
        with self.assertRaises(ValueError):
            hs.apply_uci("e7e8x")
        with self.assertRaises(ValueError):
            hs.apply_uci("a3a4")  # empty square
        before = hs.fen()
        self.assertEqual(hs.fen(), before)

    def test_castle_requires_a_rook(self):
        hs = HostState(piecemap={4: "K", 60: "k"}, wK=True, wQ=True, bK=False, bQ=False)
        with self.assertRaises(ValueError):
            hs.apply_uci("e1g1")
        self.assertIn(4, hs.piecemap)


if __name__ == "__main__":
    unittest.main()
