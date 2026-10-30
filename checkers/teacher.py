"""A transparent one-ply teacher for workshop data and demo games."""

from __future__ import annotations

import random
import re

from draughts import Move


def _material(fen: str) -> tuple[int, int]:
    """Return (black, white) material from English draughts FEN."""
    fields = fen.split(":")
    if len(fields) < 3:
        return 0, 0
    totals = {"W": 0, "B": 0}
    for side, field in zip(("W", "B"), fields[1:3]):
        for token in re.findall(r"K?\d+", field):
            totals[side] += 3 if token.startswith("K") else 2
    return totals["B"], totals["W"]


def _king_count(fen: str) -> int:
    return len(re.findall(r"K\d+", fen))


class HeuristicTeacher:
    """Prefer captures, material gains, and promotion; break ties by seed."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose(self, board) -> tuple[str, dict]:
        candidates = []
        before = _material(board.fen)
        mover = board.turn
        for legal in board.legal_moves():
            trial = board.copy()
            trial.push(Move(trial, board_move=legal.board_move))
            after = _material(trial.fen)
            black_delta = after[0] - before[0]
            white_delta = after[1] - before[1]
            score = (black_delta - white_delta) if mover == 1 else (white_delta - black_delta)
            captured = len(legal.captures)
            score += captured * 10
            score += 3 * max(0, _king_count(trial.fen) - _king_count(board.fen))
            candidates.append((score, legal.pdn_move))
        if not candidates:
            raise ValueError("teacher cannot choose from a position without legal moves")
        best_score = max(score for score, _ in candidates)
        best = [move for score, move in candidates if score == best_score]
        selected = self.rng.choice(best)
        return selected, {"heuristic_score": best_score, "tied_best_moves": len(best)}
