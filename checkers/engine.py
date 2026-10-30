"""Thin authoritative wrapper around pydraughts' English draughts engine."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from draughts import BLACK, WHITE, Board, Move

from checkers.notation import normalize_move


@dataclass
class CheckersGame:
    """One game of English/American draughts (8x8, 1-32 square notation).

    pydraughts is the source of truth for move legality, captures, promotion,
    and game termination. A complete multiple-capture path is one turn.
    """

    initial_fen: str = "startpos"
    board: Any = field(init=False)
    history: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.board = Board(variant="english", fen=self.initial_fen)
        self.history = []

    def current_player(self) -> str:
        return "black" if self.board.turn == BLACK else "red"

    def legal_moves(self) -> list[str]:
        return [move.pdn_move for move in self.board.legal_moves()]

    def make_move(self, move: str) -> dict[str, Any]:
        if not isinstance(move, str):
            raise ValueError("move must be a string in PDN square notation")
        requested = normalize_move(move)
        choices = self.legal_moves()
        accepted = next((choice for choice in choices if normalize_move(choice) == requested), None)
        if accepted is None:
            raise ValueError(f"illegal move {move!r}; legal moves are {choices}")
        self.board.push(Move(self.board, pdn_move=accepted))
        self.history.append(accepted)
        return {"accepted": True, "move": accepted, "status": self.status()}

    def status(self) -> dict[str, Any]:
        winner = self.board.winner()
        if winner == WHITE:
            result = "red_wins"
        elif winner == BLACK:
            result = "black_wins"
        elif winner == 0:
            result = "draw"
        else:
            result = "in_progress"
        return {"over": self.board.is_over(), "result": result, "current_player": self.current_player()}

    def piece_counts(self) -> dict[str, int]:
        fields = self.board.fen.split(":")
        counts = {"red": 0, "black": 0}
        for color, field_text in zip(("red", "black"), fields[1:3]):
            counts[color] = len(re.findall(r"K?\d+", field_text))
        return counts

    def state(self) -> dict[str, Any]:
        return {
            "variant": "english",
            "notation": "PDN 1-32; '-' for quiet moves, 'x' for captures",
            "fen": self.board.fen,
            "board": str(self.board),
            "current_player": self.current_player(),
            "pieces": self.piece_counts(),
            "status": self.status(),
        }
