"""Validated JSON tools exposed to the model and the workshop UI."""

from __future__ import annotations

from typing import Any, Callable

from checkers.engine import CheckersGame


TOOL_SCHEMAS = [
    {"name": "get_board_state", "description": "Read the current authoritative board, player, and result.", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"name": "get_legal_moves", "description": "List every legal complete move for the current player. Captures are mandatory.", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"name": "get_move_history", "description": "Read the moves played so far in PDN notation.", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"name": "make_move", "description": "Play one complete legal move in PDN square notation, such as 9-13 or 22x15x8.", "parameters": {"type": "object", "properties": {"move": {"type": "string"}}, "required": ["move"], "additionalProperties": False}},
    {"name": "check_game_status", "description": "Check whether the game has ended and which player won or whether it was a draw.", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
]


def qwen_tool_schemas(tool_schemas: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Return the OpenAI-style function objects expected by Qwen's HF template."""
    return [
        {"type": "function", "function": schema}
        for schema in (TOOL_SCHEMAS if tool_schemas is None else tool_schemas)
    ]


class ToolDispatcher:
    def __init__(self, game: CheckersGame):
        self.game = game
        self._handlers: dict[str, Callable[..., Any]] = {
            "get_board_state": self._get_board_state,
            "get_legal_moves": self._get_legal_moves,
            "get_move_history": self._get_move_history,
            "make_move": self._make_move,
            "check_game_status": self._check_game_status,
        }

    def call(self, name: str, arguments: Any) -> dict[str, Any]:
        if name not in self._handlers:
            raise ValueError(f"unknown tool: {name}")
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object")
        handler = self._handlers[name]
        expected = {"move"} if name == "make_move" else set()
        if set(arguments) != expected:
            raise ValueError(f"invalid arguments for {name}: expected keys {sorted(expected)}")
        return handler(**arguments)

    def _get_board_state(self) -> dict[str, Any]:
        return self.game.state()

    def _get_legal_moves(self) -> dict[str, Any]:
        return {"moves": self.game.legal_moves(), "current_player": self.game.current_player()}

    def _get_move_history(self) -> dict[str, Any]:
        return {"moves": list(self.game.history)}

    def _make_move(self, move: str) -> dict[str, Any]:
        return self.game.make_move(move)

    def _check_game_status(self) -> dict[str, Any]:
        return self.game.status()
