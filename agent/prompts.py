"""Shared prompt format for generated trajectories and model inference."""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = (
    "You are a checkers agent. Use the provided tools to inspect the authoritative "
    "position, get the legal moves, and submit one complete legal move with make_move. "
    "Use a tool call for every action; never return a direct play action or edit the board."
)


def initial_user_message(state: dict[str, Any], history: list[str]) -> str:
    """Serialize the same initial state used in supervised trajectories."""
    return json.dumps({"position": state, "history": history}, sort_keys=True)
