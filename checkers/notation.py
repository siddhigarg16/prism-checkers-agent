"""Helpers for numeric 1–32 PDN move notation."""


def normalize_move(move: str) -> str:
    """Normalize capture separators for matching, preserving the full path."""
    return move.strip().replace("x", "-").replace("X", "-")
