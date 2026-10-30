"""Evaluate a model on the same held-out positions; no fallback in move rate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.harness import AgentHarness
from checkers.engine import CheckersGame
from evaluation.metrics import summarize_positions


def evaluate_positions(model, positions: list[dict]) -> dict:
    rows = []
    for position in positions:
        game = CheckersGame(initial_fen=position["fen"])
        harness = AgentHarness(model=model, game=game, fallback_enabled=True)
        outcome = harness.play_turn()
        proposals = [event for event in harness.events if event["type"] == "proposal"]
        failures = [event for event in harness.events if event["type"] == "validation_failure"]
        calls = [event for event in harness.events if event["type"] == "tool_call"]
        tool_results = [event for event in harness.events if event["type"] == "tool_result"]
        rows.append({
            "legal_before_fallback": outcome["source"] == "model" and outcome["move"] in position["legal_moves"],
            "task_completed": outcome["source"] == "model",
            "teacher_match": outcome["source"] == "model" and outcome["move"] == position["teacher_move"],
            "valid_output_rate": (len(proposals) - len(failures)) / max(1, len(proposals)),
            "first_tool_correct": bool(calls and calls[0]["tool"] == "get_board_state"),
            "tool_arguments_valid": bool(tool_results) and all(result["ok"] for result in tool_results),
            "used_fallback": outcome["source"] == "fallback",
            "latency_ms": sum(p["latency_ms"] for p in proposals) if proposals else None,
        })
    return {"summary": summarize_positions(rows), "positions": rows}


if __name__ == "__main__":
    raise SystemExit("Load a Hugging Face model and call evaluate_positions(model, positions) from a notebook or script.")
