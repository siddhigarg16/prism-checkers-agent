"""Aggregate protocol and match metrics with model/fallback separation."""

from __future__ import annotations

from statistics import mean
from typing import Iterable


def summarize_positions(results: Iterable[dict]) -> dict:
    rows = list(results)
    if not rows:
        return {"positions": 0}
    latencies = [row["latency_ms"] for row in rows if row.get("latency_ms") is not None]
    return {
        "positions": len(rows),
        "valid_output_rate": mean(row["valid_output_rate"] for row in rows),
        "tool_selection_accuracy": mean(row["first_tool_correct"] for row in rows),
        "tool_argument_validity": mean(row["tool_arguments_valid"] for row in rows),
        "legal_move_rate_before_fallback": mean(row["legal_before_fallback"] for row in rows),
        "task_completion_rate": mean(row["task_completed"] for row in rows),
        "teacher_move_agreement": mean(row["teacher_match"] for row in rows),
        "fallback_rate": mean(row["used_fallback"] for row in rows),
        "mean_latency_ms": mean(latencies) if latencies else None,
    }


def summarize_matches(matches: Iterable[dict]) -> dict:
    rows = list(matches)
    if not rows:
        return {"matches": 0}
    outcomes = []
    for row in rows:
        result = row["status"]["result"]
        side = row["model_side"]
        if result == "draw":
            outcomes.append("draw")
        elif result == "in_progress":
            outcomes.append("unfinished")
        else:
            winner = result.removesuffix("_wins")
            outcomes.append("win" if winner == side else "loss")
    return {"matches": len(rows), "wins": outcomes.count("win"), "draws": outcomes.count("draw"),
            "losses": outcomes.count("loss"), "unfinished": outcomes.count("unfinished"),
            "fallbacks": sum(row["fallback_count"] for row in rows),
            "model_moves": sum(row["move_sources"].get("model", 0) for row in rows)}
