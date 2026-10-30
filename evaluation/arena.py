"""Run complete matches without mixing model choices and teacher/fallback moves."""

from __future__ import annotations

from typing import Any

from agent.harness import AgentHarness
from checkers.engine import CheckersGame
from checkers.teacher import HeuristicTeacher


def play_match(model: Any, model_side: str = "black", seed: int = 0, max_plies: int = 240) -> dict:
    if model_side not in {"black", "red"}:
        raise ValueError("model_side must be 'black' or 'red'")
    game = CheckersGame()
    teacher = HeuristicTeacher(seed)
    harness = AgentHarness(model=model, game=game)
    events = []
    move_sources = {"model": 0, "teacher": 0, "fallback": 0}
    for ply in range(max_plies):
        if game.status()["over"]:
            break
        if game.current_player() == model_side:
            old_event_count = len(harness.events)
            result = harness.play_turn()
            new_events = harness.events[old_event_count:]
            events.extend({"ply": ply + 1, **event} for event in new_events)
            if result["move"]:
                move_sources[result["source"]] = move_sources.get(result["source"], 0) + 1
        else:
            move, metadata = teacher.choose(game.board)
            result = game.make_move(move)
            events.append({"ply": ply + 1, "type": "teacher_move", "move": move, "teacher": metadata})
            move_sources["teacher"] += 1
    status = game.status()
    return {"variant": "english", "model_side": model_side, "plies": len(game.history), "status": status,
            "moves": list(game.history), "move_sources": move_sources, "fallback_count": harness.fallback_count,
            "tool_calls": harness.tool_calls, "events": events, "final_state": game.state()}
