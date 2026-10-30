"""Bounded model/tool loop with explicit event and fallback accounting."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.protocol import ProtocolError, parse_action
from agent.prompts import SYSTEM_PROMPT, initial_user_message
from agent.tools import TOOL_SCHEMAS, ToolDispatcher
from checkers.engine import CheckersGame


class AgentModel(Protocol):
    def next_action(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str | dict[str, Any]: ...


@dataclass
class AgentHarness:
    model: AgentModel
    game: CheckersGame = field(default_factory=CheckersGame)
    max_tool_calls: int = 6
    max_invalid_outputs: int = 2
    fallback_enabled: bool = True
    events: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    fallback_count: int = 0

    def __post_init__(self) -> None:
        self.tools = ToolDispatcher(self.game)

    def play_turn(self) -> dict[str, Any]:
        if self.game.status()["over"]:
            raise RuntimeError("cannot run an agent turn after the game has ended")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_user_message(self.game.state(), self.game.history)},
        ]
        invalid_outputs = 0
        turn_calls = 0
        while turn_calls < self.max_tool_calls and invalid_outputs <= self.max_invalid_outputs:
            started = time.perf_counter()
            raw = self.model.next_action(messages, TOOL_SCHEMAS)
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.events.append({"type": "proposal", "raw": raw, "latency_ms": elapsed_ms})
            try:
                action = parse_action(raw)
            except ProtocolError as exc:
                invalid_outputs += 1
                self.events.append({"type": "validation_failure", "error": str(exc), "raw": raw})
                messages.append({"role": "user", "content": f"Invalid tool call: {exc}. Return one valid call to a provided tool."})
                continue

            if action["action"] == "finish_turn":
                self.events.append({"type": "finish_turn"})
                break

            turn_calls += 1
            self.tool_calls += 1
            self.events.append({"type": "tool_call", "tool": action["tool"], "arguments": action["arguments"]})
            try:
                result = self.tools.call(action["tool"], action["arguments"])
                result_message = {"ok": True, "result": result}
            except (ValueError, TypeError) as exc:
                result_message = {"ok": False, "error": str(exc)}
            self.events.append({"type": "tool_result", "tool": action["tool"], **result_message})
            messages.append({"role": "assistant", "content": json.dumps(action, sort_keys=True)})
            messages.append({"role": "tool", "name": action["tool"], "content": json.dumps(result_message, sort_keys=True)})
            if action["tool"] == "make_move" and result_message["ok"]:
                return {"move": result["move"], "source": "model", "status": result["status"]}

        if self.fallback_enabled:
            legal_moves = self.game.legal_moves()
            if legal_moves:
                fallback = legal_moves[0]
                result = self.game.make_move(fallback)
                self.fallback_count += 1
                self.events.append({"type": "fallback", "move": fallback, "reason": "model did not complete a legal move within configured limits"})
                return {"move": fallback, "source": "fallback", "status": result["status"]}
        return {"move": None, "source": "none", "status": self.game.status()}
