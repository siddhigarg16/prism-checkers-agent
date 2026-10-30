"""Parse Qwen tool calls and normalize them to the harness's internal action protocol."""

from __future__ import annotations

import json
import re
from typing import Any


class ProtocolError(ValueError):
    pass


def parse_action(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, str):
        text = raw.strip()
        text = re.sub(r"(?:<\|im_end\|>|</s>|<\|endoftext\|>)\s*$", "", text).strip()
        wrapped = re.fullmatch(r"<tool_call>\s*(\{.*\})\s*</tool_call>", text, flags=re.DOTALL)
        if wrapped:
            text = wrapped.group(1)
        try:
            action = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"output is not valid JSON: {exc.msg}") from exc
    else:
        action = raw
    if not isinstance(action, dict):
        raise ProtocolError("action must be a JSON object")
    # Qwen2.5 emits native function calls as {"name": ..., "arguments": ...}
    # inside <tool_call> tags. Normalize that model-facing format to the
    # harness's stable internal call_tool representation.
    if set(action) == {"name", "arguments"}:
        action = {"action": "call_tool", "tool": action["name"], "arguments": action["arguments"]}
    kind = action.get("action")
    if kind == "finish_turn":
        if set(action) != {"action"}:
            raise ProtocolError("finish_turn accepts only the action field")
        return {"action": kind}
    if kind != "call_tool":
        raise ProtocolError("action must be call_tool or finish_turn")
    if set(action) != {"action", "tool", "arguments"}:
        raise ProtocolError("call_tool requires exactly action, tool, and arguments")
    if not isinstance(action["tool"], str) or not action["tool"]:
        raise ProtocolError("tool must be a non-empty string")
    if not isinstance(action["arguments"], dict):
        raise ProtocolError("arguments must be a JSON object")
    return action
