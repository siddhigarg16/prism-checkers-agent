"""Hugging Face Qwen inference adapter with native Qwen tool-call formatting."""

from __future__ import annotations

import os
from typing import Any

from agent.protocol import ProtocolError, parse_action
from agent.tools import TOOL_SCHEMAS, qwen_tool_schemas


def to_chat_template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map harness actions/results to Qwen's native assistant/tool message format."""
    converted = []
    for message in messages:
        role = message["role"]
        if role == "tool":
            converted.append({
                "role": "tool",
                "name": message.get("name", "tool"),
                "content": str(message.get("content", "")),
            })
            continue
        if role == "assistant":
            content = str(message.get("content", ""))
            try:
                action = parse_action(content)
            except ProtocolError:
                action = None
            if action and action["action"] == "call_tool":
                converted.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": action["tool"],
                            "arguments": action["arguments"],
                        },
                    }],
                })
                continue
        else:
            content = str(message.get("content", ""))
        converted.append({"role": role, "content": content})
    return converted


def format_qwen_tool_chat(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] = TOOL_SCHEMAS,
) -> str:
    """Render a conversation and fail early if its tokenizer lacks Qwen tool support."""
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=qwen_tool_schemas(tools),
        parallel_tool_call=False,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if "<tools>" not in prompt or "<tool_call>" not in prompt:
        raise RuntimeError(
            "This tokenizer chat template did not render Qwen tool definitions. "
            "Use Qwen2.5-Instruct with a Transformers version that supports tools= in apply_chat_template."
        )
    return prompt


class HuggingFaceAgentModel:
    """Generate one native Qwen tool call at a time using Qwen2.5 and optional LoRA."""

    def __init__(self, model_name: str | None = None, adapter_path: str | None = None, max_new_tokens: int = 128):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the optional model dependencies from requirements-ml.txt") from exc

        self.model_name = model_name or os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        self.adapter_path = adapter_path or os.getenv("ADAPTER_PATH")
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        if self.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.model.eval()
        self._torch = torch

    def next_action(self, messages: list[dict[str, Any]], tools=TOOL_SCHEMAS) -> str:
        prompt = format_qwen_tool_chat(
            self.tokenizer,
            to_chat_template_messages(messages),
            add_generation_prompt=True,
            tools=tools,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self._torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, encoded["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
