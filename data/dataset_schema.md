# Trajectory dataset schema

Each JSONL record is one complete agent turn sampled from a game. Games are
assigned wholly to train, validation, or test before their positions are split,
so a game cannot leak across splits.

Required fields:

* `id`: stable trajectory identifier.
* `game_id`: source game identifier, used to preserve split boundaries.
* `variant`: always `english` (American/English draughts).
* `start_fen`: authoritative pydraughts position before the turn.
* `player`: `black` or `red`.
* `messages`: ordered chat turns. Assistant content is always one JSON protocol
  action. `tool` messages are structured results returned by the harness.
* `teacher_move`: complete legal PDN move selected by the heuristic teacher.
* `game_result`: final game result, or `in_progress` when the game was capped.
* `teacher`: transparent heuristic metadata; this is not an LLM explanation.

Stored assistant actions use the harness's internal forms `{"action":"call_tool","tool":NAME,"arguments":{...}}`
and `{"action":"finish_turn"}`. During Qwen2.5 fine-tuning, `call_tool` actions
are rendered through the model's Hugging Face chat template as native
`<tool_call>` messages. Each tool response is a JSON object with
`ok` and either `result` or `error`. Move notation uses numeric dark squares
1–32; `-` marks a quiet move and `x` a capture. Complete multi-jump paths are
represented as one move, such as `9x27`.

`test_positions.jsonl` uses the same position fields without chat messages and
contains `legal_moves` and `teacher_move` for held-out evaluation.
