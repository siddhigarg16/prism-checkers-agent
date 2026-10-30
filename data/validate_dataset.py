"""Validate trajectory JSONL records and replay every teacher action."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.protocol import parse_action
from checkers.engine import CheckersGame


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise ValueError(f"missing dataset file: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: each row must be an object")
        rows.append(row)
    return rows


def validate_trajectory(row: dict) -> None:
    required = {"id", "game_id", "variant", "start_fen", "player", "messages", "teacher_move", "game_result", "teacher"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"{row.get('id', '<unknown>')}: missing keys {sorted(missing)}")
    if row["variant"] != "english":
        raise ValueError(f"{row['id']}: unsupported variant {row['variant']!r}")
    game = CheckersGame(initial_fen=row["start_fen"])
    if row["player"] != game.current_player():
        raise ValueError(f"{row['id']}: player does not match start_fen")
    if not isinstance(row["messages"], list):
        raise ValueError(f"{row['id']}: messages must be a list")
    actions = []
    for message in row["messages"]:
        if message.get("role") == "assistant":
            try:
                actions.append(parse_action(message["content"]))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{row['id']}: invalid assistant action: {exc}") from exc
    move_actions = [a for a in actions if a["action"] == "call_tool" and a["tool"] == "make_move"]
    if not move_actions or move_actions[-1]["arguments"].get("move") != row["teacher_move"]:
        raise ValueError(f"{row['id']}: final assistant move action does not match teacher_move")
    if row["teacher_move"] not in game.legal_moves():
        raise ValueError(f"{row['id']}: teacher move is illegal for start_fen")


def validate_positions(rows: list[dict]) -> None:
    for row in rows:
        game = CheckersGame(initial_fen=row["fen"])
        if row.get("legal_moves") != game.legal_moves():
            raise ValueError(f"{row.get('id')}: legal_moves do not match the engine")
        if row.get("teacher_move") not in game.legal_moves():
            raise ValueError(f"{row.get('id')}: held-out target is not legal")


def validate_directory(directory: Path) -> dict[str, int]:
    split_paths = {
        "train": directory / "train_trajectories.jsonl",
        "validation": directory / "validation_trajectories.jsonl",
        "test": directory / "test_trajectories.jsonl",
    }
    rows = {split: read_jsonl(path) for split, path in split_paths.items()}
    game_sets = {}
    for split, split_rows in rows.items():
        for row in split_rows:
            validate_trajectory(row)
        game_sets[split] = {row["game_id"] for row in split_rows}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = game_sets[first] & game_sets[second]
        if overlap:
            raise ValueError(f"games leak across {first} and {second}: {sorted(overlap)}")
    positions = read_jsonl(directory / "test_positions.jsonl")
    validate_positions(positions)
    return {split: len(split_rows) for split, split_rows in rows.items()} | {"test_positions": len(positions)}


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    print(json.dumps(validate_directory(args.data_dir), indent=2))
