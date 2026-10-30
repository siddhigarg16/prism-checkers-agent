"""Generate reproducible teacher-guided tool-use trajectories."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.prompts import SYSTEM_PROMPT, initial_user_message
from checkers.engine import CheckersGame
from checkers.teacher import HeuristicTeacher

CURATED_POSITIONS = [
    ("forced-capture", "B:W14:B9"),
    ("multi-jump", "B:W14,23:B9"),
    ("promotion", "B:W30:B25"),
    ("endgame", "B:W31:B22"),
    ("multiple-options", "B:W30,31:B22,23"),
]
HELDOUT_POSITIONS = [
    ("forced-capture", "B:W15:B10"),
    ("multi-jump", "B:W6,14:B1"),
    ("promotion", "B:W31:B26"),
    ("endgame", "B:W31:B24"),
    ("multiple-options", "B:W30,32:B23,24"),
]


def _tool_result(name: str, result: dict) -> dict:
    return {"role": "tool", "name": name, "content": json.dumps({"ok": True, "result": result}, sort_keys=True)}


def _assistant_tool(name: str, arguments: dict) -> dict:
    return {"role": "assistant", "content": json.dumps({"action": "call_tool", "tool": name, "arguments": arguments}, sort_keys=True)}


def create_game(game_id: str, seed: int, max_plies: int = 120) -> tuple[list[dict], list[dict], str]:
    game = CheckersGame()
    teacher = HeuristicTeacher(seed)
    trajectories, positions = [], []
    for ply in range(max_plies):
        if game.status()["over"]:
            break
        state = game.state()
        legal_moves = game.legal_moves()
        move, teacher_meta = teacher.choose(game.board)
        traj_id = f"{game_id}-t{ply:03d}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_user_message(state, game.history)},
            _assistant_tool("get_board_state", {}),
            _tool_result("get_board_state", state),
            _assistant_tool("get_legal_moves", {}),
            _tool_result("get_legal_moves", {"moves": legal_moves, "current_player": game.current_player()}),
            _assistant_tool("make_move", {"move": move}),
        ]
        played = game.make_move(move)
        messages.append(_tool_result("make_move", played))
        messages.append({"role": "assistant", "content": json.dumps({"action": "finish_turn"})})
        trajectories.append({
            "id": traj_id, "game_id": game_id, "variant": "english", "start_fen": state["fen"],
            "player": state["current_player"], "messages": messages, "teacher_move": move,
            "game_result": "in_progress", "teacher": teacher_meta,
        })
        positions.append({"id": traj_id, "game_id": game_id, "variant": "english", "fen": state["fen"],
                          "player": state["current_player"], "legal_moves": legal_moves, "teacher_move": move})
    result = game.status()["result"]
    for record in trajectories:
        record["game_result"] = result
    return trajectories, positions, result


def create_curated_record(scenario: str, fen: str, seed: int) -> tuple[dict, dict]:
    game = CheckersGame(initial_fen=fen)
    teacher_move, teacher_meta = HeuristicTeacher(seed).choose(game.board)
    state = game.state()
    legal_moves = game.legal_moves()
    game_id = f"curated-{scenario}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_user_message(state, [])},
    ]
    # Include a recoverable tool error so the learner sees the error/result loop.
    if scenario == "forced-capture":
        messages.append(_assistant_tool("make_move", {"move": "1-2"}))
        messages.append({"role": "tool", "name": "make_move", "content": json.dumps({"ok": False, "error": "illegal move '1-2'; forced capture is available"})})
    messages.extend([
        _assistant_tool("get_board_state", {}), _tool_result("get_board_state", state),
        _assistant_tool("get_legal_moves", {}),
        _tool_result("get_legal_moves", {"moves": legal_moves, "current_player": game.current_player()}),
        _assistant_tool("make_move", {"move": teacher_move}),
    ])
    messages.append(_tool_result("make_move", game.make_move(teacher_move)))
    messages.append({"role": "assistant", "content": json.dumps({"action": "finish_turn"})})
    record = {"id": f"{game_id}-t000", "game_id": game_id, "variant": "english", "start_fen": state["fen"],
              "player": state["current_player"], "messages": messages, "teacher_move": teacher_move,
              "game_result": "in_progress", "teacher": teacher_meta, "scenario": scenario}
    position = {"id": record["id"], "game_id": game_id, "variant": "english", "fen": state["fen"],
                "player": state["current_player"], "legal_moves": legal_moves, "teacher_move": teacher_move,
                "scenario": scenario}
    return record, position


def generate(games: int, seed: int, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = list(range(games))
    random.Random(seed).shuffle(ids)
    train_end = max(1, int(games * 0.8))
    val_end = max(train_end, int(games * 0.9))
    split_by_game = {f"game-{n:04d}": ("train" if index < train_end else "validation" if index < val_end else "test")
                     for index, n in enumerate(ids)}
    records = {"train": [], "validation": [], "test": []}
    positions = {"validation": [], "test": []}
    for n in range(games):
        game_id = f"game-{n:04d}"
        game_records, game_positions, _ = create_game(game_id, seed + n)
        split = split_by_game[game_id]
        if split == "train":
            records[split].extend(game_records)
        else:
            records[split].extend(game_records)
            positions[split].extend(game_positions)
    for n, (scenario, fen) in enumerate(CURATED_POSITIONS):
        record, _ = create_curated_record(scenario, fen, seed + 10_000 + n)
        records["train"].append(record)
    heldout_curated = []
    for scenario, fen in HELDOUT_POSITIONS:
        game = CheckersGame(initial_fen=fen)
        target, _ = HeuristicTeacher(seed + 20_000).choose(game.board)
        heldout_curated.append({"id": f"heldout-{scenario}", "game_id": f"heldout-{scenario}",
                                "variant": "english", "fen": game.board.fen,
                                "player": game.current_player(), "legal_moves": game.legal_moves(),
                                "teacher_move": target, "scenario": scenario})
    names = {"train": "train_trajectories.jsonl", "validation": "validation_trajectories.jsonl", "test": "test_trajectories.jsonl"}
    counts = {}
    for split, name in names.items():
        path = output_dir / name
        with path.open("w", encoding="utf-8") as handle:
            for row in records[split]:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        counts[split] = len(records[split])
    with (output_dir / "test_positions.jsonl").open("w", encoding="utf-8") as handle:
        for row in positions["test"] + heldout_curated:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    if args.games < 3:
        parser.error("at least 3 games are required to create train/validation/test splits")
    print(json.dumps(generate(args.games, args.seed, args.output_dir), indent=2))
