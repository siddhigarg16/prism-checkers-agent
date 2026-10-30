# Prism Checkers Agent

A tool-using Checkers AI agent built using **Qwen2.5-0.5B-Instruct** and fine-tuned with **LoRA (Low-Rank Adaptation)**.

## Overview

This project explores fine-tuning a language model to interact with an external game environment through structured tool calls.

The agent plays English/American Checkers by communicating with a custom execution harness that validates tool calls, retrieves board states, identifies legal moves, and executes moves using the Pydraughts engine.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Qwen2.5-0.5B-Instruct
- PEFT / LoRA
- Pydraughts
- Kaggle Notebooks

## Key Features

- Tool-calling architecture for AI-driven gameplay.
- Custom execution harness for parsing and validating model actions.
- Legal move generation and execution using Pydraughts.
- LoRA-based fine-tuning of Qwen2.5.
- Training and validation using recorded game trajectories.
- Evaluation of baseline and fine-tuned models.
- Optional match simulation against a heuristic opponent.
- Checkers match replay and visualization.

## Model Fine-Tuning

The project uses LoRA to fine-tune the attention layers of Qwen2.5 while keeping the original model weights frozen.

The training process uses recorded agent trajectories and supervised tool-call prediction.

## Evaluation

The evaluation framework examines:
- Valid tool-call generation.
- Successful completion of turns.
- Legal move execution.
- Agreement with heuristic teacher moves.
- Performance against a heuristic opponent.

## Project Structure

- `agent/` – Model interaction, tool definitions, protocol parsing, and execution harness.
- `checkers/` – Checkers game engine and move logic.
- `data/` – Dataset generation, validation, and game trajectories.
- `evaluation/` – Model evaluation and match simulation.
- `notebooks/` – Fine-tuning and experimentation workflow.

## Running the Project

The training and evaluation workflow is provided in the Jupyter notebook under `notebooks/`.

The notebook is designed for Kaggle GPU environments. Refer to the notebook for dataset setup, model configuration, training, and evaluation instructions.
