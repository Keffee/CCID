"""Run a saved model without loading answer labels."""
from __future__ import annotations
import argparse
import torch
from pathlib import Path
from .checkpoint import restore_model
from .io import load_episodes, write_jsonl
from .memory import load_snapshot
from .runtime import memory_inputs, prediction_record


def run_prediction(checkpoint, backbone, episodes_path, output, *, device="cpu",
                   trust_remote_code=False, memory_path=None, scope="global",
                   as_of=None, top_k=4):
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("choose a new predictions path")
    snapshot = load_snapshot(memory_path) if memory_path else None
    if snapshot and as_of is None:
        raise ValueError("memory requires an explicit as_of date")
    model = restore_model(
        checkpoint, backbone, device=device, trust_remote_code=trust_remote_code,
        expected_memory_sha256=snapshot.digest if snapshot else None,
        expected_memory_policy=dict(scope=scope, top_k=top_k) if snapshot else None)
    _, episodes = load_episodes(episodes_path)
    rows = []
    with torch.inference_mode():
        for item in episodes:
            episode = item.to(device)
            memory, types, features, ids = memory_inputs(
                snapshot, episode, scope=scope, as_of=as_of or "1970-01-01",
                top_k=top_k)
            result = model(episode, memory=memory, memory_types=types, memory_features=features)
            rows.append(prediction_record(
                episode, result, ids, snapshot.digest if snapshot else None))
    write_jsonl(destination, rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--memory")
    parser.add_argument("--scope", default="global")
    parser.add_argument("--as-of")
    parser.add_argument("--memory-top-k", type=int, default=4)
    args = parser.parse_args()
    run_prediction(
        args.checkpoint, args.backbone, args.episodes, args.output,
        device=args.device, trust_remote_code=args.trust_remote_code,
        memory_path=args.memory, scope=args.scope, as_of=args.as_of,
        top_k=args.memory_top_k)


if __name__ == "__main__":
    main()
