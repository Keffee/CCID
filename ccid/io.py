"""Small readers for the model's input contract; no dataset preparation."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from .contracts import Episode

EPISODE_FIELDS = {
    "request_id", "candidate_ids", "candidate_states",
    "history_states", "base_scores",
}


def load_episodes(path: str | Path, expected_split: str | None = None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if set(payload) != {"split", "episodes"}:
        raise ValueError("episode file must contain only split and episodes")
    split = payload["split"]
    if split not in {"train", "validation", "test"}:
        raise ValueError("unknown split")
    if expected_split is not None and split != expected_split:
        raise ValueError("unexpected split")
    episodes = []
    for row in payload["episodes"]:
        if set(row) != EPISODE_FIELDS:
            raise ValueError("unexpected episode fields; keep labels in a separate file")
        if not isinstance(row["request_id"], str) or not row["request_id"]:
            raise ValueError("request_id must be a nonempty string")
        if any(not isinstance(x, str) or not x for x in row["candidate_ids"]):
            raise ValueError("candidate_ids must be nonempty strings")
        episode = Episode(
            row["request_id"], tuple(row["candidate_ids"]),
            torch.as_tensor(row["candidate_states"]).float(),
            torch.as_tensor(row["history_states"]).float(),
            torch.as_tensor(row["base_scores"]).float(),
        )
        episode.validate()
        episodes.append(episode)
    if not episodes or len({x.request_id for x in episodes}) != len(episodes):
        raise ValueError("episode file is empty or contains duplicate request IDs")
    return split, episodes


def load_labels(path: str | Path) -> dict[str, str]:
    labels = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != {"request_id", "target_id"}:
            raise ValueError("label rows require only request_id and target_id")
        if any(not isinstance(row[k], str) or not row[k] for k in row):
            raise ValueError("label IDs must be nonempty strings")
        if row["request_id"] in labels:
            raise ValueError("duplicate label request ID")
        labels[row["request_id"]] = row["target_id"]
    if not labels:
        raise ValueError("empty label file")
    return labels


def check_label_join(episodes, labels):
    if {x.request_id for x in episodes} != set(labels):
        raise ValueError("labels must exactly match the complete episode set")


def write_jsonl(path: str | Path, rows):
    with Path(path).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
