"""Offline reflection and immutable, condition-matched memory."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import torch

CASE_ACTION = {
    "preserved": "preserve_supported_leader",
    "corrected": "reconsider_supported_promotion",
    "damaged": "recheck_promotion",
    "unresolved": "reassess_disagreement",
}
TRIGGERS = ("short_history", "medium_history", "long_history")
AMBIGUITIES = ("any", "close_scores", "clear_leader")


def score_ambiguity(scores: torch.Tensor) -> str:
    if scores.ndim != 1 or scores.numel() == 0 or not torch.isfinite(scores).all():
        raise ValueError("finite nonempty candidate scores required")
    if scores.numel() == 1:
        return "clear_leader"
    normalized = scores.float().softmax(0).topk(2).values
    return "close_scores" if float(normalized[0] - normalized[1]) <= 0.1 else "clear_leader"


def validate_trajectory(value):
    result = tuple(float(x) for x in value)
    if len(result) != 3 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in result):
        raise ValueError("trajectory requires three finite rates in [0, 1]")
    return result


def history_trigger(count: int) -> str:
    if count < 0:
        raise ValueError("history_count must be nonnegative")
    return TRIGGERS[0] if count <= 3 else TRIGGERS[1] if count <= 7 else TRIGGERS[2]


@dataclass(frozen=True)
class Lesson:
    lesson_id: str
    scope: str
    trigger: str
    case_type: str
    action: str
    confidence: float
    support: int
    vector: tuple[float, ...]
    expires_on: str
    ambiguity: str = "any"
    trajectory: tuple[float, ...] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "trajectory", validate_trajectory(self.trajectory))
        if self.ambiguity not in AMBIGUITIES:
            raise ValueError("unknown visible ambiguity condition")
        object.__setattr__(self, "vector", tuple(float(x) for x in self.vector))
        if not self.lesson_id or not self.scope:
            raise ValueError("lesson identity and scope are required")
        if self.trigger not in TRIGGERS or self.case_type not in CASE_ACTION:
            raise ValueError("unknown visible trigger or case type")
        if self.action != CASE_ACTION[self.case_type]:
            raise ValueError("action does not match case type")
        if not 0 <= self.confidence <= 1 or self.support < 1:
            raise ValueError("invalid confidence or support")
        if not self.vector or not all(math.isfinite(x) for x in self.vector):
            raise ValueError("lesson vector must be finite and nonempty")
        date.fromisoformat(self.expires_on)


@dataclass(frozen=True)
class MemorySnapshot:
    snapshot_id: str
    source_split: str
    lessons: tuple[Lesson, ...]

    def __post_init__(self):
        object.__setattr__(self, "lessons", tuple(self.lessons))
        if not self.snapshot_id or self.source_split != "train":
            raise ValueError("memory requires a named training-only snapshot")
        if len({x.lesson_id for x in self.lessons}) != len(self.lessons):
            raise ValueError("duplicate lesson identity")

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class ReflectionCase:
    scope: str
    history_count: int
    case_type: str
    vector: tuple[float, ...]
    ambiguity: str = "any"
    trajectory: tuple[float, ...] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "trajectory", validate_trajectory(self.trajectory))
        if self.ambiguity not in AMBIGUITIES:
            raise ValueError("unknown visible ambiguity condition")
        object.__setattr__(self, "vector", tuple(float(x) for x in self.vector))
        history_trigger(self.history_count)
        if not self.scope or self.case_type not in CASE_ACTION:
            raise ValueError("invalid reflection case")
        if not self.vector or not all(math.isfinite(x) for x in self.vector):
            raise ValueError("invalid reflection vector")


def reflect(cases: Iterable[ReflectionCase], *, snapshot_id: str,
            source_split: str, min_support: int, expires_on: str) -> MemorySnapshot:
    """Aggregate diagnosed training trajectories into conditional lessons."""
    if source_split != "train" or min_support < 1:
        raise ValueError("reflection requires training cases and positive support")
    date.fromisoformat(expires_on)
    groups = defaultdict(list)
    total = Counter()
    for case in cases:
        trigger = history_trigger(case.history_count)
        groups[(case.scope, trigger, case.case_type, case.ambiguity)].append(case)
        total[(case.scope, trigger, case.ambiguity)] += 1
    lessons = []
    for (scope, trigger, case_type, ambiguity), records in sorted(groups.items()):
        vectors = [case.vector for case in records]
        if len(vectors) < min_support:
            continue
        width = len(vectors[0])
        if any(len(v) != width for v in vectors):
            raise ValueError("reflection vectors have inconsistent widths")
        vector = tuple(sum(v[i] for v in vectors) / len(vectors) for i in range(width))
        key = json.dumps([snapshot_id, scope, trigger, case_type, ambiguity])
        trajectory = tuple(sum(c.trajectory[i] for c in records) / len(records)
                           for i in range(3))
        lessons.append(Lesson(
            hashlib.sha256(key.encode()).hexdigest()[:16], scope, trigger,
            case_type, CASE_ACTION[case_type], len(vectors) / total[(scope, trigger, ambiguity)],
            len(vectors), vector, expires_on, ambiguity, trajectory,
        ))
    return MemorySnapshot(snapshot_id, source_split, tuple(lessons))


def retrieve_memory(snapshot: MemorySnapshot, *, scope: str, history_count: int,
                    as_of: str, top_k: int, input_dim: int,
                    candidate_scores: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Return a copy; inference cannot modify the frozen snapshot."""
    if top_k < 0 or input_dim < 1 or not scope:
        raise ValueError("invalid memory retrieval request")
    cutoff = date.fromisoformat(as_of)
    trigger = history_trigger(history_count)
    ambiguity = score_ambiguity(candidate_scores) if candidate_scores is not None else "any"
    selected = [
        x for x in snapshot.lessons
        if x.scope in (scope, "global") and x.trigger == trigger
        and x.ambiguity in ("any", ambiguity)
        and date.fromisoformat(x.expires_on) >= cutoff
    ]
    selected.sort(key=lambda x: (-x.confidence, -x.support, x.lesson_id))
    selected = selected[:top_k]
    if any(len(x.vector) != input_dim for x in selected):
        raise ValueError("memory vector width does not match candidate input")
    vectors = torch.tensor([x.vector for x in selected], dtype=torch.float32)
    return vectors.reshape(len(selected), input_dim), tuple(x.lesson_id for x in selected)


def save_snapshot(snapshot: MemorySnapshot, path: str | Path):
    Path(path).write_text(json.dumps(
        {"snapshot": asdict(snapshot), "sha256": snapshot.digest},
        indent=2, sort_keys=True) + "\n")


def load_snapshot(path: str | Path) -> MemorySnapshot:
    payload = json.loads(Path(path).read_text())
    value = payload["snapshot"]
    snapshot = MemorySnapshot(
        value["snapshot_id"], value["source_split"],
        tuple(Lesson(**x) for x in value["lessons"]),
    )
    if snapshot.digest != payload["sha256"]:
        raise ValueError("memory snapshot checksum mismatch")
    return snapshot
