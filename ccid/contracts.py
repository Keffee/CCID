"""Typed inference inputs and candidate-aligned debate outputs."""
from dataclasses import dataclass, field
import math
from typing import Optional

import torch
from torch import Tensor


def _finite_float_tensor(value: Tensor, name: str, ndim: int) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


@dataclass
class Episode:
    """One visible request. Targets and outcome metadata are deliberately absent."""

    request_id: str
    candidate_ids: tuple[str, ...]
    candidate_states: Tensor
    history_states: Tensor
    base_scores: Tensor

    def validate(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a nonempty string")
        if not isinstance(self.candidate_ids, tuple) or not self.candidate_ids:
            raise ValueError("candidate_ids must be a nonempty tuple")
        if any(not isinstance(item, str) or not item.strip() for item in self.candidate_ids):
            raise ValueError("candidate identities must be nonempty strings")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate identities must be unique")
        _finite_float_tensor(self.candidate_states, "candidate_states", 2)
        _finite_float_tensor(self.history_states, "history_states", 2)
        _finite_float_tensor(self.base_scores, "base_scores", 1)
        count, width = self.candidate_states.shape
        if count != len(self.candidate_ids) or width < 1:
            raise ValueError("candidate states must align with candidate identities")
        if self.base_scores.shape != (count,):
            raise ValueError("base_scores must have one value per candidate")
        if self.history_states.shape[1] != width:
            raise ValueError("history and candidate state widths must agree")
        if len({self.candidate_states.device, self.history_states.device,
                self.base_scores.device}) != 1:
            raise ValueError("episode tensors must be on one device")

    def to(self, device: torch.device | str) -> "Episode":
        return Episode(
            self.request_id, self.candidate_ids, self.candidate_states.to(device),
            self.history_states.to(device), self.base_scores.to(device),
        )


@dataclass
class ModelConfig:
    input_dim: int = 2048
    state_dim: int = 128
    max_rounds: int = 4
    top_k: int = 5
    disagreement_threshold: float = 0.05
    max_calls: int = 12
    adaptive: bool = True
    interface: str = "latent"
    feedback: bool = True
    role_mode: str = "utility_risk"

    def __post_init__(self) -> None:
        for name in ("input_dim", "state_dim", "max_rounds", "top_k"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_calls) is not int or self.max_calls < 0:
            raise ValueError("max_calls must be a nonnegative integer")
        if (not isinstance(self.disagreement_threshold, (int, float))
                or not math.isfinite(self.disagreement_threshold)
                or not 0 <= self.disagreement_threshold <= 1):
            raise ValueError("disagreement_threshold must be finite and in [0, 1]")
        if type(self.adaptive) is not bool or type(self.feedback) is not bool:
            raise ValueError("adaptive and feedback must be booleans")
        if self.interface not in {"latent", "score_only"}:
            raise ValueError("interface must be latent or score_only")
        if self.role_mode not in {"utility_risk", "duplicate_utility", "unified"}:
            raise ValueError("unsupported role_mode")
        if self.adaptive and self.role_mode != "utility_risk":
            raise ValueError(
                "adaptive stopping requires utility_risk semantics; "
                "use fixed rounds (adaptive=False) for role controls")


@dataclass
class RoleResponse:
    """Rows retain canonical candidate identity, not the current ranking slots."""

    candidate_ids: tuple[str, ...]
    role: str
    round_index: int
    state_version: int
    latent: Tensor
    logits: Tensor

    def validate(
        self, *, candidate_ids: Optional[tuple[str, ...]] = None,
        round_index: Optional[int] = None, state_version: Optional[int] = None,
        state_dim: Optional[int] = None,
    ) -> None:
        if (not isinstance(self.candidate_ids, tuple) or not self.candidate_ids
                or len(set(self.candidate_ids)) != len(self.candidate_ids)
                or any(not isinstance(item, str) or not item for item in self.candidate_ids)):
            raise ValueError("response candidate identities must be nonempty and unique")
        if candidate_ids is not None and self.candidate_ids != candidate_ids:
            raise ValueError("response candidate identities are misaligned")
        if self.role not in {"utility", "risk"}:
            raise ValueError("unknown response role")
        for name in ("round_index", "state_version"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if round_index is not None and self.round_index != round_index:
            raise ValueError("response round metadata is stale")
        if state_version is not None and self.state_version != state_version:
            raise ValueError("response state version is stale")
        _finite_float_tensor(self.latent, "response latent", 2)
        _finite_float_tensor(self.logits, "response logits", 1)
        count = len(self.candidate_ids)
        if self.latent.shape[0] != count or self.latent.shape[1] < 1:
            raise ValueError("response latent rows must align with candidate identities")
        if state_dim is not None and self.latent.shape[1] != state_dim:
            raise ValueError("response latent width is inconsistent")
        if self.logits.shape != (count,):
            raise ValueError("response logits must align with candidate identities")
        if self.latent.device != self.logits.device:
            raise ValueError("response tensors must share a device")


@dataclass
class RoundOutput:
    round_index: int
    incumbent_index: int
    scores: Tensor
    states: Tensor
    utility: RoleResponse
    risk: RoleResponse
    disagreement: float
    calls: int


@dataclass
class DebateOutput:
    candidate_ids: tuple[str, ...]
    initial_scores: Tensor
    scores: Tensor
    rounds: list[RoundOutput] = field(default_factory=list)
    stop_reason: str = "call_budget"
    permitted_calls: int = 0
    realized_calls: int = 0

    @property
    def ranking(self) -> Tensor:
        """Descending indices into candidate_ids, with stable tie ordering."""
        return torch.argsort(self.scores, descending=True, stable=True)
