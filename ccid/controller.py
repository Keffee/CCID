"""Bounded stopping based on ranking stability and challenge/veto conflict."""
import math

import torch
from torch import Tensor

from .contracts import ModelConfig


CALLS_PER_ROUND = 3


def challenge_veto_conflict(utility_logits: Tensor, risk_logits: Tensor,
                            incumbent_index: int) -> float:
    """Maximum joint promotion challenge and veto, excluding the incumbent."""
    if (utility_logits.ndim != 1 or utility_logits.shape != risk_logits.shape
            or utility_logits.numel() == 0):
        raise ValueError("role logits must be aligned nonempty vectors")
    if not bool(torch.isfinite(utility_logits).all() and torch.isfinite(risk_logits).all()):
        raise ValueError("role logits must be finite")
    if not 0 <= incumbent_index < utility_logits.numel():
        raise ValueError("incumbent index is outside the candidate pool")
    if utility_logits.numel() == 1:
        return 0.0
    conflict = utility_logits.detach().sigmoid() * risk_logits.detach().sigmoid()
    challengers = torch.arange(conflict.numel(), device=conflict.device) != incumbent_index
    return float(conflict[challengers].max().item())


class DebateController:
    """Track consecutive stable rounds. Hard round/call limits are caller-owned."""

    def __init__(self, config: ModelConfig, initial_scores: Tensor):
        config.__post_init__()
        if initial_scores.ndim != 1 or initial_scores.numel() == 0:
            raise ValueError("initial scores must be a nonempty vector")
        if not bool(torch.isfinite(initial_scores).all()):
            raise ValueError("initial scores must be finite")
        self.config = config
        self.candidate_count = initial_scores.numel()
        self.top_k = min(config.top_k, self.candidate_count)
        self.previous = self._top(initial_scores)
        self.stable_rounds = 0

    def _top(self, scores: Tensor) -> Tensor:
        return torch.argsort(scores.detach(), descending=True, stable=True)[:self.top_k]

    def observe(self, scores: Tensor, disagreement: float) -> bool:
        if scores.shape != (self.candidate_count,) or not bool(torch.isfinite(scores).all()):
            raise ValueError("round scores must be finite and candidate-aligned")
        if not math.isfinite(disagreement) or not 0 <= disagreement <= 1:
            raise ValueError("disagreement must be finite and in [0, 1]")
        current = self._top(scores)
        stable = (torch.equal(current, self.previous)
                  and disagreement <= self.config.disagreement_threshold)
        self.stable_rounds = self.stable_rounds + 1 if stable else 0
        self.previous = current
        return self.config.adaptive and self.stable_rounds >= 2
