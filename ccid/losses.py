"""Candidate and listwise supervision, separate from inference inputs."""
import torch
from torch.nn import functional as F


def debate_loss(output, target_id: str, *, utility_weight: float = 1.0,
                risk_weight: float = 1.0, listwise_weight: float = 1.0,
                role_mode: str = "utility_risk"):
    if not output.rounds:
        raise ValueError("training requires at least one complete round")
    if min(utility_weight, risk_weight, listwise_weight) < 0:
        raise ValueError("loss weights must be nonnegative")
    if utility_weight + risk_weight + listwise_weight <= 0:
        raise ValueError("at least one loss weight must be positive")
    if role_mode not in {"utility_risk", "duplicate_utility", "unified"}:
        raise ValueError("unknown role mode")
    sample = output.rounds[0].scores
    labels = torch.tensor([x == target_id for x in output.candidate_ids],
                          dtype=sample.dtype, device=sample.device)
    reachable = bool(labels.sum().item())
    target_index = labels.argmax().reshape(1)
    losses = []
    for step in output.rounds:
        risk_labels = labels[step.incumbent_index] * (1 - labels)
        utility_loss = F.binary_cross_entropy_with_logits(step.utility.logits, labels)
        risk_loss = F.binary_cross_entropy_with_logits(step.risk.logits, risk_labels)
        if role_mode == "duplicate_utility":
            risk_loss = F.binary_cross_entropy_with_logits(step.risk.logits, labels)
        elif role_mode == "unified":
            utility_loss = (utility_loss + F.binary_cross_entropy_with_logits(
                step.utility.logits, risk_labels)) / 2
            risk_loss = (risk_loss + F.binary_cross_entropy_with_logits(
                step.risk.logits, labels)) / 2
        listwise = (F.cross_entropy(step.scores.unsqueeze(0), target_index)
                    if reachable else step.scores.sum() * 0)
        losses.append(utility_weight * utility_loss + risk_weight * risk_loss
                      + listwise_weight * listwise)
    return torch.stack(losses).mean()
