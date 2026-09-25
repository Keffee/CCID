"""Target-free inference serialization and frozen memory lookup."""
from __future__ import annotations
import torch
from .memory import retrieve_memory

CASE_TYPES = ("preserved", "corrected", "damaged", "unresolved")


def memory_inputs(snapshot, episode, *, scope, as_of, top_k):
    if snapshot is None:
        return None, None, None, ()
    tokens, ids = retrieve_memory(
        snapshot, scope=scope, history_count=episode.history_states.shape[0],
        as_of=as_of, top_k=top_k,
        input_dim=episode.candidate_states.shape[1], candidate_scores=episode.base_scores,
    )
    entries = {x.lesson_id: x for x in snapshot.lessons}
    types = torch.tensor([CASE_TYPES.index(entries[x].case_type) for x in ids],
                         dtype=torch.long, device=episode.candidate_states.device)
    features = torch.tensor([entries[x].trajectory for x in ids], dtype=torch.float32,
                            device=episode.candidate_states.device).reshape(len(ids), 3)
    return tokens.to(episode.candidate_states.device), types, features, ids


def prediction_record(episode, output, memory_ids=(), snapshot_hash=None):
    def order(scores):
        indices = torch.argsort(scores.detach(), descending=True, stable=True).tolist()
        return [episode.candidate_ids[i] for i in indices]
    return dict(
        request_id=episode.request_id, candidate_ids=list(episode.candidate_ids),
        initial_ranking=order(output.initial_scores),
        ranking=order(output.scores),
        scores=output.scores.detach().float().cpu().tolist(),
        stop_reason=output.stop_reason,
        permitted_calls=output.permitted_calls,
        realized_calls=output.realized_calls,
        memory_ids=list(memory_ids), memory_snapshot_sha256=snapshot_hash,
        rounds=[dict(
            round_index=x.round_index,
            incumbent_id=episode.candidate_ids[x.incumbent_index],
            ranking=order(x.scores),
            utility=x.utility.logits.detach().sigmoid().float().cpu().tolist(),
            risk=x.risk.logits.detach().sigmoid().float().cpu().tolist(),
            disagreement=x.disagreement, calls=x.calls,
        ) for x in output.rounds],
    )
