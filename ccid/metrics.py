"""Complete-denominator ranking and transition metrics."""
from collections import Counter


def transition(initial, final, target, candidate_ids):
    if target not in candidate_ids:
        return "unreachable"
    if initial == target:
        return "preserved" if final == target else "damaged"
    return "corrected" if final == target else "unresolved"


def evaluate_records(rows, labels, k=8):
    if k < 1 or not rows:
        raise ValueError("positive k and nonempty predictions required")
    ids = [row["request_id"] for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(labels):
        raise ValueError("predictions and labels must have an exact unique join")
    counts = Counter()
    hit1 = hitk = reciprocal = covered = calls = 0
    for row in rows:
        ranking = row["ranking"]
        initial = row["initial_ranking"]
        if (not ranking or len(ranking) != len(set(ranking))
                or len(initial) != len(set(initial)) or set(ranking) != set(initial)):
            raise ValueError("prediction must preserve the complete candidate set")
        target = labels[row["request_id"]]
        rank = ranking.index(target) + 1 if target in ranking else None
        hit1 += rank == 1
        hitk += rank is not None and rank <= k
        reciprocal += 1 / rank if rank is not None and rank <= k else 0
        covered += rank is not None
        counts[transition(initial[0], ranking[0], target, ranking)] += 1
        calls += row["realized_calls"]
    n = len(rows)
    return dict(contexts=n, k=k, pass_at_1=hit1/n, recall_at_k=hitk/n,
                mrr_at_k=reciprocal/n, coverage=covered/n,
                mean_calls=calls/n, transitions=dict(counts))
