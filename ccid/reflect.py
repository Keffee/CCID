"""Build a frozen memory snapshot from completed training trajectories."""
import argparse
import json
from pathlib import Path
import torch
from .io import check_label_join, load_episodes, load_labels
from .memory import ReflectionCase, reflect, save_snapshot, score_ambiguity
from .metrics import evaluate_records, transition


def build_memory(episodes_path, predictions_path, labels_path, *, scope,
                 snapshot_id, min_support, expires_on, output):
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("choose a new snapshot path")
    _, episodes = load_episodes(episodes_path, expected_split="train")
    labels = load_labels(labels_path)
    check_label_join(episodes, labels)
    predictions = [json.loads(x) for x in Path(predictions_path).read_text().splitlines()
                   if x.strip()]
    evaluate_records(predictions, labels)
    by_id = {x["request_id"]: x for x in predictions}
    cases = []
    for episode in episodes:
        row = by_id[episode.request_id]
        canonical = list(episode.candidate_ids)
        initial = [canonical[i] for i in torch.argsort(
            episode.base_scores, descending=True, stable=True).tolist()]
        if row.get("candidate_ids") != canonical or row["initial_ranking"] != initial:
            raise ValueError("trajectory candidate identity or initial ranking differs")
        previous = initial
        conflicts, promotions, reversals, weighted = [], [], [], []
        leaders = [initial[0]]
        steps = row["rounds"]
        if not steps:
            continue
        for index, step in enumerate(steps):
            ranking = step["ranking"]
            if (step["round_index"] != index or len(ranking) != len(canonical)
                    or len(set(ranking)) != len(canonical) or set(ranking) != set(canonical)
                    or step["incumbent_id"] != previous[0] or step["calls"] != 3):
                raise ValueError("inconsistent training trajectory")
            utility = torch.tensor(step["utility"], dtype=torch.float32)
            risk = torch.tensor(step["risk"], dtype=torch.float32)
            if (utility.shape != (len(canonical),) or risk.shape != utility.shape
                    or not torch.isfinite(utility).all() or not torch.isfinite(risk).all()
                    or not ((utility >= 0) & (utility <= 1)).all()
                    or not ((risk >= 0) & (risk <= 1)).all()):
                raise ValueError("invalid candidate-aligned role probabilities")
            conflict = float(step["disagreement"])
            if not 0 <= conflict <= 1:
                raise ValueError("invalid trajectory conflict")
            old = canonical.index(previous[0])
            new = canonical.index(ranking[0])
            weights = utility - risk
            weighted.append((weights[:, None] * episode.candidate_states).mean(0))
            weighted.append(episode.candidate_states[new] - episode.candidate_states[old])
            conflicts.append(conflict)
            promotions.append(float(new != old))
            reversals.append(float(new != old and ranking[0] in leaders[:-1]))
            leaders.append(ranking[0])
            previous = ranking
        if (row["ranking"] != previous or row["realized_calls"] != 3 * len(steps)
                or row["realized_calls"] > row["permitted_calls"]):
            raise ValueError("incomplete final trajectory or invalid budget")
        case_type = transition(initial[0], row["ranking"][0],
                               labels[episode.request_id], canonical)
        if case_type == "unreachable":
            continue
        # Only visible messages and states form the prototype; answers diagnose its case.
        vector = torch.stack(weighted).mean(0)
        trajectory = tuple(sum(values) / len(values)
                           for values in (conflicts, promotions, reversals))
        cases.append(ReflectionCase(
            scope, episode.history_states.shape[0], case_type,
            tuple(vector.detach().cpu().tolist()),
            ambiguity=score_ambiguity(episode.base_scores), trajectory=trajectory))
    snapshot = reflect(cases, snapshot_id=snapshot_id, source_split="train",
                       min_support=min_support, expires_on=expires_on)
    save_snapshot(snapshot, destination)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--expires-on", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    snapshot = build_memory(
        args.episodes, args.predictions, args.labels, scope=args.scope,
        snapshot_id=args.snapshot_id, min_support=args.min_support,
        expires_on=args.expires_on, output=args.output)
    print(json.dumps({"lessons": len(snapshot.lessons), "sha256": snapshot.digest}))


if __name__ == "__main__":
    main()
