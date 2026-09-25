# CCID

Candidate-Conditioned Implicit Debate for recommendation.

CCID keeps a continuous state for each candidate. Utility and risk roles assess the current ranking, an adjudicator updates the ranking and states, and the roles read those updates in the next round. A controller stops after two stable rounds or when the call budget runs out. An optional frozen memory snapshot supplies lessons from earlier training trajectories.

## Install

Use Python 3.10 or later and install PyTorch for your machine, then:

```bash
pip install -e .
```

## What you need

The backbone must be a local Hugging Face decoder that accepts `inputs_embeds` and returns `last_hidden_state`. The included adapter supports the Qwen3-style decoder used by OneRec. It loads local files only; a different backbone may need an adapter in `ccid/backbone.py`.

Each episode file is a PyTorch dictionary with `split` (`train`, `validation`, or `test`) and an `episodes` list. Each episode contains:

| Field | Shape / type |
| --- | --- |
| `request_id` | Unique string |
| `candidate_ids` | List of unique candidate IDs in proposer order |
| `candidate_states` | Float tensor `[K, D]`; candidate-aligned proposer states |
| `history_states` | Float tensor `[H, D]`; visible history, possibly empty |
| `base_scores` | Float tensor `[K]`; higher scores rank first |

Lists may have different lengths. Supply valid candidates only, without padding or duplicate IDs. Inputs must use history available before the request. Train and validation requests must be disjoint.

Keep supervision in separate JSONL files with `request_id` and `target_id`. Include every request, even when its target is absent from the candidate pool.

## Train

Edit the paths and device in `configs/train.json`, then run:

```bash
python -m ccid.train --config configs/train.json
```

Training updates the communication adapters, role heads, and adjudicator readouts. The backbone stays frozen. Each round uses utility supervision, incumbent-dependent damage-risk supervision, and listwise ranking loss. Validation MRR selects `best.pt`; `history.json` records losses and complete-denominator metrics.

The batch size controls gradient accumulation across requests. Candidates within a request are processed together. Use a new output directory for each run.

## Predict and evaluate

Prediction does not read answer labels:

```bash
python -m ccid.predict \
  --checkpoint runs/ccid/best.pt \
  --backbone external/OneRec-1.7B \
  --episodes inputs/test.pt \
  --output runs/ccid/test_predictions.jsonl \
  --device cuda

python -m ccid.evaluate \
  --predictions runs/ccid/test_predictions.jsonl \
  --labels inputs/test_labels.jsonl \
  --output runs/ccid/test_metrics.json \
  --k 8
```

The checkpoint contains trainable parameters, not external backbone weights. Loading checks the frozen backbone weights and model configuration against training. Evaluation includes all requests and reports Pass@1, MRR, recall, coverage, corrections, damage, and realized calls.

## Reflection memory

Run prediction on the training episodes, then build a snapshot from those completed trajectories:

```bash
python -m ccid.reflect \
  --episodes inputs/train.pt \
  --predictions runs/ccid/train_predictions.jsonl \
  --labels inputs/train_labels.jsonl \
  --scope recommendation \
  --snapshot-id memory-v1 \
  --min-support 5 \
  --expires-on 2030-01-01 \
  --output runs/memory-v1.json
```

Set an appropriate expiry date for your run. Lessons summarize preserved, corrected, damaged, and unresolved decisions within a scope, history-length condition, and initial score ambiguity. They aggregate role messages, disagreement, promotions, and reversals into a revision-state prototype and action type, without answer IDs. The included reflection operator uses deterministic aggregation.

To train with memory, add this object to the training config and choose a new output directory:

```json
"memory": {
  "path": "runs/memory-v1.json",
  "scope": "recommendation",
  "as_of": "2026-09-26",
  "top_k": 4
}
```

Use the request cutoff date for `as_of`. For prediction, pass the same snapshot with `--memory`, `--scope`, and `--as-of`; keep the scope and retrieval count used during training. Retrieval happens once per request; every round reads the same snapshot and never writes to it.
