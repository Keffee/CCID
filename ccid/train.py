"""Train CCID adapters and role/readout heads on a frozen decoder."""
from __future__ import annotations
import argparse
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import random
from pathlib import Path
import torch
from .backbone import load_decoder
from .checkpoint import backbone_fingerprint, save_checkpoint
from .io import check_label_join, load_episodes, load_labels
from .losses import debate_loss
from .memory import load_snapshot
from .metrics import evaluate_records
from .model import CCID, ModelConfig
from .runtime import memory_inputs, prediction_record


def run_training(config):
    options = config["training"]
    seed = int(options.get("seed", 1))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    epochs, batch_size = int(options["epochs"]), int(options["batch_size"])
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if options["learning_rate"] <= 0 or options.get("grad_clip", 1.0) <= 0:
        raise ValueError("learning rate and gradient clip must be positive")
    output_dir = Path(config["output_dir"])
    if output_dir.exists():
        raise FileExistsError("choose a new output_dir")
    data = config["data"]
    _, train = load_episodes(data["train"], "train")
    _, validation = load_episodes(data["validation"], "validation")
    train_labels, validation_labels = (load_labels(data["train_labels"]),
                                       load_labels(data["validation_labels"]))
    check_label_join(train, train_labels)
    check_label_join(validation, validation_labels)
    if {x.request_id for x in train} & {x.request_id for x in validation}:
        raise ValueError("train and validation request IDs overlap")
    model_config = ModelConfig(**config["model"])
    if model_config.max_calls < 3:
        raise ValueError("training requires a complete debate round")
    backbone = config["backbone"]
    decoder = load_decoder(
        backbone["path"], dtype=backbone.get("dtype", "float32"),
        trust_remote_code=backbone.get("trust_remote_code", False))
    fingerprint = backbone_fingerprint(decoder)
    device = options.get("device", "cuda")
    model = CCID(decoder, model_config).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(options["learning_rate"]),
        weight_decay=float(options.get("weight_decay", 1e-4)))
    memory_config = config.get("memory") or {}
    snapshot = load_snapshot(memory_config["path"]) if memory_config.get("path") else None
    memory_kwargs = dict(scope=memory_config.get("scope", "global"),
                         as_of=memory_config.get("as_of", "1970-01-01"),
                         top_k=memory_config.get("top_k", 4))
    if snapshot is not None and "as_of" not in memory_config:
        raise ValueError("memory requires an explicit as_of date")
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = output_dir / "best.pt"
    best_score = float("-inf")
    history = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start:start+batch_size]
            optimizer.zero_grad(set_to_none=True)
            for index in indices:
                episode = train[index].to(device)
                tokens, types, features, _ = memory_inputs(snapshot, episode, **memory_kwargs)
                output = model(episode, memory=tokens, memory_types=types, memory_features=features)
                loss = debate_loss(
                    output, train_labels[episode.request_id],
                    role_mode=model_config.role_mode,
                    **options.get("loss_weights", {}))
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                (loss / len(indices)).backward()
                loss_sum += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                          options.get("grad_clip", 1.0),
                                          error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        records = []
        with torch.inference_mode():
            for item in validation:
                episode = item.to(device)
                tokens, types, features, ids = memory_inputs(snapshot, episode, **memory_kwargs)
                result = model(episode, memory=tokens, memory_types=types, memory_features=features)
                records.append(prediction_record(
                    episode, result, ids, snapshot.digest if snapshot else None))
        metrics = evaluate_records(records, validation_labels, k=options.get("k", 8))
        history.append(dict(epoch=epoch, loss=loss_sum/len(train), validation=metrics))
        score = metrics["mrr_at_k"]
        if score > best_score:
            best_score = score
            save_checkpoint(
                model, checkpoint, backbone_sha256=fingerprint, epoch=epoch,
                validation=metrics, backbone_dtype=backbone.get("dtype", "float32"),
                memory_sha256=snapshot.digest if snapshot else None,
                memory_policy={k: memory_kwargs[k] for k in ("scope", "top_k")} if snapshot else None,
                training_config={k: options[k] for k in (
                    "seed", "epochs", "batch_size", "learning_rate", "weight_decay",
                    "grad_clip", "k", "loss_weights") if k in options})
        (output_dir/"history.json").write_text(json.dumps(history, indent=2)+"\n")
        print(json.dumps(history[-1]), flush=True)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_training(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
