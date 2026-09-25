"""Adapter-only checkpoints bound to the external frozen backbone."""
from __future__ import annotations
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import torch
from .backbone import load_decoder
from .model import CCID, ModelConfig


def backbone_fingerprint(decoder) -> str:
    digest = hashlib.sha256()
    # Bind behavior as well as weights, without embedding machine-specific paths.
    ignored = {"_name_or_path", "transformers_version", "_commit_hash"}
    config = {k: v for k, v in decoder.config.to_dict().items() if k not in ignored}
    digest.update(json.dumps(config, sort_keys=True, default=str).encode())
    for name, value in decoder.state_dict().items():
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(cpu.shape), cpu.dtype)).encode())
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(model, path, *, backbone_sha256, epoch, validation,
                    backbone_dtype="float32", memory_sha256=None,
                    training_config=None, memory_policy=None):
    payload = dict(
        format_version=1, model_config=asdict(model.config),
        training_config=training_config, memory_policy=memory_policy,
        parameters={k: v.detach().cpu().clone() for k,v in model.named_parameters()
                    if v.requires_grad},
        backbone_sha256=backbone_sha256, backbone_dtype=backbone_dtype,
        memory_sha256=memory_sha256, epoch=epoch, validation=validation,
    )
    torch.save(payload, path)


def restore_model(checkpoint, backbone_path, *, device="cpu",
                  trust_remote_code=False, expected_memory_sha256=None,
                  expected_memory_policy=None):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    if payload.get("memory_sha256") != expected_memory_sha256:
        raise ValueError("use the same frozen memory snapshot as training")
    if payload.get("memory_policy") != expected_memory_policy:
        raise ValueError("memory scope and retrieval budget differ from training")
    decoder = load_decoder(
        str(backbone_path), dtype=payload["backbone_dtype"],
        trust_remote_code=trust_remote_code)
    if backbone_fingerprint(decoder) != payload["backbone_sha256"]:
        raise ValueError("backbone weights differ from the training checkpoint")
    model = CCID(decoder, ModelConfig(**payload["model_config"]))
    params = {k:v for k,v in model.named_parameters() if v.requires_grad}
    saved = payload["parameters"]
    if set(params) != set(saved):
        raise ValueError("checkpoint parameter set does not match model")
    with torch.no_grad():
        for key, parameter in params.items():
            if saved[key].shape != parameter.shape:
                raise ValueError(f"checkpoint shape mismatch: {key}")
            parameter.copy_(saved[key])
    return model.to(device).eval()
