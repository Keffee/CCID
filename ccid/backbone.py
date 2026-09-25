"""Frozen external decoder with differentiable continuous input embeddings."""
from typing import Optional

import torch
from torch import Tensor, nn


class FrozenDecoder(nn.Module):
    """Keep the external decoder in eval mode without disabling input gradients.

    Each batch row is an independent candidate sequence. The adapter never
    concatenates candidates into one attention sequence.
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        config = getattr(decoder, "config", None)
        width = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
        if not isinstance(width, int) or width < 1:
            raise ValueError("decoder config must expose hidden_size or n_embd")
        self.decoder = decoder
        self.hidden_size = width
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> "FrozenDecoder":
        super().train(False)
        return self

    def forward(
        self, inputs_embeds: Tensor, attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.hidden_size:
            raise ValueError("decoder inputs must have shape [batch, length, hidden_size]")
        if not inputs_embeds.is_floating_point() or not bool(torch.isfinite(inputs_embeds).all()):
            raise ValueError("decoder embeddings must be finite floating-point values")
        batch, length, _ = inputs_embeds.shape
        if batch < 1 or length < 1:
            raise ValueError("decoder input batch and length must be nonempty")
        if attention_mask is None:
            attention_mask = torch.ones((batch, length), dtype=torch.long,
                                        device=inputs_embeds.device)
        if attention_mask.shape != (batch, length):
            raise ValueError("attention_mask must match the candidate sequences")
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("attention_mask must contain only zero and one")
        if not bool(attention_mask.bool().any(dim=1).all()):
            raise ValueError("every candidate sequence needs an unmasked token")
        capacity = getattr(self.decoder.config, "max_position_embeddings", None)
        if capacity is None:
            capacity = getattr(self.decoder.config, "n_positions", None)
        if capacity is not None and length > capacity:
            raise ValueError(f"candidate sequence length {length} exceeds decoder capacity {capacity}")
        reference = next(self.decoder.parameters(), None)
        if reference is not None:
            inputs_embeds = inputs_embeds.to(device=reference.device, dtype=reference.dtype)
        attention_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.long)
        # Reset positions after padding so padding cannot shift candidate evidence.
        position_ids = attention_mask.cumsum(-1).sub(1).clamp_min(0)
        output = self.decoder(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, use_cache=False, return_dict=True,
        )
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None or hidden.shape != inputs_embeds.shape:
            raise ValueError("decoder must return aligned last_hidden_state")
        return hidden


def load_decoder(
    path: str, dtype: str = "float32", trust_remote_code: bool = False,
) -> nn.Module:
    """Load user-supplied local Hugging Face decoder weights, never download them."""
    dtypes = {"float32": torch.float32, "float16": torch.float16,
              "bfloat16": torch.bfloat16}
    if dtype not in dtypes:
        raise ValueError("dtype must be float32, float16, or bfloat16")
    from transformers import AutoModel
    return AutoModel.from_pretrained(
        path, torch_dtype=dtypes[dtype], local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
