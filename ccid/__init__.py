"""CCID: candidate-keyed continuous inter-agent debate."""
from .backbone import FrozenDecoder, load_decoder
from .contracts import DebateOutput, Episode, ModelConfig, RoleResponse, RoundOutput
from .model import CCID

__all__ = [
    "CCID", "DebateOutput", "Episode", "FrozenDecoder", "ModelConfig",
    "RoleResponse", "RoundOutput", "load_decoder",
]
