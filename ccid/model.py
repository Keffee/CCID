"""Candidate-keyed multi-round utility/risk communication over a frozen decoder."""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .backbone import FrozenDecoder
from .contracts import DebateOutput, Episode, ModelConfig, RoleResponse, RoundOutput
from .controller import CALLS_PER_ROUND, DebateController, challenge_veto_conflict


@dataclass
class _Message:
    candidate_id: str
    role: str
    round_index: int
    state_version: int
    latent: Tensor
    logit: Tensor


class _Board:
    """Identity maps are the only route from a candidate to evolving evidence."""

    def __init__(self, candidate_ids: tuple[str, ...], states: Tensor, scores: Tensor):
        self.candidate_ids = candidate_ids
        self.states = dict(zip(candidate_ids, states.unbind(0)))
        self.scores = dict(zip(candidate_ids, scores.unbind(0)))
        self.responses: dict[str, list[_Message]] = {item: [] for item in candidate_ids}
        self.state_version = 0

    def state_tensor(self) -> Tensor:
        return torch.stack([self.states[item] for item in self.candidate_ids])

    def score_tensor(self) -> Tensor:
        return torch.stack([self.scores[item] for item in self.candidate_ids])

    def record(self, response: RoleResponse, round_index: int, state_dim: int) -> None:
        response.validate(candidate_ids=self.candidate_ids, round_index=round_index,
                          state_version=self.state_version, state_dim=state_dim)
        for row, item in enumerate(response.candidate_ids):
            self.responses[item].append(_Message(
                item, response.role, response.round_index, response.state_version,
                response.latent[row], response.logits[row],
            ))

    def update(self, states: Tensor, scores: Tensor, state_version: int) -> None:
        if states.shape[0] != len(self.candidate_ids) or scores.shape != (len(self.candidate_ids),):
            raise ValueError("adjudication cannot change candidate identity alignment")
        self.states = dict(zip(self.candidate_ids, states.unbind(0)))
        self.scores = dict(zip(self.candidate_ids, scores.unbind(0)))
        self.state_version = state_version


class CCID(nn.Module):
    """Frozen-decoder debate with trainable continuous adapters and readouts.

    Three logical calls comprise a complete round: utility, risk, adjudication.
    Candidate batching is an implementation detail, not a compute-equivalence claim.

    utility_risk uses independent semantic roles and heads. duplicate_utility
    retains independent heads but gives both the utility role identity; its loss
    must supervise both for utility. unified evaluates the same utility head twice
    on the same board, with both utility and risk objectives assigned to that head.
    All configurations instantiate the same parameter structure.
    """

    def __init__(self, decoder: nn.Module, config: ModelConfig):
        super().__init__()
        config.__post_init__()
        self.config = config
        self.backbone = FrozenDecoder(decoder)
        width = self.backbone.hidden_size
        state = config.state_dim
        self.initial_state = nn.Sequential(nn.LayerNorm(config.input_dim),
                                           nn.Linear(config.input_dim, state), nn.GELU())
        self.context_projection = nn.Linear(config.input_dim, width)
        self.memory_projection = nn.Linear(config.input_dim, width)
        # Four diagnosed case types plus neutral, untyped memory.
        self.memory_type_embedding = nn.Embedding(5, width)
        # Visible trajectory summary: conflict, promotion frequency, reversal rate.
        self.memory_feature_adapter = nn.Linear(3, width)
        self.state_adapter = nn.Linear(state, width)
        self.leader_adapter = nn.Linear(state, width)
        self.response_adapter = nn.Linear(state, width)
        self.scalar_adapter = nn.Linear(1, width)
        self.rank_adapter = nn.Linear(2, width)
        self.role_embedding = nn.Embedding(3, width)
        self.round_embedding = nn.Embedding(config.max_rounds + 1, width)
        self.version_embedding = nn.Embedding(config.max_rounds + 1, width)
        self.token_type_embedding = nn.Embedding(6, width)
        self.query_embedding = nn.Embedding(3, width)
        self.role_latent_heads = nn.ModuleDict(
            {role: nn.Linear(width, state) for role in ("utility", "risk")})
        self.role_logit_heads = nn.ModuleDict(
            {role: nn.Linear(width, 1) for role in ("utility", "risk")})
        self.score_head = nn.Linear(width, 1)
        self.state_projection = nn.Linear(width, state)

    def _role_id(self, role: str) -> int:
        if role == "adjudicator":
            return 2
        if self.config.role_mode in {"duplicate_utility", "unified"}:
            return 0
        return 0 if role == "utility" else 1

    def _message_token(self, message: _Message, candidate_id: str,
                       current_round: int, allow_current: bool) -> Tensor:
        latest = current_round if allow_current else current_round - 1
        expected_version = (message.round_index
                            if self.config.feedback and self.config.interface == "latent" else 0)
        if (message.candidate_id != candidate_id or message.role not in {"utility", "risk"}
                or not 0 <= message.round_index <= latest
                or message.state_version != expected_version
                or message.latent.shape != (self.config.state_dim,)
                or message.logit.ndim != 0):
            raise ValueError("candidate response identity, role, round, or state version is inconsistent")
        token = self.scalar_adapter(message.logit.reshape(1))
        if self.config.interface == "latent":
            token = token + self.response_adapter(message.latent)
        return (token + self.role_embedding.weight[self._role_id(message.role)]
                + self.round_embedding.weight[message.round_index if self.config.feedback else 0]
                + self.version_embedding.weight[message.state_version]
                + self.token_type_embedding.weight[4])

    def _read(self, role: str, board: _Board, history: Tensor, memory: Tensor,
              memory_types: Tensor, memory_features: Tensor, round_index: int,
              current: tuple[RoleResponse, ...] = ()) -> Tensor:
        states = board.state_tensor()
        scores = board.score_tensor()
        ranking = torch.argsort(scores.detach(), descending=True, stable=True)
        ranks = torch.empty_like(ranking)
        ranks[ranking] = torch.arange(ranking.numel(), device=ranking.device)
        leader = states[ranking[0]]
        role_id = self._role_id(role)
        # Repeating the initial board must also repeat its input round metadata.
        input_round = round_index if self.config.feedback else 0
        role_token = self.role_embedding.weight[role_id] + self.round_embedding.weight[input_round]
        history_tokens = self.context_projection(history) + self.token_type_embedding.weight[0]
        memory_tokens = (self.memory_projection(memory)
                         + self.memory_type_embedding(memory_types)
                         + self.memory_feature_adapter(memory_features)
                         + self.token_type_embedding.weight[1])
        leader_token = self.leader_adapter(leader) + self.token_type_embedding.weight[2]
        current_maps = []
        for response in current:
            response.validate(candidate_ids=board.candidate_ids, round_index=round_index,
                              state_version=board.state_version, state_dim=self.config.state_dim)
            current_maps.append({item: _Message(
                item, response.role, response.round_index, response.state_version,
                response.latent[row], response.logits[row],
            ) for row, item in enumerate(response.candidate_ids)})
        sequences = []
        for row, candidate_id in enumerate(board.candidate_ids):
            own_token = self.state_adapter(board.states[candidate_id]) + self.token_type_embedding.weight[3]
            features = torch.stack((
                ranks[row].to(dtype=scores.dtype) / max(1, len(board.candidate_ids) - 1),
                scores[row],
            ))
            rank_token = self.rank_adapter(features) + self.token_type_embedding.weight[5]
            pieces = [role_token[None], history_tokens, memory_tokens, leader_token[None],
                      own_token[None], rank_token[None]]
            for message in board.responses[candidate_id]:
                pieces.append(self._message_token(message, candidate_id, round_index, False)[None])
            for messages in current_maps:
                pieces.append(self._message_token(messages[candidate_id], candidate_id,
                                                  round_index, True)[None])
            # A causal readout must trail all evidence it is meant to consume.
            query = self.query_embedding.weight[role_id] + self.round_embedding.weight[input_round]
            pieces.append(query[None])
            sequences.append(torch.cat(pieces, dim=0))
        lengths = {sequence.shape[0] for sequence in sequences}
        if len(lengths) != 1:
            raise ValueError("candidate response histories must remain round-aligned")
        embeddings = torch.stack(sequences)
        attention_mask = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.long)
        hidden = self.backbone(embeddings, attention_mask)
        return hidden[:, -1].to(dtype=self.score_head.weight.dtype)

    def _response(self, role: str, board: _Board, history: Tensor, memory: Tensor,
                  memory_types: Tensor, memory_features: Tensor, round_index: int) -> RoleResponse:
        hidden = self._read(role, board, history, memory, memory_types, memory_features, round_index)
        head = "utility" if self.config.role_mode == "unified" else role
        logits = self.role_logit_heads[head](hidden).squeeze(-1)
        latent = (self.role_latent_heads[head](hidden)
                  if self.config.interface == "latent"
                  else hidden.new_zeros((hidden.shape[0], self.config.state_dim)))
        response = RoleResponse(board.candidate_ids, role, round_index,
                                board.state_version, latent, logits)
        response.validate(candidate_ids=board.candidate_ids, round_index=round_index,
                          state_version=board.state_version, state_dim=self.config.state_dim)
        return response

    def forward(
        self, episode: Episode, memory: Optional[Tensor] = None,
        memory_types: Optional[Tensor] = None, memory_features: Optional[Tensor] = None,
    ) -> DebateOutput:
        episode.validate()
        self.config.__post_init__()
        if episode.candidate_states.shape[1] != self.config.input_dim:
            raise ValueError("episode input width differs from ModelConfig.input_dim")
        reference = self.score_head.weight
        candidates = episode.candidate_states.to(device=reference.device, dtype=reference.dtype)
        history = episode.history_states.to(device=reference.device, dtype=reference.dtype)
        initial_scores = episode.base_scores.to(device=reference.device, dtype=reference.dtype)
        if memory is None:
            if memory_types is not None and memory_types.numel() != 0:
                raise ValueError("memory_types require memory vectors")
            memory = candidates.new_empty((0, self.config.input_dim))
        if (not isinstance(memory, Tensor) or memory.ndim != 2
                or memory.shape[1] != self.config.input_dim
                or not memory.is_floating_point()
                or not bool(torch.isfinite(memory).all())):
            raise ValueError("memory must be a finite [items, input_dim] floating tensor")
        # Freeze the request snapshot, while leaving its receiving adapters trainable.
        memory = memory.detach().to(device=reference.device, dtype=reference.dtype).clone()
        if memory_types is None:
            memory_types = torch.full((len(memory),), 4, device=reference.device, dtype=torch.long)
        elif (not isinstance(memory_types, Tensor) or memory_types.shape != (len(memory),)
              or memory_types.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
              or not bool(((memory_types >= 0) & (memory_types <= 4)).all())):
            raise ValueError("memory_types must be integer IDs in [0, 4], one per memory item")
        else:
            memory_types = memory_types.detach().to(device=reference.device, dtype=torch.long).clone()
        if memory_features is None:
            memory_features = memory.new_zeros((len(memory), 3))
        elif (not isinstance(memory_features, Tensor)
              or memory_features.shape != (len(memory), 3)
              or not memory_features.is_floating_point()
              or not bool(torch.isfinite(memory_features).all())):
            raise ValueError("memory_features must be a finite [items, 3] floating tensor")
        else:
            memory_features = memory_features.detach().to(
                device=reference.device, dtype=reference.dtype).clone()
        output = DebateOutput(episode.candidate_ids, initial_scores, initial_scores,
                              permitted_calls=self.config.max_calls)
        if self.config.max_calls < CALLS_PER_ROUND:
            return output
        initial_states = self.initial_state(candidates)
        board = _Board(episode.candidate_ids, initial_states, initial_scores)
        controller = DebateController(self.config, initial_scores)
        for round_index in range(self.config.max_rounds):
            if output.realized_calls + CALLS_PER_ROUND > self.config.max_calls:
                output.stop_reason = "call_budget"
                break
            incumbent = int(torch.argsort(board.score_tensor().detach(),
                                          descending=True, stable=True)[0].item())
            utility = self._response("utility", board, history, memory, memory_types, memory_features, round_index)
            risk = self._response("risk", board, history, memory, memory_types, memory_features, round_index)
            hidden = self._read("adjudicator", board, history, memory, memory_types,
                                memory_features, round_index, (utility, risk))
            scores = self.score_head(hidden).squeeze(-1)
            states = (self.state_projection(hidden)
                      if self.config.interface == "latent" else initial_states)
            if not bool(torch.isfinite(scores).all() and torch.isfinite(states).all()):
                raise ValueError("adjudicator returned nonfinite values")
            disagreement = challenge_veto_conflict(utility.logits, risk.logits, incumbent)
            result = RoundOutput(round_index, incumbent, scores, states, utility, risk,
                                 disagreement, CALLS_PER_ROUND)
            output.rounds.append(result)
            output.scores = scores
            output.realized_calls += CALLS_PER_ROUND
            if controller.observe(scores, disagreement):
                output.stop_reason = "stable"
                break
            if round_index + 1 >= self.config.max_rounds:
                output.stop_reason = "max_rounds"
                break
            if self.config.feedback:
                board.record(utility, round_index, self.config.state_dim)
                board.record(risk, round_index, self.config.state_dim)
                next_version = round_index + 1 if self.config.interface == "latent" else 0
                board.update(states, scores, next_version)
        return output
