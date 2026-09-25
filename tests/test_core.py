"""Core contracts and real frozen-decoder regression tests."""
import dataclasses
import importlib.util
import os
import tempfile
import unittest

os.environ.setdefault("USE_TF", "0")
import torch
from transformers import GPT2Config, GPT2Model

try:
    from ccid import CCID, Episode, ModelConfig
    from ccid.backbone import FrozenDecoder, load_decoder
    from ccid.contracts import RoleResponse
    from ccid.controller import DebateController, challenge_veto_conflict
except ImportError:
    CCID = None


def tiny_decoder():
    return GPT2Model(GPT2Config(
        vocab_size=32, n_embd=16, n_layer=1, n_head=2,
        n_positions=128, n_ctx=128, attn_pdrop=0.3,
        resid_pdrop=0.3, embd_pdrop=0.3,
    ))


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(CCID, "CCID core API must exist")
        torch.manual_seed(19)
        torch.set_num_threads(1)

    def episode(self, order=("a", "b", "c")):
        return Episode("request", order, torch.randn(3, 8),
                       torch.randn(2, 8), torch.tensor([3.0, 2.0, 1.0]))

    def model(self, **kwargs):
        values = dict(input_dim=8, state_dim=6, max_rounds=3,
                      top_k=3, max_calls=9, adaptive=False)
        values.update(kwargs)
        return CCID(tiny_decoder(), ModelConfig(**values))

    def test_episode_contract_and_empty_history(self):
        episode = self.episode()
        episode.validate()
        self.assertEqual({field.name for field in dataclasses.fields(episode)},
                         {"request_id", "candidate_ids", "candidate_states",
                          "history_states", "base_scores"})
        episode.history_states = torch.empty(0, 8)
        episode.validate()
        moved = episode.to("cpu")
        self.assertEqual(moved.candidate_ids, episode.candidate_ids)
        self.assertEqual(tuple(moved.history_states.shape), (0, 8))

    def test_episode_rejects_duplicate_empty_or_misaligned_candidates(self):
        episode = self.episode()
        for change in (
            {"candidate_ids": ("a", "a", "c")},
            {"candidate_ids": ("a", "", "c")},
            {"candidate_states": torch.empty(0, 8), "candidate_ids": (),
             "base_scores": torch.empty(0)},
            {"history_states": torch.zeros(2, 7)},
            {"candidate_states": torch.zeros(2, 8)},
            {"base_scores": torch.tensor([1.0, float("nan"), 2.0])},
        ):
            with self.subTest(change=tuple(change)):
                with self.assertRaises((ValueError, TypeError)):
                    dataclasses.replace(episode, **change).validate()

    def test_model_config_rejects_invalid_limits(self):
        for values in ({"max_rounds": 0}, {"max_rounds": -1},
                       {"max_calls": -1}, {"max_calls": 1.5},
                       {"top_k": 0}, {"state_dim": 0},
                       {"disagreement_threshold": float("nan")},
                       {"interface": "unknown"}):
            with self.subTest(values=values):
                with self.assertRaises((ValueError, TypeError)):
                    ModelConfig(**values)

    def test_frozen_decoder_uses_real_backbone_keeps_gradients_and_eval(self):
        adapter = FrozenDecoder(tiny_decoder())
        adapter.train()
        self.assertFalse(adapter.training)
        self.assertFalse(adapter.decoder.training)
        inputs = torch.randn(2, 4, 16, requires_grad=True)
        hidden = adapter(inputs, torch.ones(2, 4, dtype=torch.long))
        hidden[:, -1, 0].sum().backward()
        self.assertGreater(inputs.grad[:, 0].abs().sum().item(), 0)
        self.assertTrue(all(not p.requires_grad and p.grad is None
                            for p in adapter.decoder.parameters()))
        torch.testing.assert_close(adapter(inputs), adapter(inputs))

    def test_decoder_padding_mask_and_candidate_batch_isolation(self):
        adapter = FrozenDecoder(tiny_decoder())
        inputs = torch.randn(2, 5, 16)
        mask = torch.tensor([[1, 1, 0, 0, 1], [1, 1, 1, 1, 1]])
        baseline = adapter(inputs, mask)
        changed = inputs.clone()
        changed[0, 2:4] += 100
        changed[1] *= -20
        altered = adapter(changed, mask)
        torch.testing.assert_close(baseline[0, -1], altered[0, -1],
                                   rtol=1e-5, atol=1e-6)
        self.assertFalse(torch.allclose(baseline[1, -1], altered[1, -1]))
        with self.assertRaises(ValueError):
            adapter(inputs, torch.zeros(2, 5, dtype=torch.long))

    def test_decoder_can_load_external_local_huggingface_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            tiny_decoder().save_pretrained(directory)
            decoder = load_decoder(directory)
            self.assertIsInstance(decoder, GPT2Model)
            self.assertEqual(decoder.config.hidden_size, 16)
        with self.assertRaises(ValueError):
            load_decoder("unused", dtype="invalid")

    def test_output_shapes_metadata_and_exact_call_cost(self):
        model = self.model(max_calls=7)
        episode = self.episode()
        out = model(episode, torch.randn(2, 8))
        self.assertEqual(out.candidate_ids, episode.candidate_ids)
        self.assertEqual(len(out.rounds), 2)
        self.assertEqual(out.realized_calls, 6)
        self.assertEqual(out.permitted_calls, 7)
        self.assertEqual(out.stop_reason, "call_budget")
        torch.testing.assert_close(out.initial_scores, episode.base_scores)
        self.assertEqual(tuple(out.scores.shape), (3,))
        self.assertEqual(sorted(out.ranking.tolist()), [0, 1, 2])
        for index, result in enumerate(out.rounds):
            self.assertEqual(result.calls, 3)
            self.assertEqual(result.round_index, index)
            self.assertEqual(tuple(result.states.shape), (3, 6))
            for name in ("utility", "risk"):
                response = getattr(result, name)
                self.assertEqual(response.candidate_ids, episode.candidate_ids)
                self.assertEqual(response.role, name)
                self.assertEqual(response.round_index, index)
                self.assertEqual(response.state_version, index)
                self.assertEqual(tuple(response.latent.shape), (3, 6))
                self.assertEqual(tuple(response.logits.shape), (3,))
        self.assertFalse(model.backbone.decoder.training)

    def test_no_round_for_insufficient_budget_preserves_stable_ties(self):
        for budget in (0, 1, 2):
            episode = self.episode()
            episode.base_scores = torch.tensor([2.0, 2.0, 1.0])
            out = self.model(max_calls=budget)(episode)
            self.assertEqual(out.rounds, [])
            self.assertEqual(out.realized_calls, 0)
            self.assertEqual(out.stop_reason, "call_budget")
            torch.testing.assert_close(out.scores, episode.base_scores)
            self.assertEqual(out.ranking.tolist(), [0, 1, 2])

    def test_role_metadata_validation_rejects_wrong_candidate_binding(self):
        response = RoleResponse(("a", "b"), "utility", 0, 0,
                                torch.ones(2, 6), torch.zeros(2))
        response.validate(candidate_ids=("a", "b"), round_index=0,
                          state_version=0, state_dim=6)
        for changes in ({"candidate_ids": ("b", "a")},
                        {"round_index": 1}, {"state_version": 1},
                        {"state_dim": 5}):
            with self.subTest(changes=changes):
                expected = dict(candidate_ids=("a", "b"), round_index=0,
                                state_version=0, state_dim=6)
                expected.update(changes)
                with self.assertRaises(ValueError):
                    response.validate(**expected)

    def test_conflict_is_challenge_times_veto_and_excludes_incumbent(self):
        utility = torch.tensor([100.0, -100.0, 0.0])
        risk = torch.tensor([100.0, 100.0, 0.0])
        self.assertAlmostEqual(challenge_veto_conflict(utility, risk, 0), 0.25)
        self.assertEqual(challenge_veto_conflict(torch.ones(1), torch.ones(1), 0), 0)

    def test_controller_requires_two_consecutive_stable_low_conflict_rounds(self):
        config = ModelConfig(top_k=2, disagreement_threshold=0.1)
        controller = DebateController(config, torch.tensor([3.0, 2.0, 1.0]))
        self.assertFalse(controller.observe(torch.tensor([3.0, 2.0, 1.0]), 0.05))
        self.assertFalse(controller.observe(torch.tensor([3.0, 2.0, 1.0]), 0.2))
        self.assertFalse(controller.observe(torch.tensor([1.0, 3.0, 2.0]), 0.05))
        self.assertFalse(controller.observe(torch.tensor([1.0, 3.0, 2.0]), 0.05))
        self.assertTrue(controller.observe(torch.tensor([1.0, 3.0, 2.0]), 0.05))

    def test_adaptive_stops_after_two_rounds_and_fixed_uses_full_round_limit(self):
        episode = self.episode()
        episode.base_scores = torch.zeros(3)
        for adaptive, expected, reason in ((True, 2, "stable"),
                                           (False, 4, "max_rounds")):
            model = self.model(max_rounds=4, max_calls=12, adaptive=adaptive,
                               disagreement_threshold=1.0)
            with torch.no_grad():
                model.score_head.weight.zero_()
                model.score_head.bias.zero_()
            out = model(episode)
            self.assertEqual(len(out.rounds), expected)
            self.assertEqual(out.stop_reason, reason)
            self.assertEqual(out.realized_calls, 3 * expected)

    def test_final_loss_reaches_prior_role_and_adjudicated_state(self):
        model = self.model()
        model.train()
        out = model(self.episode())
        first = out.rounds[0]
        first.states.retain_grad()
        first.utility.latent.retain_grad()
        first.risk.latent.retain_grad()
        out.scores.square().sum().backward()
        for tensor in (first.states, first.utility.latent, first.risk.latent):
            self.assertIsNotNone(tensor.grad)
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertGreater(model.initial_state[1].weight.grad.abs().sum().item(), 0)

    def test_later_role_inputs_change_when_adjudicated_state_changes(self):
        model = self.model(max_rounds=2, max_calls=6)
        episode = self.episode()
        baseline = model(episode)
        handle = model.state_projection.register_forward_hook(
            lambda module, inputs, output: output + 7.0)
        altered = model(episode)
        handle.remove()
        torch.testing.assert_close(baseline.rounds[0].utility.logits,
                                   altered.rounds[0].utility.logits)
        self.assertFalse(torch.allclose(baseline.rounds[1].utility.logits,
                                       altered.rounds[1].utility.logits))
        self.assertFalse(torch.allclose(baseline.rounds[1].risk.logits,
                                       altered.rounds[1].risk.logits))

    def test_feedback_off_never_consumes_adjudicated_states(self):
        model = self.model(max_rounds=2, max_calls=6, feedback=False)
        episode = self.episode()
        baseline = model(episode)
        handle = model.state_projection.register_forward_hook(
            lambda module, inputs, output: output + 7.0)
        altered = model(episode)
        handle.remove()
        for expected, actual in zip(baseline.rounds, altered.rounds):
            torch.testing.assert_close(expected.utility.logits, actual.utility.logits)
            torch.testing.assert_close(expected.risk.logits, actual.risk.logits)
            torch.testing.assert_close(expected.scores, actual.scores)
            self.assertEqual(expected.utility.state_version, 0)

    def test_feedback_off_repeats_identical_initial_board_computation(self):
        model = self.model(feedback=False)
        output = model(self.episode(), torch.randn(2, 8))
        first = output.rounds[0]
        for item in output.rounds[1:]:
            torch.testing.assert_close(item.utility.logits, first.utility.logits)
            torch.testing.assert_close(item.risk.logits, first.risk.logits)
            torch.testing.assert_close(item.scores, first.scores)
            torch.testing.assert_close(item.states, first.states)
            self.assertGreater(item.round_index, first.round_index)

    def test_score_only_blocks_continuous_messages_but_keeps_scalar_feedback(self):
        model = self.model(max_rounds=2, max_calls=6, interface="score_only")
        episode = self.episode()
        baseline = model(episode)
        latent_hook = model.role_latent_heads["utility"].register_forward_hook(
            lambda module, inputs, output: output + 7.0)
        state_hook = model.state_projection.register_forward_hook(
            lambda module, inputs, output: output + 7.0)
        altered = model(episode)
        latent_hook.remove()
        state_hook.remove()
        torch.testing.assert_close(baseline.scores, altered.scores)
        self.assertEqual(baseline.rounds[0].utility.latent.count_nonzero().item(), 0)
        calls = [0]
        def change_first_scalar(module, inputs, output):
            calls[0] += 1
            return output + 7.0 if calls[0] == 1 else output
        scalar_hook = model.role_logit_heads["utility"].register_forward_hook(change_first_scalar)
        scalar_altered = model(episode)
        scalar_hook.remove()
        self.assertFalse(torch.allclose(baseline.rounds[1].utility.logits,
                                       scalar_altered.rounds[1].utility.logits))
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         sum(p.numel() for p in self.model(interface="latent",
                             max_rounds=2, max_calls=6).parameters()))

    def test_risk_reads_current_leader_and_nonleader_sequences_are_isolated(self):
        model = self.model(max_rounds=1, max_calls=3)
        episode = self.episode()
        baseline = model(episode).rounds[0]
        changed_leader = dataclasses.replace(episode,
                                             candidate_states=episode.candidate_states.clone())
        changed_leader.candidate_states[0] = torch.randn(8) * 3
        leader_result = model(changed_leader).rounds[0]
        self.assertFalse(torch.allclose(baseline.risk.logits[1], leader_result.risk.logits[1]))
        changed_other = dataclasses.replace(episode,
                                            candidate_states=episode.candidate_states.clone())
        changed_other.candidate_states[2] = torch.randn(8) * 3
        other_result = model(changed_other).rounds[0]
        torch.testing.assert_close(baseline.risk.logits[1], other_result.risk.logits[1])
        torch.testing.assert_close(baseline.scores[1], other_result.scores[1])

    def test_candidate_identity_survives_rank_changes_and_storage_permutation(self):
        model = self.model()
        episode = self.episode()
        def force_ranking(module, inputs, output):
            return output + torch.tensor([[0.0], [10.0], [20.0]])
        handle = model.score_head.register_forward_hook(force_ranking)
        changed = model(episode)
        handle.remove()
        self.assertEqual(changed.rounds[1].incumbent_index, 2)
        for item in changed.rounds:
            self.assertEqual(item.utility.candidate_ids, episode.candidate_ids)
        baseline = model(episode)
        permutation = torch.tensor([2, 0, 1])
        permuted = dataclasses.replace(
            episode, candidate_ids=tuple(episode.candidate_ids[i] for i in permutation),
            candidate_states=episode.candidate_states[permutation],
            base_scores=episode.base_scores[permutation])
        reordered = model(permuted)
        inverse = torch.argsort(permutation)
        for expected, actual in zip(baseline.rounds, reordered.rounds):
            torch.testing.assert_close(expected.scores, actual.scores[inverse], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(expected.utility.latent,
                                       actual.utility.latent[inverse], rtol=1e-5, atol=1e-6)

    def test_frozen_memory_is_input_only_and_empty_history_works(self):
        model = self.model()
        episode = self.episode()
        episode.history_states = torch.empty(0, 8)
        memory = torch.randn(2, 8, requires_grad=True)
        before = memory.detach().clone()
        model(episode, memory).scores.sum().backward()
        self.assertIsNone(memory.grad)
        torch.testing.assert_close(memory, before)
        with self.assertRaises(ValueError):
            model(episode, torch.zeros(2, 7))

    def test_memory_type_semantics_influence_output_and_receive_gradients(self):
        model = self.model()
        episode = self.episode()
        memory = torch.randn(2, 8)
        first = model(episode, memory, memory_types=torch.tensor([0, 0]))
        second = model(episode, memory, memory_types=torch.tensor([2, 2]))
        self.assertFalse(torch.allclose(first.scores, second.scores))
        second.scores.sum().backward()
        self.assertGreater(model.memory_type_embedding.weight.grad[2].abs().sum().item(), 0)
        neutral = model(episode, memory)
        explicit_neutral = model(episode, memory, memory_types=torch.tensor([4, 4]))
        torch.testing.assert_close(neutral.scores, explicit_neutral.scores)
        for types in (torch.tensor([0]), torch.tensor([0.0, 1.0]), torch.tensor([0, 5])):
            with self.assertRaises(ValueError):
                model(episode, memory, memory_types=types)

    def test_memory_features_influence_outputs_and_receive_adapter_gradients(self):
        import inspect
        self.assertIn("memory_features", inspect.signature(CCID.forward).parameters)
        model = self.model()
        episode = self.episode()
        memory = torch.randn(2, 8)
        features = torch.tensor([[0.2, 0.4, 0.1], [0.6, 0.7, 0.3]], requires_grad=True)
        before = features.detach().clone()
        baseline = model(episode, memory)
        explicit_default = model(episode, memory, memory_features=torch.zeros(2, 3))
        torch.testing.assert_close(baseline.scores, explicit_default.scores)
        altered = model(episode, memory, memory_features=features)
        self.assertFalse(torch.allclose(baseline.scores, altered.scores))
        altered.scores.sum().backward()
        self.assertGreater(model.memory_feature_adapter.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(features.grad)
        torch.testing.assert_close(features, before)

    def test_memory_features_reject_invalid_shape_type_and_nonfinite_values(self):
        import inspect
        self.assertIn("memory_features", inspect.signature(CCID.forward).parameters)
        model = self.model()
        episode = self.episode()
        memory = torch.randn(2, 8)
        for features in (torch.zeros(1, 3), torch.zeros(2, 2), torch.zeros(6),
                         torch.zeros(2, 3, dtype=torch.long),
                         torch.full((2, 3), float("nan")), [[0.0, 0.0, 0.0]]):
            with self.subTest(features=repr(features)):
                with self.assertRaises(ValueError):
                    model(episode, memory, memory_features=features)
        with self.assertRaises(ValueError):
            model(episode, memory_features=torch.zeros(1, 3))
        model(episode, memory_features=torch.empty(0, 3))

    def test_adaptive_stop_rejects_role_controls_without_risk_semantics(self):
        for mode in ("duplicate_utility", "unified"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "fixed rounds"):
                    ModelConfig(role_mode=mode, adaptive=True)
                ModelConfig(role_mode=mode, adaptive=False)

    def test_score_only_utility_risk_supports_adaptive_stopping(self):
        episode = self.episode()
        episode.base_scores = torch.zeros(3)
        model = self.model(interface="score_only", role_mode="utility_risk",
                           adaptive=True, disagreement_threshold=1.0)
        with torch.no_grad():
            model.score_head.weight.zero_()
            model.score_head.bias.zero_()
        output = model(episode)
        self.assertEqual(output.stop_reason, "stable")
        self.assertEqual(len(output.rounds), 2)
        self.assertEqual(output.realized_calls, 6)

    def test_optional_role_modes_keep_capacity_and_unified_reuses_role(self):
        capacities = []
        for mode in ("utility_risk", "duplicate_utility", "unified"):
            model = self.model(role_mode=mode, adaptive=False)
            capacities.append(sum(p.numel() for p in model.parameters()))
            out = model(self.episode())
            if mode == "unified":
                for item in out.rounds:
                    torch.testing.assert_close(item.utility.logits, item.risk.logits)
                    torch.testing.assert_close(item.utility.latent, item.risk.latent)
        self.assertEqual(len(set(capacities)), 1)


if __name__ == "__main__":
    unittest.main()
