import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch

from ccid.io import load_episodes, load_labels
from ccid.losses import debate_loss
from ccid.metrics import evaluate_records


class TrainingContractTests(unittest.TestCase):
    def test_labels_never_enter_runtime_and_duplicate_ids_fail(self):
        record = dict(request_id="a", candidate_ids=["x", "y"],
                      candidate_states=torch.randn(2, 8),
                      history_states=torch.randn(1, 8),
                      base_scores=torch.tensor([1., 0.]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.pt"
            torch.save({"split": "train", "episodes": [record]}, path)
            split, rows = load_episodes(path, expected_split="train")
            self.assertEqual(split, "train")
            self.assertEqual(rows[0].candidate_ids, ("x", "y"))
            record["target_id"] = "x"
            torch.save({"split": "test", "episodes": [record]}, path)
            with self.assertRaises(ValueError):
                load_episodes(path)
            record.pop("target_id")
            torch.save({"split": "train", "episodes": [record, record]}, path)
            with self.assertRaises(ValueError):
                load_episodes(path)

    def test_dynamic_incumbent_risk_and_unreachable_loss(self):
        scores = torch.tensor([0.2, 0.1], requires_grad=True)
        utility = torch.tensor([0.1, 0.2], requires_grad=True)
        risk = torch.tensor([0.3, 0.4], requires_grad=True)
        round_state = SimpleNamespace(
            incumbent_index=0, scores=scores,
            utility=SimpleNamespace(logits=utility),
            risk=SimpleNamespace(logits=risk))
        output = SimpleNamespace(candidate_ids=("a", "b"), rounds=[round_state])
        loss = debate_loss(output, "a")
        loss.backward()
        self.assertGreater(float(risk.grad[0]), 0)
        self.assertLess(float(risk.grad[1]), 0)
        risk.grad = None
        output.rounds[0].incumbent_index = 1
        debate_loss(output, "a").backward()
        self.assertTrue(bool((risk.grad > 0).all()))
        scores.grad = None
        loss = debate_loss(output, "outside")
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(scores.grad is None or bool((scores.grad == 0).all()))

    def test_complete_denominator_and_exact_label_join(self):
        rows = [
            dict(request_id="a", initial_ranking=["x","y"], ranking=["y","x"],
                 realized_calls=6, rounds=[]),
            dict(request_id="b", initial_ranking=["x","y"], ranking=["x","y"],
                 realized_calls=3, rounds=[]),
        ]
        result = evaluate_records(rows, {"a":"y","b":"outside"}, k=2)
        self.assertEqual(result["contexts"], 2)
        self.assertEqual(result["pass_at_1"], 0.5)
        self.assertEqual(result["mrr_at_k"], 0.5)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["transitions"]["corrected"], 1)
        self.assertEqual(result["transitions"]["unreachable"], 1)
        with self.assertRaises(ValueError):
            evaluate_records(rows, {"a":"y"}, k=2)
        with self.assertRaises(ValueError):
            evaluate_records(rows+[rows[0]], {"a":"y","b":"outside"}, k=2)


if __name__ == "__main__":
    unittest.main()
