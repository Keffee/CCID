import json
import tempfile
import unittest
from pathlib import Path
import torch
from transformers import GPT2Config, GPT2Model

from ccid.train import run_training
from ccid.predict import run_prediction
from ccid.checkpoint import restore_model, backbone_fingerprint
from ccid.io import load_episodes
from ccid.metrics import evaluate_records
from ccid.reflect import build_memory


class PipelineTests(unittest.TestCase):
    def test_backbone_identity_includes_behavioral_config(self):
        model = GPT2Model(GPT2Config(n_embd=8, n_head=2, n_layer=1, vocab_size=16))
        before = backbone_fingerprint(model)
        model.config.activation_function = "relu"
        self.assertNotEqual(backbone_fingerprint(model), before)
        after = backbone_fingerprint(model)
        model.config._name_or_path = "external/relocated"
        self.assertEqual(backbone_fingerprint(model), after)

    def test_training_prediction_and_adapter_only_checkpoint(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backbone = root / "backbone"
            torch.manual_seed(11)
            GPT2Model(GPT2Config(
                n_embd=16, n_head=2, n_layer=1, n_positions=128,
                vocab_size=32, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
            )).save_pretrained(backbone)
            for split in ("train", "validation", "test"):
                rows = []
                labels = []
                for i in range(3):
                    request_id = f"{split}-{i}"
                    rows.append(dict(
                        request_id=request_id, candidate_ids=["a", "b"],
                        candidate_states=torch.randn(2, 8),
                        history_states=torch.randn(i, 8),
                        base_scores=torch.tensor([0.2, 0.0]),
                    ))
                    labels.append(dict(request_id=request_id,
                                       target_id="b" if i < 2 else "outside"))
                torch.save({"split": split, "episodes": rows}, root / f"{split}.pt")
                (root / f"{split}.jsonl").write_text(
                    "".join(json.dumps(x)+"\n" for x in labels))
            cfg = dict(
                backbone=dict(path=str(backbone), dtype="float32",
                              trust_remote_code=False),
                model=dict(input_dim=8, state_dim=8, max_rounds=2, top_k=2,
                           disagreement_threshold=0.05, max_calls=6,
                           adaptive=False),
                training=dict(seed=7, epochs=2, batch_size=2,
                              learning_rate=0.001, weight_decay=0.0001,
                              grad_clip=1.0, k=2, device="cpu"),
                data=dict(train=str(root/"train.pt"),
                          train_labels=str(root/"train.jsonl"),
                          validation=str(root/"validation.pt"),
                          validation_labels=str(root/"validation.jsonl")),
                output_dir=str(root/"run"),
            )
            checkpoint = run_training(cfg)
            saved = torch.load(checkpoint, weights_only=True)
            self.assertTrue(saved["parameters"])
            self.assertFalse(any("backbone" in k for k in saved["parameters"]))
            model = restore_model(checkpoint, str(backbone), device="cpu")
            _, episodes = load_episodes(root/"test.pt")
            with torch.inference_mode():
                expected = model(episodes[0]).scores
            output = root/"predictions.jsonl"
            run_prediction(checkpoint, str(backbone), root/"test.pt", output,
                           device="cpu")
            rows = [json.loads(x) for x in output.read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertTrue(torch.allclose(
                torch.tensor(rows[0]["scores"]), expected, atol=1e-6))
            self.assertNotIn("target", output.read_text())
            self.assertTrue(all(x["realized_calls"] == 6 for x in rows))
            labels = {f"test-{i}": "b" if i<2 else "outside" for i in range(3)}
            self.assertEqual(evaluate_records(rows, labels, k=2)["contexts"], 3)
            training_predictions = root / "train_predictions.jsonl"
            run_prediction(checkpoint, str(backbone), root/"train.pt",
                           training_predictions, device="cpu")
            snapshot = build_memory(
                root/"train.pt", training_predictions, root/"train.jsonl",
                scope="demo", snapshot_id="v1", min_support=1,
                expires_on="2030-01-01", output=root/"memory.json")
            self.assertGreater(len(snapshot.lessons), 0)
            cfg["memory"] = dict(path=str(root/"memory.json"), scope="demo",
                                 as_of="2029-01-01", top_k=4)
            with self.assertRaises(FileExistsError):
                run_training(cfg)
            cfg["output_dir"] = str(root/"memory_run")
            checkpoint2 = run_training(cfg)
            memory_hash = snapshot.digest
            records2 = run_prediction(
                checkpoint2, str(backbone), root/"test.pt", root/"memory_predictions.jsonl",
                device="cpu", memory_path=root/"memory.json", scope="demo",
                as_of="2029-01-01")
            self.assertTrue(all(x["memory_snapshot_sha256"] == memory_hash for x in records2))
            with self.assertRaises(ValueError):
                run_prediction(checkpoint2, str(backbone), root/"test.pt",
                               root/"wrong_memory_predictions.jsonl", device="cpu")


if __name__ == "__main__":
    unittest.main()
