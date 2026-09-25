import dataclasses
import tempfile
import unittest
from pathlib import Path
import torch

from ccid.memory import (
    Lesson, MemorySnapshot, ReflectionCase, retrieve_memory,
    reflect, save_snapshot, load_snapshot,
)


class MemoryTests(unittest.TestCase):
    def test_condition_scope_expiry_and_frozen_snapshot(self):
        lesson = Lesson(
            lesson_id="l1", scope="books", trigger="short_history",
            case_type="damaged", action="recheck_promotion",
            confidence=0.8, support=4, vector=(1.0, 2.0),
            expires_on="2030-01-01",
        )
        snapshot = MemorySnapshot("v1", "train", (lesson,))
        before = snapshot.digest
        tokens, selected = retrieve_memory(
            snapshot, scope="books", history_count=2,
            as_of="2029-01-01", top_k=2, input_dim=2,
        )
        self.assertEqual(selected, ("l1",))
        tokens[0, 0] = 99
        again, _ = retrieve_memory(snapshot, scope="books", history_count=2,
            as_of="2029-01-01", top_k=2, input_dim=2)
        self.assertEqual(float(again[0, 0]), 1.0)
        self.assertEqual(snapshot.digest, before)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            lesson.confidence = 0.1
        for scope, count, date in [("music", 2, "2029-01-01"),
                                    ("books", 10, "2029-01-01"),
                                    ("books", 2, "2031-01-01")]:
            tokens, ids = retrieve_memory(snapshot, scope=scope,
                history_count=count, as_of=date, top_k=2, input_dim=2)
            self.assertEqual(tokens.shape, (0, 2))
            self.assertEqual(ids, ())

    def test_reflection_uses_train_cases_and_no_answers(self):
        cases = [
            ReflectionCase("books", 2, "damaged", (1.0, 0.0)),
            ReflectionCase("books", 2, "damaged", (3.0, 2.0)),
            ReflectionCase("books", 8, "corrected", (0.0, 1.0)),
        ]
        snapshot = reflect(cases, snapshot_id="v1", source_split="train",
                           min_support=2, expires_on="2030-01-01")
        self.assertEqual(len(snapshot.lessons), 1)
        self.assertEqual(snapshot.lessons[0].vector, (2.0, 1.0))
        self.assertEqual(snapshot.lessons[0].support, 2)
        self.assertEqual(snapshot.lessons[0].action, "recheck_promotion")
        with self.assertRaises(ValueError):
            reflect(cases, snapshot_id="v2", source_split="test",
                    min_support=1, expires_on="2030-01-01")
        with self.assertRaises(ValueError):
            Lesson("bad", "books", "target_in_pool", "damaged",
                   "recheck_promotion", 0.8, 2, (1.0,), "2030-01-01")

    def test_trajectory_features_change_lessons_with_same_endpoints(self):
        first = ReflectionCase("books", 2, "corrected", (1.0, 2.0),
                               ambiguity="close_scores", trajectory=(0.2, 0.3, 0.0))
        second = ReflectionCase("books", 2, "corrected", (1.0, 2.0),
                                ambiguity="close_scores", trajectory=(0.9, 0.8, 1.0))
        one = reflect([first], snapshot_id="v1", source_split="train",
                      min_support=1, expires_on="2030-01-01")
        two = reflect([second], snapshot_id="v1", source_split="train",
                      min_support=1, expires_on="2030-01-01")
        self.assertNotEqual(one.digest, two.digest)
        self.assertNotEqual(one.lessons[0].trajectory, two.lessons[0].trajectory)
        vectors, ids = retrieve_memory(one, scope="books", history_count=2,
            as_of="2029-01-01", top_k=2, input_dim=2,
            candidate_scores=torch.tensor([10.0, 0.0]))
        self.assertEqual(ids, ())

    def test_round_trip_checks_integrity(self):
        snapshot = reflect(
            [ReflectionCase("books", 1, "corrected", (1.0, 2.0))],
            snapshot_id="v1", source_split="train", min_support=1,
            expires_on="2030-01-01")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            save_snapshot(snapshot, path)
            self.assertEqual(load_snapshot(path).digest, snapshot.digest)
            path.write_text(path.read_text().replace('"v1"', '"changed"'))
            with self.assertRaises(ValueError):
                load_snapshot(path)


if __name__ == "__main__":
    unittest.main()
