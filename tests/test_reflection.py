import copy
import json
import tempfile
import unittest
from pathlib import Path
import torch
from ccid.reflect import build_memory


class ReflectionTests(unittest.TestCase):
    def test_intermediate_messages_matter_and_bad_lineage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = dict(
                request_id="request", candidate_ids=["a", "b"],
                candidate_states=torch.tensor([[1., 0.], [0., 1.]]),
                history_states=torch.empty(0, 2),
                base_scores=torch.tensor([1., 0.]))
            torch.save({"split":"train","episodes":[episode]}, root/"episodes.pt")
            (root/"labels.jsonl").write_text(
                json.dumps({"request_id":"request","target_id":"b"})+"\n")
            trace = dict(
                request_id="request", candidate_ids=["a","b"],
                initial_ranking=["a","b"], ranking=["b","a"],
                permitted_calls=6, realized_calls=6,
                rounds=[
                    dict(round_index=0, incumbent_id="a", ranking=["b","a"],
                         utility=[0.2,0.8], risk=[0.0,0.1],
                         disagreement=0.08, calls=3),
                    dict(round_index=1, incumbent_id="b", ranking=["b","a"],
                         utility=[0.1,0.9], risk=[0.8,0.0],
                         disagreement=0.08, calls=3)])
            def build(row, name):
                (root/"trace.jsonl").write_text(json.dumps(row)+"\n")
                return build_memory(
                    root/"episodes.pt", root/"trace.jsonl", root/"labels.jsonl",
                    scope="demo", snapshot_id="v1", min_support=1,
                    expires_on="2030-01-01", output=root/name)
            one = build(trace,"one.json")
            changed = copy.deepcopy(trace)
            changed["rounds"][0]["utility"] = [0.9,0.1]
            changed["rounds"][0]["risk"] = [0.3,0.6]
            changed["rounds"][0]["disagreement"] = 0.27
            two = build(changed,"two.json")
            self.assertNotEqual(one.lessons[0].vector, two.lessons[0].vector)
            self.assertNotEqual(one.lessons[0].trajectory, two.lessons[0].trajectory)
            bad = copy.deepcopy(trace)
            bad["candidate_ids"] = ["b","a"]
            with self.assertRaises(ValueError):
                build(bad,"bad.json")
            bad = copy.deepcopy(trace)
            bad["rounds"][1]["incumbent_id"] = "a"
            with self.assertRaises(ValueError):
                build(bad,"bad2.json")
            bad = copy.deepcopy(trace)
            bad["realized_calls"] = 3
            with self.assertRaises(ValueError):
                build(bad,"bad3.json")


if __name__ == "__main__":
    unittest.main()
