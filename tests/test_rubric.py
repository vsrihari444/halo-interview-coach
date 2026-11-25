import json
import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from coach.rubric import auto_check, extract_json, fabricated_numbers, schema_errors  # noqa: E402

SEED = [json.loads(l) for l in open(os.path.join(ROOT, "data/seed/seed.jsonl"), encoding="utf-8") if l.strip()]


class SeedTests(unittest.TestCase):
    def test_seed_size_and_ids(self):
        self.assertGreaterEqual(len(SEED), 50)
        self.assertEqual(len({r["id"] for r in SEED}), len(SEED))

    def test_seed_passes_all_checks(self):
        for ex in SEED:
            res = auto_check(json.dumps(ex["feedback"]), ex)
            self.assertTrue(res["schema_valid"], (ex["id"], res))
            self.assertTrue(res["grounded"], (ex["id"], res))
            self.assertEqual(res["score_mae_vs_reference"], 0)
            if "calibrated" in res:
                self.assertTrue(res["calibrated"], ex["id"])


class CheckTests(unittest.TestCase):
    def test_extract_json_from_fenced_text(self):
        self.assertEqual(extract_json('Sure!\n```json\n{"a": 1}\n```'), {"a": 1})
        self.assertIsNone(extract_json("no json here"))

    def test_fabricated_metric_is_flagged(self):
        fb = {"rewrite_tip": "Say you cut latency by 40% at Google."}
        self.assertEqual(fabricated_numbers(fb, "I made the API faster.", []), ["40"])

    def test_placeholders_and_confirmed_facts_are_allowed(self):
        fb = {"rewrite_tip": "Say you cut CI from 25 minutes to 9 minutes and improved [metric] by [x%]."}
        self.assertEqual(fabricated_numbers(fb, "I made tests faster.", ["CI went from 25 minutes to 9 minutes"]), [])

    def test_schema_errors(self):
        self.assertTrue(schema_errors({"scores": {"clarity": 9}}))


if __name__ == "__main__":
    unittest.main()
