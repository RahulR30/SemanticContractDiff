import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import patch, MagicMock
from main.LLM_Orchestrator import Orchestrator

class TestOrchestrator(unittest.TestCase):

    def setUp(self):
        self.orch = Orchestrator(threshold=0.95)

    def test_low_score_triggers_llm(self):
        with patch.object(self.orch, "call_llm", return_value="High legal impact.") as mock_llm:
            results = self.orch.analyze(["old text"], ["new text"], [0.80])
            mock_llm.assert_called_once()
            self.assertEqual(len(results), 1)

    def test_high_score_skips_llm(self):
        with patch.object(self.orch, "call_llm", return_value="irrelevant") as mock_llm:
            results = self.orch.analyze(["old text"], ["new text"], [0.97])
            mock_llm.assert_not_called()
            self.assertEqual(len(results), 0)

    def test_score_exactly_at_threshold_skips_llm(self):
        with patch.object(self.orch, "call_llm", return_value="irrelevant") as mock_llm:
            results = self.orch.analyze(["old text"], ["new text"], [0.95])
            mock_llm.assert_not_called()

    def test_mixed_scores_flags_correct_clauses(self):
        scores = [0.80, 0.97, 0.60, 0.99]
        paragraphs_a = ["a1", "a2", "a3", "a4"]
        paragraphs_b = ["b1", "b2", "b3", "b4"]

        with patch.object(self.orch, "call_llm", return_value="some analysis"):
            results = self.orch.analyze(paragraphs_a, paragraphs_b, scores)

        flagged_indices = [r["clause_index"] for r in results]
        self.assertEqual(flagged_indices, [0, 2])

    def test_result_has_required_keys(self):
        with patch.object(self.orch, "call_llm", return_value="Legal impact found."):
            results = self.orch.analyze(["original"], ["revised"], [0.80])
            keys = results[0].keys()
            for key in ["clause_index", "original_text", "revised_text", "score", "analysis"]:
                self.assertIn(key, keys)

    def test_result_score_is_rounded(self):
        with patch.object(self.orch, "call_llm", return_value="Impact."):
            results = self.orch.analyze(["a"], ["b"], [0.812345])
            self.assertEqual(results[0]["score"], 0.8123)

    def test_empty_inputs_return_empty_list(self):
        results = self.orch.analyze([], [], [])
        self.assertEqual(results, [])

    def test_mismatched_lengths_raise_error(self):
        with self.assertRaises(ValueError):
            self.orch.analyze(["a", "b"], ["x"], [0.80, 0.70])

    def test_custom_threshold_respected(self):
        orch_strict = Orchestrator(threshold=0.70)
        with patch.object(orch_strict, "call_llm", return_value="Impact.") as mock_llm:
            orch_strict.analyze(["a"], ["b"], [0.75])
            mock_llm.assert_not_called()  # 0.75 >= 0.70, should skip


def run_manual_test():
    orch = Orchestrator()

    paragraphs_a = [
        "The party shall provide notice within 30 days.",
        "Payment shall be made in USD."
    ]
    paragraphs_b = [
        "The party will provide notice within 60 days.",
        "Payment will be made in USD."
    ]
    scores = [0.81, 0.97]

    results = orch.analyze(paragraphs_a, paragraphs_b, scores)
    print("\n--- RESULTS ---")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import json
    print("=== Running unit tests ===\n")
    unittest.main(argv=[''], exit=False, verbosity=2)
    print("\n=== Running manual LLM test ===\n")
    run_manual_test()