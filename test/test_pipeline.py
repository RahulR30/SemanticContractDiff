from unittest.mock import patch

import pytest

from main.pipeline import run_pipeline


@pytest.mark.parametrize('original,revised,expected', [
    (['same'], ['same', 'new clause'], ('', 'new clause')),
    (['same', 'removed clause'], ['same'], ('removed clause', '')),
    ([], ['new clause'], ('', 'new clause')),
    (['removed clause'], [], ('removed clause', '')),
])
def test_unmatched_trailing_clauses_are_analyzed(original, revised, expected):
    with patch('main.pipeline.ExtractParagraphs') as extract, \
         patch('main.pipeline.compute_similarity', return_value=[1.0] * min(len(original), len(revised))), \
         patch('main.LLM_Orchestrator.Orchestrator.call_llm', return_value='Changed') as llm:
        extract.return_value.text_to_paragraph.side_effect = [original, revised]
        results = run_pipeline('a.pdf', 'b.pdf')
    llm.assert_called_once_with(*expected)
    assert len(results) == 1
    assert results[0]['score'] == 0.0
    assert (results[0]['original_text'], results[0]['revised_text']) == expected


def test_empty_contracts_produce_no_analysis():
    with patch('main.pipeline.ExtractParagraphs') as extract, \
         patch('main.score_map.get_model') as model, \
         patch('main.LLM_Orchestrator.Orchestrator.call_llm') as llm:
        extract.return_value.text_to_paragraph.return_value = []
        assert run_pipeline('a.pdf', 'b.pdf') == []
    model.assert_not_called()
    llm.assert_not_called()
