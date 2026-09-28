"""UI smoke coverage without downloads or external model requests."""
from unittest.mock import patch
from pathlib import Path
import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest


def test_sample_diff_can_run_without_api_key(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()
    assert not app.exception
    app.checkbox[0].check().run()
    assert app.checkbox[1].value is False
    with patch('main.rag_pipeline.run_rag_pipeline', return_value=[{
        'clause_index': 0, 'original_text': '<b>old</b>', 'revised_text': '<b>new</b>',
        'score': 0.5, 'analysis': 'Model explanation disabled.'
    }]) as compare:
        app.button[0].click().run()
    assert not app.exception
    assert compare.call_args.kwargs['call_llm'] is False
    assert any('&lt;b&gt;old&lt;/b&gt;' in m.value for m in app.markdown)
