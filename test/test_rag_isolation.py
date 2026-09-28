from types import SimpleNamespace
from unittest.mock import patch

import pytest
pytest.importorskip('langchain_chroma')
pytest.importorskip('langchain_huggingface')
pytest.importorskip('langchain_openai')
from langchain_core.embeddings import Embeddings
from main.rag_pipeline import run_rag_pipeline


class FixedEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [1., 0.] if text == 'old original' else [0., 1.]


def test_reused_directory_does_not_retrieve_previous_contracts(tmp_path):
    with patch('main.vector_store.get_embeddings', return_value=FixedEmbeddings()), \
         patch('main.rag_pipeline.ExtractParagraphs') as extract:
        extract.side_effect = [SimpleNamespace(text=t) for t in [
            'old original', 'old revision', 'new original', 'old original'
        ]]
        run_rag_pipeline('a.pdf', 'b.pdf', persist_directory=str(tmp_path), call_llm=False)
        second = run_rag_pipeline('c.pdf', 'd.pdf', persist_directory=str(tmp_path), call_llm=False)
    assert len(second) == 1
    assert second[0]['original_text'] == 'new original'
    assert second[0]['score'] == 0.0


def test_empty_pdf_reports_missing_text_instead_of_no_changes():
    with patch('main.rag_pipeline.ExtractParagraphs') as extract:
        extract.side_effect = [SimpleNamespace(text=''), SimpleNamespace(text='new clause')]
        with pytest.raises(ValueError, match='extractable text'):
            run_rag_pipeline('a.pdf', 'b.pdf', call_llm=False)
