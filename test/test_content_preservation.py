"""Regression cases for PDF content that must not disappear during comparison."""
from unittest.mock import MagicMock, patch

import fitz
import pytest

from main.extract_paragraphs import ExtractParagraphs


def test_real_single_line_pdf_is_not_discarded(tmp_path):
    filename = tmp_path / 'one-line.pdf'
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), 'Payment is due within 30 days.')
        doc.save(filename)
    assert ExtractParagraphs(str(filename)).text == 'Payment is due within 30 days.'


def test_first_and_last_clauses_survive(tmp_path):
    filename = tmp_path / 'clauses.pdf'
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), 'First obligation\nMiddle obligation\nFinal obligation')
        doc.save(filename)
    assert ExtractParagraphs(str(filename)).text.splitlines() == [
        'First obligation', 'Middle obligation', 'Final obligation']


def test_document_closes_when_page_extraction_raises():
    page = MagicMock()
    page.get_text.side_effect = RuntimeError('unreadable page')
    pdf = MagicMock()
    pdf.__iter__.return_value = iter([page])
    with patch('main.extract_paragraphs.fitz.open', return_value=pdf):
        with pytest.raises(RuntimeError, match='unreadable page'):
            ExtractParagraphs('bad.pdf')
    pdf.close.assert_called_once()
