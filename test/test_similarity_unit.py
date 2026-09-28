from unittest.mock import patch

import numpy as np
import pytest

from main.score_map import compute_similarity


def test_empty_inputs_do_not_load_model():
    with patch('main.score_map.get_model') as model:
        assert compute_similarity([], []) == []
    model.assert_not_called()


def test_mismatched_lengths_fail_before_loading_model():
    with patch('main.score_map.get_model') as model:
        with pytest.raises(ValueError):
            compute_similarity(['a'], [])
    model.assert_not_called()


def test_scores_are_pairwise_cosines_not_cross_document_matches():
    with patch('main.score_map.get_model') as model:
        model.return_value.encode.side_effect = [
            np.array([[1., 0.], [1., 0.], [-1., 0.]]),
            np.array([[2., 0.], [0., 2.], [1., 0.]]),
        ]
        result = compute_similarity(['a', 'b', 'c'], ['x', 'y', 'z'])
    assert result == pytest.approx([1., 0., -1.])
