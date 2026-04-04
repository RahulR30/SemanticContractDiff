""" Pooja Ramakrishnan: test_score_map.py """

import unittest
from main.score_map import compute_similarity

class TestComputeSimilarity(unittest.TestCase):
    """Tests compute_similarity function"""

    def test_unequal_length_lists(self) -> None:
        """Tests if function raises an error for unequal length lists"""
        list_a = ["Hello my name is", "how is your day"]
        list_b = ["I'm doing good"]

        with self.assertRaises(ValueError):
            compute_similarity(list_a, list_b)
    
    def test_exacty_equal_lists(self) -> None:
        """Tests if the cosine similarity is 1 for exactly same lists"""
        list_a = ["I think this candy bar is good", "This candy bar is excellent"]
        list_b = ["I think this candy bar is good", "This candy bar is excellent"]

        result = compute_similarity(list_a, list_b)

        for score in result:
            self.assertAlmostEqual(score, 1.0, places=2)
    
    def test_unequal_lists(self) -> None:
        """Tests if the cosine similarity is a low number for different lists"""
        list_a = ["I think this candy bar is good", "This candy bar is excellent"]
        list_b = ["This candy bar tastes like cardboard", "This candy bar is horrible"]

        result = compute_similarity(list_a, list_b)

        for score in result:
            self.assertLess(score, 0.9)


if __name__ == "__main__":
    unittest.main()