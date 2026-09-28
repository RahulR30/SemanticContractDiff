""" Pooja Ramakrishnan: score_map.py """

from sklearn.metrics.pairwise import cosine_similarity

from functools import lru_cache


@lru_cache(maxsize=1)
def get_model():
    """Load weights only when nonempty text actually needs scoring."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

def compute_similarity(list_a: list[str], list_b: list[str]) -> list[float]:
    """Determines how similar the indexed paragraphs are to each other"""

    if len(list_a) != len(list_b):
        raise ValueError("Both lists must have the same length")
    
    if not list_a:
        return []

    model = get_model()
    sen_embedding_b = model.encode(list_b)
    sen_embedding_a = model.encode(list_a)

    score_map: list[float] = []

    for i in range(len(list_a)):
        score = cosine_similarity(sen_embedding_a[i].reshape(1, -1), sen_embedding_b[i].reshape(1, -1))[0][0]
        score_map.append(float(score))

    return score_map