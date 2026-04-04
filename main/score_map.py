""" Pooja Ramakrishnan: score_map.py """

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def compute_similarity(list_a: list[str], list_b: list[str]) -> list[float]:
    """Determines how similar the indexed paragraphs are to each other"""

    if len(list_a) != len(list_b):
        raise ValueError("Both lists must have the same length")
    
    sen_embedding_b = model.encode(list_b)
    sen_embedding_a = model.encode(list_a)

    score_map: list[float] = []

    for i in range(len(list_a)):
        score = cosine_similarity(sen_embedding_a[i].reshape(1, -1), sen_embedding_b[i].reshape(1, -1))[0][0]
        score_map.append(float(score))

    return score_map