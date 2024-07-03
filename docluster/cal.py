import numpy as np


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


matrix = np.load('mean.npy')
txt = np.loadtxt("0.x")

similarities = [cosine_similarity(txt, row) for row in matrix]

print(similarities)
