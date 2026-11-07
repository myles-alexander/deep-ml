import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    scores = np.array(scores)
    maxx = np.max(scores,keepdims=True)
    adjusted = scores - maxx
    exp_adj = np.exp(adjusted)
    exp = np.exp(adjusted)
    return exp / np.sum(exp_adj,keepdims=True)