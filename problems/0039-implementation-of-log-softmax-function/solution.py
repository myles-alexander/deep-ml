import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	val = np.log(np.sum(np.exp(scores-np.max(scores,keepdims=True)),keepdims=True))
	return scores - np.max(scores,keepdims=True) - val