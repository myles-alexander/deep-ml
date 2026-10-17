import numpy as np

def ffn(x: list[float], W1: list[list[float]], b1: list[float], W2: list[list[float]], b2: list[float], dropout_p: float=0.1, seed: int=42) -> list[float]:
	"""
	Implement a position-wise feed-forward block with residual and dropout.

	Args:
		x: input vector
		W1, b1: first linear layer parameters
		W2, b2: second linear layer parameters
		dropout_p: dropout probability
		seed: random seed for reproducibility

	Returns:
		Output vector after FFN block (rounded to 4 decimals)
	"""
	# Your code here
	y = np.array(x) @ np.array(W1).T + np.array(b1)
	y = np.where(y>0,y,0)
	y = y @ np.array(W2).T + np.array(b2)

	keep_prob = 1-dropout_p
	np.random.seed(seed)
	mask = np.random.rand(*y.shape) > dropout_p

	y = (y * mask) / keep_prob
	return y + x