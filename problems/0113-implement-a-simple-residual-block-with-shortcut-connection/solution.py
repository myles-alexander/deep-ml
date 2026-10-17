import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	# Your code here
	y = x @ w1
	y = y @ w2 
	y = np.where(y>0,y,0)
	y = y + x
	y = np.where(y>0,y,0)
	return y