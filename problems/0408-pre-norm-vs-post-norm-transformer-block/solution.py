import numpy as np

def transformer_block(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray, mode: str, eps: float = 1e-5) -> np.ndarray:
	"""
	Apply a transformer block with two sublayers using either pre-norm or post-norm.
	
	Args:
		x: Input array of shape (seq_len, d_model)
		W1, b1: Weights and bias for first sublayer
		W2, b2: Weights and bias for second sublayer
		gamma1, beta1: LayerNorm params for first normalization
		gamma2, beta2: LayerNorm params for second normalization
		mode: 'pre_norm' or 'post_norm'
		eps: Epsilon for numerical stability
	
	Returns:
		Output array of shape (seq_len, d_model)
	"""
	if mode == 'pre_norm':
		mean_x = np.mean(x,axis=-1,keepdims=True)
		var_x = np.var(x,axis=-1,keepdims=True)
		f1 = gamma1*((x-mean_x)/ np.sqrt(var_x+eps)) + beta1
		h = x + (f1 @ W1 + b1)

		mean_h = np.mean(h,axis=-1,keepdims=True)
		var_h = np.var(h,axis=-1,keepdims=True)
		fh = gamma2*((h-mean_h)/ np.sqrt(var_h+eps)) + beta2
		y = h + (fh @ W2 + b2)
		return y
	elif mode == 'post_norm':
		h = x + (x @ W1 + b1)
		mean_h = np.mean(h,axis=-1,keepdims=True)
		var_h = np.var(h,axis=-1,keepdims=True)
		f1 = gamma1*((h-mean_h)/ np.sqrt(var_h+eps)) + beta1

		y = f1 + (f1 @ W2 + b2)
		mean_y = np.mean(y,axis=-1,keepdims=True)
		var_y = np.var(y,axis=-1,keepdims=True)
		f2 = gamma2*((y-mean_y)/ np.sqrt(var_y+eps)) + beta2
		return f2
	else:
		raise ValueError

		