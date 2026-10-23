import numpy as np

def pos_encoding(position: int, d_model: int):
	if position == 0 or d_model <= 0:
		return -1
    
    # Initialize the positional encoding matrix
    pe = np.zeros((position, d_model))
    
    # Generate position indices (col vector) and depth indices (row vector)
    pos = np.arange(position)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    
    # Compute the internal angles using the standard 10000^(2i / d_model) formula
    # (i // 2) * 2 ensures that the same exponent rate is shared by paired sin/cos terms
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    angle_rads = pos * angle_rates
    
    # Apply sin to even indices and cos to odd indices
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    # Cast and return as float16
    return pe.astype(np.float16)
