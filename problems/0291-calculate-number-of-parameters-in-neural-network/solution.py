def count_parameters(layers: list) -> int:
    """
    Count the total number of trainable parameters in a neural network.
    
    Args:
        layers: A list of dictionaries describing each layer.
                Each dict contains 'type' and layer-specific parameters.
    
    Returns:
        Total number of trainable parameters as an integer.
    """
    # Your code here
    count = 0
    for layer in layers:
        if layer['type'] == 'dense':
            count += (layer['input_size'] * layer['output_size'] + layer['output_size']*layer['use_bias'])
        elif layer['type'] == 'conv2d':
            if isinstance(layer['kernel_size'], int): 
                layer['kernel_size'] = (layer['kernel_size'],)
            else:
                pass
            count += (layer['kernel_size'][0]*layer['kernel_size'][0]*layer['in_channels']*layer['out_channels']+layer['use_bias']*layer['out_channels'])
        else:
            count += layer['num_embeddings'] * layer['embedding_dim']
    return count