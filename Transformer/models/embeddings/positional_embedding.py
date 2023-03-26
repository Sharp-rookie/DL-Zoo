import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Positional Embedding, sinusoid encoding for the position of words, which gives positional information to the model.

    Args:
        d_model: the dimension of embedding
        max_length: the maximum length of the input sequence in the whole dataset
        device: the device of the model

    Inputs: x
        - **x**: (batch_size, seq_len)
    
    Outputs: x
        - **x**: (batch_size, seq_len, d_model)

    Examples:
        >>> positional_embedding = PositionalEmbedding(512, 100, 'cpu')
        >>> x = torch.randn(32, 50)
        >>> x = positional_embedding(x)
        >>> x.size()
        torch.Size([32, 50, 512])
    """

    def __init__(self, d_model, max_length, device):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_length, d_model, dtype=torch.float32, device=device)
        
        position = torch.arange(0, max_length, dtype=torch.float32, device=device).unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2, dtype=torch.float32, device=device)

        pe[:, 0::2] = torch.sin(position * (10000 ** (-_2i / d_model)))
        pe[:, 1::2] = torch.cos(position * (10000 ** (-_2i / d_model)))
        
        pe.requires_grad = False  # no need to train in original transformer (but maybe trainable in other tasks!!!)
        self.register_buffer('pe', pe)

    
    def forward(self, x):

        batch_size, seq_len = x.size()
        return self.pe[:seq_len, :]

