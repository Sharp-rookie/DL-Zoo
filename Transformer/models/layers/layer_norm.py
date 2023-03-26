import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization, which normalizes the input to the activation function for each layer.

    Args:
        d_model: the dimension of embedding
        eps: a value added to the denominator for numerical stability

    Inputs: x
        - **x**: (batch_size, seq_len, d_model)
    
    Outputs: x
        - **x**: (batch_size, seq_len, d_model)

    Examples:
        >>> layer_norm = LayerNorm(512)
        >>> x = torch.randn(32, 50, 512)
        >>> x = layer_norm(x)
        >>> x.size()
        torch.Size([32, 50, 512])
    """

    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2