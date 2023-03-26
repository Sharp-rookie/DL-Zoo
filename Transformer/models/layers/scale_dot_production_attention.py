import math
import torch.nn as nn
import torch.nn.functional as F


class ScaleDotProductionAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self):
        super(ScaleDotProductionAttention, self).__init__()

    def forward(self, q, k, v, mask=None):
        """
        Args:
            query: [batch_size, n_heads, seq_len, d_k]
            key: [batch_size, n_heads, seq_len, d_k]
            value: [batch_size, n_heads, seq_len, d_k]
            mask: [batch_size, 1, seq_len, seq_len]

        Returns:
            output: [batch_size, n_heads, seq_len, d_k]
            score: [batch_size, n_heads, seq_len, seq_len]
        """

        batch_size, n_heads, seq_len, d_model = q.size()

        # 1. dot product Query and Key^T to compute similarity scores
        k_t = k.transpose(-2, -1)  # transpose
        score = q @ k_t / math.sqrt(d_model)  # scale

        # 2. apply mask to scores
        if mask is not None:
            score = score.masked_fill(mask == 0, 1e-12)  # mask out

        # 3. apply softmax to get attention weights
        score = F.softmax(score, dim=-1)
        
        # 4. multiply attention weights with Value to get output
        output = score @ v
        
        return output, score