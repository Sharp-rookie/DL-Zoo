import torch.nn as nn

from models.layers.scale_dot_production_attention import ScaleDotProductionAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention

    Args:
        d_model: the dimension of input and output
        n_heads: the number of heads

    Returns:
        output: [batch_size, seq_len, d_model]
        score: [batch_size, n_heads, seq_len, seq_len]
    """

    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dk = dv = dq = d_model / n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.attention = ScaleDotProductionAttention()

        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len, seq_len]

        Returns:
            output: [batch_size, seq_len, d_model]
            score: [batch_size, n_heads, seq_len, seq_len]
        """

        batch_size = q.size(0)

        # 1. linear transformation
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # 2. split into n_heads, [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 3. scale dot production attention
        output, score = self.attention(q, k, v, mask)

        # 4. concat, [batch_size, n_heads, seq_len, d_k] -> [batch_size, seq_len, n_heads, d_k] -> [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5. linear transformation
        output = self.w_concat(output)

        return output, score