import torch.nn as nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention 
from models.layers.position_wise_feed_forward import PositionWiseFeedForward


class EncoderBlock(nn.Module):
    """
    Encoder Block, which is composed of input Multi-Head Attention and Position-wise Feed Forward Network.
    The output of this module is the input for the next layer.

    Args:
        d_model (int): The number of expected features in the input
        n_heads (int): Number of heads
        d_ff (int): Feed forward dimension
        dropout (float): Dropout rate
        device (str): The device of the model

    Inputs: x, mask
        - **x** (FloatTensor): ``(batch_size, seq_length, d_model)``
        - **mask** (ByteTensor): ``(batch_size, 1, seq_length, seq_length)``

    Outputs: x
        - **x** (FloatTensor): ``(batch_size, seq_length, d_model)``

    Examples:
        >>> encoder_block = EncoderBlock(d_model=512, n_heads=8, d_ff=2048, dropout=0.1)
        >>> x = torch.FloatTensor(2, 3, 512)
        >>> x.size()
        torch.Size([2, 3, 512])
        >>> mask = torch.ByteTensor(2, 1, 3).fill_(1)
        >>> mask.size()
        torch.Size([2, 1, 3])
        >>> encoder_block(x, mask).size()
        torch.Size([2, 3, 512])
    """

    def __init__(self, d_model, n_heads, d_ff, dropout, device):
        super(EncoderBlock, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.layer_norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, in_mask):

        # 1. compute self-attention
        residual = x
        x, _ = self.self_attention(q=x, k=x, v=x, mask=in_mask)

        # 2. add & norm
        x = self.dropout1(x)
        x = self.layer_norm1(residual + x)
        
        # 3. position-wise feed forward
        residual = x
        x = self.feed_forward(x)

        # 4. add & norm
        x = self.dropout2(x)
        x = self.layer_norm2(residual + x)
                             
        return x