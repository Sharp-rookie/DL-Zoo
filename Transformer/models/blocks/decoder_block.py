import torch.nn as nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention 
from models.layers.position_wise_feed_forward import PositionWiseFeedForward


class DecoderBlock(nn.Module):
    """
    Decoder Block, which is composed of output & input-output Multi-Head Attention and Position-wise Feed Forward Network.
    The output of this module is the input for the next layer.

    Args:
        d_model (int): The number of expected features in the input
        n_heads (int): Number of heads
        d_ff (int): Feed forward dimension
        dropout (float): Dropout rate
        device (str): The device of the model

    Inputs: x_out, x_in, out_mask, in_mask
        - **x_out** (FloatTensor): ``(batch_size, seq_length, d_model)``
        - **x_in** (FloatTensor): ``(batch_size, seq_length, d_model)``
        - **out_mask** (ByteTensor): ``(batch_size, 1, seq_length, seq_length)``
        - **in_mask** (ByteTensor): ``(batch_size, 1, seq_length, seq_length)``

    Outputs: x
        - **x** (FloatTensor): ``(batch_size, seq_length, d_model)``

    Examples:
        >>> decoder_block = DecoderBlock(d_model=512, n_heads=8, d_ff=2048, dropout=0.1)
        >>> x_out = torch.FloatTensor(2, 3, 512)
        >>> x_out.size()
        torch.Size([2, 3, 512])
        >>> x_in = torch.FloatTensor(2, 3, 512)
        >>> x_in.size()
        torch.Size([2, 3, 512])
        >>> out_mask = torch.ByteTensor(2, 1, 3).fill_(1)
        >>> out_mask.size()
        torch.Size([2, 1, 3])
        >>> in_mask = torch.ByteTensor(2, 1, 3).fill_(1)
        >>> in_mask.size()
        torch.Size([2, 1, 3])
        >>> decoder_block(x_out, x_in, out_mask, in_mask).size()
        torch.Size([2, 3, 512])
    """

    def __init__(self, d_model, n_heads, d_ff, dropout, device):
        super(DecoderBlock, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.layer_norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.in_out_attention = MultiHeadAttention(d_model, n_heads)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x_out, x_in, out_mask, in_mask):

        # 1. compute self-attention
        residual = x_out
        x_out, _ = self.self_attention(q=x_out, k=x_out, v=x_out, mask=out_mask)

        # 2. add & norm
        x_out = self.dropout1(x_out)
        x_out = self.layer_norm1(residual + x_out)
        
        if x_in is not None:
            # 3. compute encoder - decoder attention
            residual = x_out
            x_out, _ = self.in_out_attention(q=x_out, k=x_in, v=x_in, mask=in_mask)

            # 4. add & norm
            x_out = self.dropout2(x_out)
            x_out = self.layer_norm2(residual + x_out)
        
        # 5. compute feed forward
        residual = x_out
        x_out = self.feed_forward(x_out)

        # 6. add & norm
        x_out = self.dropout3(x_out)
        x_out = self.layer_norm3(residual + x_out)
                             
        return x_out