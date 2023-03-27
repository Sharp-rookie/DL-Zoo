import torch
import torch.nn as nn

from models.blocks.encoder_block import EncoderBlock
from models.blocks.decoder_block import DecoderBlock
from models.embeddings.transformer_embedding import TransformerEmbedding


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
    if hasattr(m, 'bias'):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Encoder(nn.Module):
    """Encoder Module of the Transformer model"""

    def __init__(self, vocab_size, max_length, in_pad_idx, d_model, n_heads, d_ff, n_blocks, dropout, device):
        super(Encoder, self).__init__()

        self.embedding = TransformerEmbedding(vocab_size=vocab_size, 
                                              d_model=d_model, 
                                              max_length=max_length, 
                                              pad_idx=in_pad_idx, 
                                              dropout=dropout, 
                                              device=device)
        
        self.blocks = nn.ModuleList([EncoderBlock(d_model=d_model, 
                                                  n_heads=n_heads, 
                                                  d_ff=d_ff, 
                                                  dropout=dropout, 
                                                  device=device) for _ in range(n_blocks)])

    def forward(self, x, in_mask):
        
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, in_mask)
        
        return x
    

class Decoder(nn.Module):
    """Decoder Module of the Transformer model"""

    def __init__(self, vocab_size, max_length, out_pad_idx, d_model, n_heads, d_ff, n_blocks, dropout, device):
        super(Decoder, self).__init__()

        self.embedding = TransformerEmbedding(vocab_size=vocab_size, 
                                              d_model=d_model, 
                                              max_length=max_length, 
                                              pad_idx=out_pad_idx, 
                                              dropout=dropout, 
                                              device=device)
        
        self.blocks = nn.ModuleList([DecoderBlock(d_model=d_model, 
                                                  n_heads=n_heads, 
                                                  d_ff=d_ff, 
                                                  dropout=dropout, 
                                                  device=device) for _ in range(n_blocks)])

        self.linear = nn.Linear(d_model, vocab_size)  # Linear layer to map the decoder output to the vocab size

    def forward(self, x_out, x_in, out_mask, in_mask):
        
        x_out = self.embedding(x_out)
        for block in self.blocks:
            x_out = block(x_out, x_in, out_mask, in_mask)

        output = self.linear(x_out)
        
        return output


class Transformer(nn.Module):
    """
    Transformer model for NLP tasks.

    Args:
        vocab_size (list): List of the vocabulary size of the input and output sequences.
        max_length (int): Maximum length of the input and output sequences.
        pad_idx (list): List of the padding index of the input and output sequences.
        d_model (int): Dimension of the model.
        n_heads (int): Number of heads in the multi-head attention layer.
        d_ff (int): Dimension of the feed forward layer.
        n_blocks (int): Number of encoder and decoder blocks.
        dropout (float): Dropout rate.
        device (torch.device): Device to run the model on.
    """

    def __init__(self, vocab_sizes, max_length, pad_idxes, d_model, n_heads, d_ff, n_blocks, dropout, device):
        super(Transformer, self).__init__()

        self.pad_idxes = pad_idxes  # Padding index of the vocabulary (only used for the nn.embedding in nlp tasks)

        self.encoder = Encoder(vocab_size=vocab_sizes[0], 
                               max_length=max_length, 
                               in_pad_idx=pad_idxes[0], 
                               d_model=d_model, 
                               n_heads=n_heads, 
                               d_ff=d_ff, 
                               n_blocks=n_blocks, 
                               dropout=dropout, 
                               device=device)
        
        self.decoder = Decoder(vocab_size=vocab_sizes[1], 
                               max_length=max_length, 
                               out_pad_idx=pad_idxes[1], 
                               d_model=d_model, 
                               n_heads=n_heads, 
                               d_ff=d_ff, 
                               n_blocks=n_blocks, 
                               dropout=dropout, 
                               device=device)

        self.apply(initialize_weights)

    def forward(self, x_in, x_out):

        in_mask = self.mutual_mask(x_in, x_in, self.pad_idxes[0], self.pad_idxes[0])
        in_out_mask =  self.mutual_mask(x_in, x_out, self.pad_idxes[0], self.pad_idxes[1])
        out_mask = self.mutual_mask(x_out, x_out, self.pad_idxes[1], self.pad_idxes[1]) & self.no_peek_mask(x_out, x_out)

        x_in_out = self.encoder(x_in, in_mask)
        x_out = self.decoder(x_out, x_in_out, out_mask, in_out_mask)
        
        return x_out
    
    def mutual_mask(self, x, y, x_pad_idx, y_pad_idx):
        """
        Mask the padding tokens of the input and output sequences.
        """
        
        x_seq_len, y_seq_len = x.size(1), y.size(1)
        x_mask = (x != x_pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, x_seq_len)
        y_mask = (y != y_pad_idx).unsqueeze(1).unsqueeze(3)  # (batch_size, 1, y_seq_len, 1)
        mask = x_mask.repeat(1, 1, y_seq_len, 1) & y_mask.repeat(1, 1, 1, x_seq_len)  # (batch_size, 1, y_seq_len, x_seq_len)
        
        return mask
    
    def no_peek_mask(self, q, k):
        """
        Mask the down half of the attention matrix to prevent peeking into the future (information from future tokens).
        """
        
        len_q = q.size(1)
        len_k = k.size(1)
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(q.device)
        
        return mask