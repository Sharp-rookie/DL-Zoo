import torch.nn as nn

from models.embeddings.token_embedding import TokenEmbedding
from models.embeddings.positional_embedding import PositionalEmbedding


class TransformerEmbedding(nn.Module):
    """
    Transformer Embedding, the embedding of the input sequence.

    Args:
        vocab_size: the size of vocabulary
        d_model: the dimension of embedding
        max_length: the maximum length of the input sequence in the whole dataset
        pad_idx: the index of padding token
        dropout: the dropout rate
        device: the device of the model

    Inputs: x
        - **x**: (batch_size, seq_len)

    Outputs: x
        - **x**: (batch_size, seq_len, d_model)

    Examples:
        >>> transformer_embedding = TransformerEmbedding(vocab_size=100, d_model=512, max_length=100, pad_idx=0, dropout=0.1, device='cpu')
        >>> x = torch.LongTensor(2, 3).random_(0, 100)
        >>> x.size()
        torch.Size([2, 3])
        >>> transformer_embedding(x).size()
        torch.Size([2, 3, 512])
    """

    def __init__(self, vocab_size, d_model, max_length, pad_idx, dropout, device):
        super(TransformerEmbedding, self).__init__()

        self.token_embedding = TokenEmbedding(vocab_size, d_model, pad_idx, device)
        self.positional_embedding = PositionalEmbedding(d_model, max_length, device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.token_embedding(x) + self.positional_embedding(x)
        return self.dropout(x)