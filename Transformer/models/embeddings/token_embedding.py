"""
Only used for NLP tasks
"""
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token Embedding, which is a simple lookup table that stores embeddings of a fixed dictionary and size.
    Convert the input tokens (integers) of different length to embeddings (floats) with same length(512).

    Args:
        vocab_size (int): Size of the vocabulary
        embed_dim (int): Embedding dimension
        pad_idx (int): Padding index
        device (str): The device of the model

    Inputs: x
        - **x** (LongTensor): ``(batch_size, seq_length)``
    
    Outputs: x
        - **x** (FloatTensor): ``(batch_size, seq_length, embed_dim)``

    Examples:
        >>> token_embedding = TokenEmbedding(vocab_size=100, embed_dim=512, pad_idx=1)
        >>> x = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
        >>> x.size()
        torch.Size([2, 3])
        >>> token_embedding(x).size()
        torch.Size([2, 3, 512])
    """

    def __init__(self, vocab_size, d_model, pad_idx, device):
        super(TokenEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=pad_idx, device=device)

    def forward(self, x):
        return self.token_embedding(x)