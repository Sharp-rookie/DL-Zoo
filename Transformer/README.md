## Transformer

to do

&emsp;

## 运行步骤

### Environment Setup

原论文把Transformer用于英文和德文之间的翻译任务，因此这里也使用NLP任务作为示例。这里具体采用torchtext中集成的[Multi30K](https://arxiv.org/abs/1605.00459)数据集，并使用[SpaCy](https://spacy.io/usage/)包来进行标识化（tokenize）。两个包的安装指令为：

```sh
pip install torchtext==0.6.0
pip install spacy==3.5.1
python -m spacy download en
python -m spacy download de
```

### Execution

```sh
python train.py
```

&emsp;

## 实现细节

### Self-Attention Block Components

#### Scaled Dot-Product Attention

<img src="https://my-picture-1311448338.file.myqcloud.com/img/202303261428489.png" alt="image-20230326142849456" style="zoom:50%;" />

常用的注意力评分函数包括2种：加性注意力、乘性注意力两种。

- 加性注意力：

    <img src="https://my-picture-1311448338.file.myqcloud.com/img/202303262108566.png" alt="image-20230326210832492" style="zoom: 67%;" />

    Query和Key不要求等长，`W_q`、`W_k`和`w_v`都是可学习的参数。

- 乘性注意力：

    原文采用的是乘性注意力，因为其运算可以转化为矩阵乘法，计算和存储效率都更高。并除以`d_model`的平方根，从而保证在Qoery维度过大时点积的大小不至于太大导致softmax后梯度消失。

**注意**，在NLP任务中，由于句子不等长所以需要通过补零（padding）来保证输入的向量维度相同，因此在计算注意力权重时需要把补零的元素mask掉。但是在cv、时序数据的其他任务中不一定需要这样。

缩放点积注意力的架构设计和代码实现如下：

<img src="https://my-picture-1311448338.file.myqcloud.com/img/202303261428363.png" alt="image-20230326142825256" style="zoom:50%;" />

```python
class ScaleDotProductionAttention(nn.Module):

    def __init__(self):
        super(ScaleDotProductionAttention, self).__init__()

    def forward(self, q, k, v, mask=None):

        batch_size, n_heads, seq_len, d_model = q.size()

        k_t = k.transpose(-2, -1)
        score = q @ k_t / math.sqrt(d_model)

        if mask is not None:
            score = score.masked_fill(mask == 0, 1e-12)

        score = F.softmax(score, dim=-1)
        
        output = score @ v
        
        return output, score
```



#### Multi-Head Attention

<img src="https://my-picture-1311448338.file.myqcloud.com/img/202303261431183.png" alt="image-20230326143107141" style="zoom:50%;" />

CNN的一个优点在于卷积核的不同层可以提取相同像素区域的不同语义信息。原文使用多头注意力机制以追求类似的效果，即通过不同的线性投影从多个角度来观察Query、Key、Value，从而得到多种信息的组合。同时，为了保证计算开销不会因为头数`h`而大量提升，每个头的输入只包含原本的`d_model`维的`1/h`维。

多头注意力的架构设计和代码实现如下：

<img src="https://my-picture-1311448338.file.myqcloud.com/img/202303261430184.png" alt="image-20230326143041132" style="zoom:50%;" />

```python
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.attention = ScaleDotProductionAttention()

        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        batch_size = q.size(0)

        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        output, score = self.attention(q, k, v, mask)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.w_concat(output)

        return output, score
```

#### Position-wise Feed-Forward Network

<img src="https://my-picture-1311448338.file.myqcloud.com/img/202303261455348.png" alt="image-20230326145519301" style="zoom:50%;" />

一个单层MLP。

位置前向传播的代码实现如下：

```python
class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        output = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return output
```

#### Layer Normalize

BatchNorm以样本的每个特征维度为对象，按照Batch计算mean和std进行归一化，这在batch size太小，或者像NLP这种不同样本（句子）的长度通过补零（padding）来保证等长的数据集上偏差太大。而LayerNorm则是以样本为对象，按照其所有维度的值计算mean和std进行归一化，不会受到样本数量或每个样本实际长度不等的影响（或许在NLP以外的任务上，用BN的Transformer也ok）。

层归一化的代码实现如下：

```python
class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

### Embedding

#### Token Embedding

<img src="C:/Users/lrk/AppData/Roaming/Typora/typora-user-images/image-20230326154803021.png" alt="image-20230326154803021" style="zoom:50%;" />

由于文本需要映射为数字才能被处理，因此文本内容的 Tokenization（标识化）和Embedding是NLP任务的必要内容，而对于CV或时序数据处理的任务则没有这个需要，因为本来就是数字。Tokenize是把一句话划分为单个词语（token），这个过程中句子长度不同时，用padding来补足（attention需要把这些padding给mask掉）。Embedding是维护并训练一个查找表，要通过训练能把意思相近的token映射为相似的数值。

embedding的代码直接用torch.nn封装好的：

```python
class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, pad_idx, device):
        super(TokenEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=pad_idx, device=device)

    def forward(self, x):
        return self.token_embedding(x)
```

#### Positional Embedding

<img src="https://my-picture-1311448338.file.myqcloud.com/img/202303261547683.png" alt="image-20230326154748642" style="zoom:50%;" />

单纯的self-attention虽然能够一次看到所有的历史数据（key-value），但是却无法得知“时序”或“顺序”这一信息，因此对输入数据的前后位置信息进行编码并加入到输入中，保证Transformer有时序信息帮助判断。一个良好的位置编码方案要做到：

- 能为每个时间步输出一个独一无二的编码
- 不同长度的句子之间，任何两个时间步之间的距离应该保持一致
- 模型应该能毫不费力地泛化到更长的句子，因此它的值应该是有界的
- 必须是确定性的

综上，原文作者收到二进制数从大到小排列时，高位到低位上的0、1变化频率由慢到快的这一规律的启发，提出了文中的位置编码方案。**注意**，原文的位置编码设置为不可训练。但是在一些数据预测任务中有其他的位置编码方案，同时是可以一起进行训练的。

位置编码的效果和代码实现如下：

<img src="https://my-picture-1311448338.file.myqcloud.com/img/202303262147964.png" alt="image-20230326214750893" style="zoom: 33%;" />

```python
class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_length, device):
        super(PositionalEmbedding, self).__init__()

        pe = torch.zeros(max_length, d_model, dtype=torch.float32, device=device)
        
        position = torch.arange(0, max_length, dtype=torch.float32, device=device).unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2, dtype=torch.float32, device=device)

        pe[:, 0::2] = torch.sin(position * (10000 ** (-_2i / d_model)))
        pe[:, 1::2] = torch.cos(position * (10000 ** (-_2i / d_model)))
        
        pe.requires_grad = False
        self.register_buffer('pe', pe)

    
    def forward(self, x):

        batch_size, seq_len = x.size()
        return self.pe[:seq_len, :]
```

### Block Stack in Encoder/Decoder

#### Encoder block

编码器部分由多个block组成，每个block包含两个子层：多头自注意力层和位置前向传播层。

<img src="https://my-picture-1311448338.file.myqcloud.com/img/202303261555643.png" alt="image-20230326155520599" style="zoom:50%;" />

```python
class EncoderBlock(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout, device):
        super(EncoderBlock, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.layer_norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, in_mask):

        residual = x
        x, _ = self.self_attention(q=x, k=x, v=x, mask=in_mask)

        x = self.dropout1(x)
        x = self.layer_norm1(residual + x)
        
        residual = x
        x = self.feed_forward(x)

        x = self.dropout2(x)
        x = self.layer_norm2(residual + x)
                             
        return x
```

#### Decoder block

解码器部分由多个block组成，每个block包含三个子层：多头自注意力层、编码信息注意力层和位置前向传播层。

<img src="https://my-picture-1311448338.file.myqcloud.com/img/202303261555723.png" alt="image-20230326155552681" style="zoom:50%;" />

```python
class DecoderBlock(nn.Module):
    
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

        residual = x_out
        x_out, _ = self.self_attention(q=x_out, k=x_out, v=x_out, mask=out_mask)

        x_out = self.dropout1(x_out)
        x_out = self.layer_norm1(residual + x_out)
        
        if x_in is not None:
            residual = x_out
            x_out, _ = self.in_out_attention(q=x_out, k=x_in, v=x_in, mask=in_mask)

            x_out = self.dropout2(x_out)
            x_out = self.layer_norm2(residual + x_out)
        
        residual = x_out
        x_out = self.feed_forward(x_out)

        x_out = self.dropout3(x_out)
        x_out = self.layer_norm3(residual + x_out)
                             
        return x_out
```

#### Transformer

整个架构分为编码器和解码器，编解码器的输入都是经过mask的数据（或者都包含mask信息）。并且解码器最后有一个Linear层完成隐向量向词向量的映射。

<img src="https://my-picture-1311448338.file.myqcloud.com/img/202303261634975.png" alt="image-20230326163411902" style="zoom:50%;" />

```python
class Encoder(nn.Module):

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

        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x_out, x_in, out_mask, in_mask):
        
        x_out = self.embedding(x_out)
        for block in self.blocks:
            x_out = block(x_out, x_in, out_mask, in_mask)

        output = self.linear(x_out)
        
        return output


class Transformer(nn.Module):

    def __init__(self, vocab_sizes, max_length, pad_idxes, d_model, n_heads, d_ff, n_blocks, dropout, device):
        super(Transformer, self).__init__()

        self.pad_idxes = pad_idxes

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
        out_mask = self.mutual_mask(x_out, x_out, self.pad_idxes[1], self.pad_idxes[1]) & self.no_peak_mask(x_out, x_out)

        x_in_out = self.encoder(x_in, in_mask)
        x_out = self.decoder(x_out, x_in_out, out_mask, in_out_mask)
        
        return x_out
    
    def mutual_mask(self, x, y, x_pad_idx, y_pad_idx):
        
        x_seq_len, y_seq_len = x.size(1), y.size(1)
        x_mask = (x != x_pad_idx).unsqueeze(1).unsqueeze(2)
        y_mask = (y != y_pad_idx).unsqueeze(1).unsqueeze(3)
        mask = x_mask.repeat(1, 1, y_seq_len, 1) & y_mask.repeat(1, 1, 1, x_seq_len)
        
        return mask
    
    def no_peak_mask(self, q, k):
        
        len_q = q.size(1)
        len_k = k.size(1)
        mask = torch.triu(torch.ones((len_q, len_k)), diagonal=1).bool().to(q.device)
        
        return mask
```

&emsp;

参考：

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [hyunwoongko/transformer: PyTorch Implementation of "Attention Is All You Need"](https://github.com/hyunwoongko/transformer)
- [tunz/transformer-pytorch: Transformer implementation in PyTorch](https://github.com/tunz/transformer-pytorch)
