import torch

# GPU device setting
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# model parameter setting
batch_size = 128
max_length = 256
d_model = 512
n_blocks = 6
n_heads = 8
d_ff = 2048
dropout = 0.1

# optimizer parameter setting
init_lr = 1e-5
factor = 0.9
patience = 10
warmup = 100
max_epoch = 1000
clip = 1.0
weight_decay = 5e-4