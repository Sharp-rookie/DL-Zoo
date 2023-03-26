import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim

from data import *
from models.transformer import Transformer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])  # shift right (ignore <eos> token), only used in this nlp tasks
        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        print('\rstep :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item(), end='')
    
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg[:, :-1])
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
            print('\rstep :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item(), end='')
    
    return epoch_loss / len(iterator)


def run(model, optimizer, scheduler, criterion, max_epoch, best_loss):
    train_losses, valid_losses = [], []
    for epoch in range(max_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if epoch > warmup:
            scheduler.step(valid_loss)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f'checkpoints/model-{epoch}.pt')
        
        print(f'\n Epoch[{epoch+1}/{max_epoch}] | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    return best_loss


if __name__ == '__main__':

    os.makedirs('checkpoints', exist_ok=True)

    # Model
    vocab_sizes = [in_vocab_size, out_vocab_size]
    pad_idxes = [in_pad_idx, out_pad_idx]
    model = Transformer(vocab_sizes, max_length, pad_idxes, d_model, n_heads, d_ff, n_blocks, dropout, device).to(device)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(params=model.parameters(), lr=init_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True, factor=factor, patience=patience)
    criterion = nn.CrossEntropyLoss(ignore_index=in_pad_idx)

    # Train
    best_loss = run(model, optimizer, scheduler, criterion, max_epoch, best_loss=float('inf'))
