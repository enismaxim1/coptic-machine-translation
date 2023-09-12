import torch
import torch.nn as nn
import copy
import math

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def make_splits(path: str, train_split: float = .8, valid_split: float = .1, test_split: float = .1):
    # TODO: allow streaming for big datasets
    # file path should be of form src-tgt.src or src-tgt.tgt
    languages, language = path.split(".")
    with open(path, 'r') as file:
        lines = file.readlines()
    
    num_train, num_valid, num_test = int(train_split * len(lines)), int(valid_split * len(lines)), int(test_split * len(lines))

    with open(f"{languages}_train.{language}", 'w') as train:
        train.writelines(lines[:num_train])

    with open(f"{languages}_valid.{language}", 'w') as train:
        train.writelines(lines[num_train:num_train+num_valid])

    with open(f"{languages}_test.{language}", 'w') as train:
        train.writelines(lines[num_train+num_valid:])