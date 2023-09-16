import os
import torch
import torch.nn as nn
import copy
import math
from tqdm import tqdm

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

    train_filename = f"{languages}_train.{language}"
    valid_filename = f"{languages}_valid.{language}"
    test_filename = f"{languages}_test.{language}"



    for filename in [train_filename, valid_filename, test_filename]:
        if os.path.exists(filename):
            raise FileExistsError(f"{filename} already exists.")

    with open(train_filename, 'w') as train:
        train.writelines(lines[:num_train])

    with open(valid_filename, 'w') as train:
        train.writelines(lines[num_train:num_train+num_valid])

    with open(test_filename, 'w') as train:
        train.writelines(lines[num_train+num_valid:])



def huggingface_to_parallel(dataset, src_language, tgt_language, len_dataset):
    src_filename = f"{src_language}-{tgt_language}.{src_language}"
    tgt_filename = f"{src_language}-{tgt_language}.{tgt_language}"
    
    if os.path.exists(src_filename):
        raise FileExistsError(f"{src_filename} already exists.")
    
    if os.path.exists(tgt_filename):
        raise FileExistsError(f"{tgt_filename} already exists.")

    with open(src_filename, 'w', encoding='utf-8') as src_file, open(tgt_filename, 'w', encoding='utf-8') as tgt_file:
        for example in tqdm(dataset, total = len_dataset):
            translation = example['translation']
            src_file.write(translation[src_language] + '\n')
            tgt_file.write(translation[tgt_language] + '\n')
    
    make_splits(src_filename)
    make_splits(tgt_filename)