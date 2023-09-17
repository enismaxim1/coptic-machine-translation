
import torch
from torch.nn.functional import pad
import os
from tokenizer_utils import tokenize
from transformers import PreTrainedTokenizerFast

def file_iter(path):
    with open(path, 'r') as file:
        yield from file
        
def get_language_iter(path: str, src_language: str, tgt_language: str):
    # TODO: support compressed formats
    src_path =  f"{path}.{src_language}"
    tgt_path = f"{path}.{tgt_language}"
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Path {src_path} not found. Please add source data file to this location.")
    if not os.path.exists(tgt_path):
        raise FileNotFoundError(f"Path {tgt_path} not found. Please add target data file to this location.")
    return zip(file_iter(src_path), file_iter(tgt_path))


def from_streaming_dataset(dataset, language: str):
    for data in dataset:
        # NOTE: cannot yield both translations at the same time, since we need two distinct iterators
        yield data["translation"][language]



def collate_batch_huggingface(
    batch, 
    src_tokenizer: PreTrainedTokenizerFast, 
    tgt_tokenizer: PreTrainedTokenizerFast, 
    device, 
    max_padding=128
):
    
    def wrap_special(sentence):
        return src_tokenizer.bos_token + sentence + src_tokenizer.eos_token
    
    src_texts = [wrap_special(item[0]) for item in batch]
    tgt_texts = [wrap_special(item[1]) for item in batch]

    # Tokenize and pad source sequences
    src_encodings = src_tokenizer(src_texts, return_tensors='pt', add_special_tokens=True, padding='max_length', truncation=True, max_length=max_padding)
    src_ids = src_encodings['input_ids'].to(device)

    # Tokenize and pad target sequences
    tgt_encodings = tgt_tokenizer(tgt_texts, return_tensors='pt', add_special_tokens=True, padding='max_length', truncation=True, max_length=max_padding)
    tgt_ids = tgt_encodings['input_ids'].to(device)

    return src_ids, tgt_ids




