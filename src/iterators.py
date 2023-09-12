
import torch
from torch.nn.functional import pad
import os
from tokenizer_utils import tokenize

def file_iter(path):
    with open(path, 'r') as file:
        yield from file
        
def get_language_iters(path: str, src_language: str, tgt_language: str):
    # TODO: support compressed formats
    src_path =  f"{path}.{src_language}"
    tgt_path = f"{path}.{tgt_language}"
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Path {src_path} not found. Please add source data file to this location.")
    if not os.path.exists(tgt_path):
        raise FileNotFoundError(f"Path {tgt_path} not found. Please add target data file to this location.")
    return file_iter(src_path), file_iter(tgt_path)

def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


