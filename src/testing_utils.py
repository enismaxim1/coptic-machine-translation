
from architecture import EncoderDecoder, Generator
from utils import subsequent_mask

import torch
import torch.nn as nn
from torch import Tensor




def loss(x: float, crit: nn.Module) -> Tensor: 
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator: Generator, criterion: nn.Module):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x: Tensor, y: Tensor, norm: float):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss
    


def greedy_decode(model: EncoderDecoder, src: Tensor, src_mask: Tensor, max_len: int, start_symbol: int):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def check_outputs(
    src_sentence,
    tgt_sentence,
    src_language,
    tgt_language,
    tokenize_src,
    tokenize_tgt,
    model,
    vocab_src,
    vocab_tgt,
    pad_idx=2,
    eos_string="</s>",
):
    from training_utils import Batch
    from training_utils import create_dataloaders

    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        src_language,
        tgt_language,
        tokenize_src,
        tokenize_tgt,
        batch_size=1,
        is_distributed=False,
    )

    vector = next(iter(valid_dataloader))

    print("\nExample %d ========\n")
    print("bahah", src_sentence, tgt_sentence)
    rb = Batch(vector[0], vector[1], pad_idx)
    greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

    src_tokens = [
        vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
    ]
    tgt_tokens = [
        vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
    ]

    print(
        "Source Text (Input)        : "
        + " ".join(src_tokens).replace("\n", "")
    )
    print(
        "Target Text (Ground Truth) : "
        + " ".join(tgt_tokens).replace("\n", "")
    )
    model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
    model_txt = (
        " ".join(
            [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
        ).split(eos_string, 1)[0]
        + eos_string
    )
    print("Model Output               : " + model_txt.replace("\n", ""))
    return (rb, src_tokens, tgt_tokens, model_out, model_txt)


