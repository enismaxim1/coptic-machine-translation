
import itertools
from typing import Tuple
from architecture import *

import copy
import os
from iterators import collate_batch, get_language_iters
from testing_utils import SimpleLossCompute, greedy_decode
from tokenizer_utils import yield_tokens
from training_utils import Batch, DummyOptimizer, DummyScheduler, LabelSmoothing, TrainState, rate, run_epoch
from transformers import M2M100Tokenizer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp
import torchtext.datasets as datasets
from torchtext.vocab import build_vocab_from_iterator, Vocab
from iterators import collate_batch
import torchtext.datasets as datasets
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
import GPUtil

class TranslationModel:
    """
    A neural machine translator between a source and target language. Upon initialization, the translator
    will either fetched cached model parameters between the two languages or it will translate using parallel
    train, validation, and test data located within the models/{src}-{target}/data directory.

    To initialize a new translation model on a language pair src-target, ensure that the models/{src}-{target}/data 
    directory has files train.src, train.tgt, valid.src, valid.tgt, test.src, and test.tgt.

    PUBLIC API:
    translate(src_sentence): Translates a sentence from the source language to the target language.
    """

    def __init__(
            self, 
            src_language: str, 
            tgt_language: str,
            src_tokenizer_override = None,
            tgt_tokenizer_override = None,
            N: int = 6,
            d_model: int = 512,
            d_ff: int = 2048,
            heads: int = 8,
            dropout: float = 0.1,
            **kwargs
            ):
        
        self.src_language = src_language
        self.tgt_language = tgt_language
        default_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")
        self.src_tokenizer = default_tokenizer if not src_tokenizer_override else src_tokenizer_override
        self.tgt_tokenizer = default_tokenizer if not tgt_tokenizer_override else tgt_tokenizer_override

        self.dir_path = f"models/{src_language}-{tgt_language}/"
        self.d_model = d_model

        self.src_vocab, self.tgt_vocab = self._load_vocab()
        self.architecture = self._make_architecture(N, d_model, d_ff, heads, dropout)
        self._load_params(self.architecture)
        

    def translate(self, src_sentence: str):
        self.architecture.eval()
        src_tokens = self.src_tokenizer(src_sentence)
        src = torch.LongTensor([[self.src_vocab[w] for w in src_tokens]])
        src = Variable(src)
        src_mask = (src != self.src_vocab["<blank>"]).unsqueeze(-2)
        out = greedy_decode(self.architecture, src, src_mask, 
                            max_len=60, start_symbol=self.tgt_vocab["<s>"])
        print("Translation:", end="\t")
        trans = "<s> "
        for i in range(1, out.size(1)):
            sym = self.tgt_vocab.get_itos()[out[0, i]]
            if sym == "</s>": break
            trans += sym + " "
        return trans

    def _make_architecture(
        self, N: int, d_model: int, d_ff: int, heads: int, dropout: float
    ):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(heads, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        architecture = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, len(self.src_vocab)), c(position)),
            nn.Sequential(Embeddings(d_model, len(self.tgt_vocab)), c(position)),
            Generator(d_model, len(self.tgt_vocab)),
        )

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in architecture.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return architecture
    

    def _load_params(
            self, architecture: EncoderDecoder
        ) -> EncoderDecoder:
        # TODO: cache models only for given hyperparameters
        config = {
            "batch_size": 32,
            "distributed": False,
            "num_epochs": 8,
            "accum_iter": 10,
            "base_lr": 1.0,
            "max_padding": 72,
            "warmup": 3000,
            "file_prefix": "model_",
        }

        model_path = f"{self.dir_path}model_final.pt"
        if not os.path.exists(model_path):
            self._train(architecture, config)
        else:
            print(f"Using cached model parameters from path {model_path}.")

        architecture.load_state_dict(torch.load(f"{self.dir_path}model_final.pt"))
    
    def _load_vocab(self):
        filename = f"{self.dir_path}vocab.pt"
        if not os.path.exists(filename):
            vocab_src, vocab_tgt = self._build_vocabulary()
            torch.save((vocab_src, vocab_tgt), filename)
        else:
            vocab_src, vocab_tgt = torch.load(filename)
            print(f"Using cached vocabulary from path {filename}.")
        return vocab_src, vocab_tgt
    

    def _train(self, architecture: EncoderDecoder, config):
        print(f"Training translation model on {self.src_language}-{self.tgt_language}.")
        if config["distributed"]:
            self._train_distributed_model(
                architecture,
                config
            )
        else:
            self._train_worker(
                architecture, 0, 1, config, False
        )
    


    def _build_vocabulary(self) -> Tuple[Vocab, Vocab]:
        """Builds a vocabulary (torch.vocab.Vocab) for a given translation model."""

        print(f"Building vocab for {self.src_language}-{self.tgt_language}.")

        train_path = f"{self.dir_path}data/train"
        valid_path = f"{self.dir_path}data/valid"
        test_path = f"{self.dir_path}data/test"

        src_train, tgt_train = get_language_iters(train_path, self.src_language, self.tgt_language)
        src_valid, tgt_valid = get_language_iters(valid_path, self.src_language, self.tgt_language)
        src_test, tgt_test = get_language_iters(test_path, self.src_language, self.tgt_language)


        src_vocab = build_vocab_from_iterator(
            yield_tokens(itertools.chain(src_train, src_valid, src_test), self.src_tokenizer),
            min_freq=2,
            specials=["<s>", "</s>", "<blank>", "<unk>"],
        )

        src_vocab.set_default_index(src_vocab["<unk>"])

        tgt_vocab = build_vocab_from_iterator(
            yield_tokens(itertools.chain(tgt_train, tgt_valid, tgt_test), self.tgt_tokenizer),
            min_freq=2,
            specials=["<s>", "</s>", "<blank>", "<unk>"],
        )

        tgt_vocab.set_default_index(tgt_vocab["<unk>"])

        return src_vocab, tgt_vocab



    def _create_dataloaders(
        self,
        device,
        batch_size=12000,
        max_padding=128,
        is_distributed=True,
    ):
        

        def collate_fn(batch):
            return collate_batch(
                batch,
                self.src_tokenizer,
                self.tgt_tokenizer,
                self.src_vocab,
                self.tgt_vocab,
                device,
                max_padding=max_padding,
                pad_id=self.tgt_vocab.get_stoi()["<blank>"],
            )
        
        train_path = f"{self.dir_path}data/train"
        valid_path = f"{self.dir_path}data/valid"
        train_iter = zip(*get_language_iters(train_path, self.src_language, self.tgt_language))
        valid_iter = zip(*get_language_iters(valid_path, self.src_language, self.tgt_language))

        train_iter_map = to_map_style_dataset(
            train_iter
        )  # DistributedSampler needs a dataset len()
        train_sampler = (
            DistributedSampler(train_iter_map) if is_distributed else None
        )
        valid_iter_map = to_map_style_dataset(valid_iter)
        valid_sampler = (
            DistributedSampler(valid_iter_map) if is_distributed else None
        )

        train_dataloader = DataLoader(
            train_iter_map,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collate_fn,
        )
        valid_dataloader = DataLoader(
            valid_iter_map,
            batch_size=batch_size,
            shuffle=(valid_sampler is None),
            sampler=valid_sampler,
            collate_fn=collate_fn,
        )
        return train_dataloader, valid_dataloader


    def _train_worker(
        self,
        architecture: EncoderDecoder,
        gpu: int,
        ngpus_per_node,
        config,
        is_distributed=False,
    ):
        print(f"Train worker process using GPU: {gpu} for training", flush=True)
        torch.cuda.set_device(gpu)

        pad_idx = self.tgt_vocab["<blank>"]
        model = architecture
        model.cuda(gpu)
        module = model
        is_main_process = True
        if is_distributed:
            dist.init_process_group(
                "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
            )
            model = DDP(model, device_ids=[gpu])
            module = model.module
            is_main_process = gpu == 0

        criterion = LabelSmoothing(
            size=len(self.tgt_vocab), padding_idx=pad_idx, smoothing=0.1
        )
        criterion.cuda(gpu)

        train_dataloader, valid_dataloader = self._create_dataloaders(
            gpu,
            batch_size=config["batch_size"] // ngpus_per_node,
            max_padding=config["max_padding"],
            is_distributed=is_distributed,
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(
                step, self.d_model, factor=1, warmup=config["warmup"]
            ),
        )
        train_state = TrainState()

        for epoch in range(config["num_epochs"]):
            if is_distributed:
                train_dataloader.sampler.set_epoch(epoch)
                valid_dataloader.sampler.set_epoch(epoch)

            model.train()
            print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
            _, train_state = run_epoch(
                (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
                model,
                SimpleLossCompute(module.generator, criterion),
                optimizer,
                lr_scheduler,
                mode="train+log",
                accum_iter=config["accum_iter"],
                train_state=train_state,
            )

            GPUtil.showUtilization()
            if is_main_process:
                checkpoint_dir = f"{self.dir_path}checkpoints/"
                if not os.path.exists(checkpoint_dir):
                    os.mkdir(checkpoint_dir)
                file_path = f"{checkpoint_dir}%s%.2d.pt" % (config["file_prefix"], epoch)
                torch.save(module.state_dict(), file_path)
            torch.cuda.empty_cache()

            print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
            model.eval()
            sloss = run_epoch(
                (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
                model,
                SimpleLossCompute(module.generator, criterion),
                DummyOptimizer(),
                DummyScheduler(),
                mode="eval",
            )
            print(sloss)
            torch.cuda.empty_cache()

        if is_main_process:
            file_path = "%s%sfinal.pt" % (self.dir_path, config["file_prefix"])
            torch.save(module.state_dict(), file_path)




    def _train_distributed_model(self, config):

        ngpus = torch.cuda.device_count()
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12356"
        print(f"Number of GPUs detected: {ngpus}")
        print("Spawning training processes ...")
        mp.spawn(
            self._train_worker,
            nprocs=ngpus,
            args=(self, ngpus, config, True),
        )





