
import itertools
from typing import Tuple
from architecture import *

import copy
import os
from iterators import collate_batch_huggingface, from_streaming_dataset, get_language_iters
from testing_utils import SimpleLossCompute, greedy_decode
from tokenizer_utils import yield_tokens
from training_utils import Batch, DummyOptimizer, DummyScheduler, LabelSmoothing, TrainState, rate, run_epoch
from transformers import M2M100Tokenizer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp
import torchtext.datasets as datasets
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
from datasets import IterableDatasetDict
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast, BertTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


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
            src_tokenizer_override: Optional[PreTrainedTokenizerBase] = None,
            tgt_tokenizer_override: Optional[PreTrainedTokenizerBase] = None,
            cloud_data_iter: Optional[IterableDatasetDict] = None,
            N: int = 6,
            d_model: int = 512,
            d_ff: int = 2048,
            heads: int = 8,
            dropout: float = 0.1,
            **kwargs
            ):
        
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.dir_path = f"models/{src_language}-{tgt_language}/"

        self.src_tokenizer = self._load_tokenizer(src_language) if not src_tokenizer_override else src_tokenizer_override
        self.tgt_tokenizer = self._load_tokenizer(tgt_language) if not tgt_tokenizer_override else tgt_tokenizer_override
        self.src_vocab = self.src_tokenizer.get_vocab()
        self.tgt_vocab = self.tgt_tokenizer.get_vocab()
        print(f"Initialized {src_language} vocab with {len(self.src_vocab)} tokens.")
        print(f"Initialized {tgt_language} vocab with {len(self.tgt_vocab)} tokens.")

        if cloud_data_iter:
            copy1, copy2 = cloud_data_iter.copy(), cloud_data_iter.copy()
            self.src_train = from_streaming_dataset(copy1["train"], src_language)
            self.src_valid = from_streaming_dataset(copy1["validation"], src_language)
            self.src_test = from_streaming_dataset(copy1["test"], src_language)
            self.tgt_train = from_streaming_dataset(copy2["train"], tgt_language)
            self.tgt_valid = from_streaming_dataset(copy2["validation"], tgt_language)
            self.tgt_test = from_streaming_dataset(copy2["test"], tgt_language)
        else:
            train_path = f"{self.dir_path}data/train"
            valid_path = f"{self.dir_path}data/valid"
            test_path = f"{self.dir_path}data/test"
            self.src_train, self.tgt_train = get_language_iters(train_path, self.src_language, self.tgt_language)
            self.src_valid, self.tgt_valid = get_language_iters(valid_path, self.src_language, self.tgt_language)
            self.src_test, self.tgt_test = get_language_iters(test_path, self.src_language, self.tgt_language)

        self.d_model = d_model

        self.architecture = self._make_architecture(N, d_model, d_ff, heads, dropout)
        self._load_params(self.architecture)
        

    def translate(self, src_sentence: str):
        raise NotImplementedError()
        self.architecture.eval()
        encoded = self.src_tokenizer(src_sentence, return_tensors = "pt")
        src = encoded["input_ids"]
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
    

    def _load_tokenizer(self, language: str):
        tokenizer_path = f"{self.dir_path}{language}_tokenizer.json"

        if not os.path.exists(tokenizer_path):
            self._train_tokenizer(language)

        tokenizer = PreTrainedTokenizerFast(tokenizer_file = tokenizer_path)
        tokenizer.pad_token = "[PAD]"
        return tokenizer
    

    def _train_tokenizer(self, language: str, vocab_size = 10000):
        
        print(f"Training new BPE tokenizer for language {language}.")

        data_path = f"{self.dir_path}data/train.{language}"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} does not exist; could not train tokenizer on language {language}.")

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size = vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

        # Train the tokenizer
        tokenizer.train(files=[data_path], trainer=trainer)
        tokenizer.save(f"{self.dir_path}{language}_tokenizer.json")



    def _create_dataloaders(
        self,
        device,
        batch_size=12000,
        max_padding=128,
        is_distributed=True,
    ):

        def collate_fn(batch):
            return collate_batch_huggingface(batch, self.src_tokenizer, self.tgt_tokenizer, device, max_padding)

        train_iter = zip(self.src_train, self.tgt_train)
        valid_iter = zip(self.src_valid, self.tgt_valid)

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
        gpu: int,
        ngpus_per_node,
        config,
        is_distributed=False,
    ):
        print(f"Train worker process using GPU: {gpu} for training", flush=True)
        torch.cuda.set_device(gpu)
        pad_idx = self.tgt_tokenizer.pad_token_id
        model = self.architecture
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
        # TODO: fix this; not even sure how it works in its present state
        mp.spawn(
            self._train_worker,
            nprocs=ngpus,
            args=(self, ngpus, config, True),
        )





