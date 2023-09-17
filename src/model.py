
import itertools
from typing import Tuple
from architecture import *

import copy
import os
from iterators import collate_batch_huggingface, from_streaming_dataset, get_language_iter
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
from pathlib import Path
import sacrebleu
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
            model_max_len: int =  5000,
            **kwargs
            ):
        
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.dir_path = f"models/{src_language}-{tgt_language}/"

        self.d_model = d_model
        self.model_max_len = model_max_len
        self.cloud_data_iter = cloud_data_iter

        self.src_tokenizer = self._load_tokenizer(src_language) if not src_tokenizer_override else src_tokenizer_override
        self.tgt_tokenizer = self._load_tokenizer(tgt_language) if not tgt_tokenizer_override else tgt_tokenizer_override
        self.src_vocab = self.src_tokenizer.get_vocab()
        self.tgt_vocab = self.tgt_tokenizer.get_vocab()
        print(f"Initialized {src_language} vocab with {len(self.src_vocab)} tokens.")
        print(f"Initialized {tgt_language} vocab with {len(self.tgt_vocab)} tokens.")

        
        

        self.architecture = self._make_architecture(N, d_model, d_ff, heads, dropout, model_max_len)
        self._load_params()
        

    def translate(self, src_sentence: str):
        
        if not src_sentence or src_sentence.isspace():
            return src_sentence
        
        self.architecture.eval()
        encoded = self.src_tokenizer(src_sentence, return_tensors = "pt")
        src = encoded["input_ids"]
        src_mask = (src != self.src_tokenizer.pad_token_id).unsqueeze(-2)
        out = greedy_decode(
            self.architecture, 
            src, 
            src_mask, 
            max_len=60, 
            start_symbol=self.tgt_tokenizer.bos_token_id, 
            end_symbol=self.tgt_tokenizer.eos_token_id
        )
        return self.tgt_tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    
    def translate_test_data(self):
        print(f"Computing translations from {self.src_language}-{self.tgt_language}.")

        data_dir = f"{self.dir_path}data/"
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        
        translation_file = f"{data_dir}translations.{self.tgt_language}"

        if os.path.exists(translation_file):
            print(f"Translation file already exists. Skipping computation.")
            return
        
        with open(translation_file, 'w') as translations:
            for test_sentence, _ in self.get_test_iters():
                translations.write(self.translate(test_sentence) + "\n")

            
    def compute_bleu(self):
        translation_file = f"{self.dir_path}data/translations.{self.tgt_language}"
        if not os.path.exists(translation_file):
            raise FileNotFoundError(f"Could not find translations file at path {translation_file}.")
        
        translations = Path(translation_file).read_text().split("\n")
        refs = [[ref] for _, ref in self.get_test_iters()]
        return sacrebleu.corpus_bleu(translations, refs)
    
    def compute_chrf(self):
        translation_file = f"{self.dir_path}data/translations.{self.tgt_language}"
        if not os.path.exists(translation_file):
            raise FileNotFoundError(f"Could not find translations file at path {translation_file}.")
        
        translations = Path(translation_file).read_text().split("\n")
        refs = [[ref] for _, ref in self.get_test_iters()]
        return sacrebleu.corpus_chrf(translations, refs)
    
    def _make_architecture(
        self, N: int, d_model: int, d_ff: int, heads: int, dropout: float, model_max_len: int
    ):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(heads, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout, model_max_len)
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
    

    def _train(self, config):
        print(f"Training translation model on {self.src_language}-{self.tgt_language}.")
        if config["distributed"]:
            self._train_distributed_model(
                config
            )
        else:
            self._train_worker(
                0, 1, config, False
        )

    def _load_params(
            self
        ):
        # TODO: cache models only for given hyperparameters
        # TODO: verify that distributed behavior works as intended
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
            self._train(config)
        else:
            print(f"Using cached model parameters from path {model_path}.")

        self.architecture.load_state_dict(torch.load(f"{self.dir_path}model_final.pt"))
    

    def _load_tokenizer(self, language: str):
        tokenizer_path = f"{self.dir_path}{language}_tokenizer.json"

        if not os.path.exists(tokenizer_path):
            self._train_tokenizer(language)

        tokenizer = PreTrainedTokenizerFast(tokenizer_file = tokenizer_path)
        tokenizer.pad_token = "[PAD]"
        # TODO: hacky fix since CLS and SEP are not actually bos and eos tokens; find a better fix for self.translate
        tokenizer.bos_token = "[BOS]"
        tokenizer.eos_token = "[EOS]"
        tokenizer.model_max_length = self.model_max_len
        return tokenizer
    

    def _train_tokenizer(self, language: str, vocab_size = 10000):
        
        print(f"Training new BPE tokenizer for language {language}.")

        data_path = f"{self.dir_path}data/train.{language}"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} does not exist; could not train tokenizer on language {language}.")

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size = vocab_size, special_tokens=["[UNK]", "[EOS]", "[BOS]", "[PAD]", "[MASK]"])

        # Train the tokenizer
        tokenizer.train(files=[data_path], trainer=trainer)
        tokenizer.save(f"{self.dir_path}{language}_tokenizer.json")


    def get_train_iters(self):
        if self.cloud_data_iter:
            iter_copy = self.cloud_data_iter.copy()
            return from_streaming_dataset(iter_copy["train"], self.src_language, self.tgt_language)
        else:
            train_path = f"{self.dir_path}data/train"
            return get_language_iter(train_path, self.src_language, self.tgt_language)

    def get_valid_iters(self):
        if self.cloud_data_iter:
            iter_copy = self.cloud_data_iter.copy()
            return from_streaming_dataset(iter_copy["validation"], self.src_language, self.tgt_language)
        else:
            train_path = f"{self.dir_path}data/valid"
            return get_language_iter(train_path, self.src_language, self.tgt_language)


    def get_test_iters(self):
        if self.cloud_data_iter:
            iter_copy = self.cloud_data_iter.copy()
            return from_streaming_dataset(iter_copy["test"], self.src_language, self.tgt_language)
        else:
            train_path = f"{self.dir_path}data/test"
            return get_language_iter(train_path, self.src_language, self.tgt_language)

    def _create_dataloaders(
        self,
        device,
        batch_size=12000,
        max_padding=128,
        is_distributed=True,
    ):

        def collate_fn(batch):
            return collate_batch_huggingface(batch, self.src_tokenizer, self.tgt_tokenizer, device, max_padding)

        train_iter = self.get_train_iters()
        valid_iter = self.get_valid_iters()

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
            args=(ngpus, config, True),
        )





