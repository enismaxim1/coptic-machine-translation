import json
import os
from pathlib import Path
from typing import Optional
from attr import dataclass
import sacrebleu
from datasets import Dataset
from tqdm import tqdm
import torch.nn as nn
from hashlib import sha256


@dataclass
class GenerationConfig:
    """
    Config for generation. Can be used to configure the generation process.
    max_length represents the maximum length of the generated sequence.
    max_new_tokens represents the maximum number of new tokens that can be generated.
    min_length represents the minimum length of the generated sequence.
    min_new_tokens represents the minimum number of new tokens that can be generated.
    early_stopping represents whether generation should stop when the model is confident in its prediction.
    do_sample represents whether sampling should be used when generating.
    num_beams represents the number of beams to use when generating. Default is 1, which means greedy search.
    num_beam_groups represents the number of groups to use when generating. Default is 1, which means no group.
    """

    max_length: int = 20
    max_new_tokens: Optional[int] = None
    min_length: int = 0
    min_new_tokens: Optional[int] = None
    early_stopping: bool = True
    do_sample: bool = False
    num_beams: int = 1
    num_beam_groups: int = 1

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    def hash_fields(self):
        sha = sha256()
        for value in self.__dict__.values():
            sha.update(str(hash(value)).encode("utf-8"))
        return sha.hexdigest()[:8]


class BaseTranslationModel:
    def __init__(
        self,
        model_name: str,
        src_language: str,
        tgt_language: str,
        model: Optional[nn.Module],
        save_to_disk=True,
    ):
        self.model_name = model_name
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.model = model
        self.dir_path = os.path.join(
            "models", f"{model_name}-{src_language}-{tgt_language}/"
        )
        # Prevent overwriting the model if it already exists.
        if os.path.exists(self.dir_path) and save_to_disk:
            raise FileExistsError(
                f"Model {model_name} at path {self.dir_path} already exists. Did you mean to load using from_pretrained?"
            )
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        self.save_to_disk = save_to_disk

    def translate(self, src_sentence: str):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    @classmethod
    def from_pretrained(self):
        raise NotImplementedError()

    def _hash_data_with_config(self, test_dataset, config):
        data_cache_files = test_dataset.cache_files
        for data_cache_file in data_cache_files:
            sha = sha256()
            sha.update(data_cache_file["filename"].encode("utf-8"))
        sha.update(config.hash_fields().encode("utf-8"))
        return sha.hexdigest()[:8]

    def _apply_kwargs(self, config, **kwargs):
        for kwarg in kwargs:
            if hasattr(config, kwarg):
                setattr(config, kwarg, kwargs[kwarg])
            else:
                raise ValueError(f"Invalid kwarg {kwarg}.")

    def translate_test_data(self, test_dataset, config=GenerationConfig(), **kwargs):
        """
        Translates test_dataset using the model and saves the translations to a file.
        Takes a GenerationConfig as input, which can be used to configure the generation process.
        If a kwarg is passed, it will override the corresponding attribute in the config.
        """

        print(f"Computing translations from {self.src_language}-{self.tgt_language}.")

        self._apply_kwargs(config, **kwargs)

        config_hash = self._hash_data_with_config(test_dataset, config)

        data_dir = os.path.join(self.dir_path, "data", config_hash)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        translation_file = os.path.join(data_dir, f"translations.{self.tgt_language}")

        if os.path.exists(translation_file):
            print(f"Translation file already exists. Skipping computation.")
            return

        config.save(os.path.join(data_dir, "generation_config.json"))

        with open(translation_file, "w") as translations:
            for language_pair in tqdm(
                test_dataset["translation"], total=len(test_dataset)
            ):
                test_sentence = language_pair[self.src_language]
                translations.write(self.translate(test_sentence, config) + "\n")

    def compute_bleu(self, test_dataset: Dataset, config=GenerationConfig(), **kwargs):
        # TODO: improve caching so that multiple test sets can be used
        self._apply_kwargs(config, **kwargs)
        data_hash = self._hash_data_with_config(test_dataset, config)
        translation_file = os.path.join(
            self.dir_path, "data", data_hash, f"translations.{self.tgt_language}"
        )
        if not os.path.exists(translation_file):
            print(
                f"Could not find cached dataset at {translation_file}. Computing translations..."
            )
            self.translate_test_data(test_dataset, config)

        translations = Path(translation_file).read_text().strip().split("\n")
        refs = [
            language_pair[self.tgt_language]
            for language_pair in test_dataset["translation"]
        ]
        bleu = sacrebleu.metrics.BLEU()
        return bleu.corpus_score(translations, [refs])

    def compute_chrf(self, test_dataset: Dataset, config=GenerationConfig(), **kwargs):
        self._apply_kwargs(config, **kwargs)
        data_hash = self._hash_data_with_config(test_dataset, config)
        translation_file = os.path.join(
            self.dir_path, "data", data_hash, f"translations.{self.tgt_language}"
        )
        if not os.path.exists(translation_file):
            print(
                f"Could not find cached dataset at {translation_file}. Computing translations..."
            )
            self.translate_test_data(test_dataset, config)

        translations = Path(translation_file).read_text().strip().split("\n")
        refs = [
            language_pair[self.tgt_language]
            for language_pair in test_dataset["translation"]
        ]
        return sacrebleu.corpus_chrf(translations, [refs])

    def print_stats(self, test_dataset: Dataset):
        print(
            f"Translation model {self.model_name} trained on {self.src_language}-{self.tgt_language}:"
        )
        print(self.compute_bleu(test_dataset))
        print(f"{self.compute_chrf(test_dataset)}\n")
