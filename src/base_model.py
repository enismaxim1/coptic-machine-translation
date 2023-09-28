import os
from pathlib import Path
from typing import Optional
import sacrebleu
from datasets import Dataset
from tqdm import tqdm
import torch.nn as nn

class BaseTranslationModel:

    def __init__(
        self,
        model_name: str,
        src_language: str,
        tgt_language: str,
        model: Optional[nn.Module],
        save_to_disk = True,
    ):
        self.model_name = model_name
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.model = model
        self.dir_path = os.path.join("models", f"{model_name}-{src_language}-{tgt_language}/")
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
    
    

    def translate_test_data(self, test_dataset):
        # TODO: improve caching so that multiple test sets can be cached
        print(f"Computing translations from {self.src_language}-{self.tgt_language}.")

        data_dir = os.path.join(self.dir_path, "data/")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        
        translation_file = os.path.join(data_dir, f"translations.{self.tgt_language}")

        if os.path.exists(translation_file):
            print(f"Translation file already exists. Skipping computation.")
            return
        
        with open(translation_file, 'w') as translations:
            for language_pair in tqdm(test_dataset['translation'], total=len(test_dataset)):
                test_sentence = language_pair[self.src_language]
                translations.write(self.translate(test_sentence) + "\n")

            
    def compute_bleu(self, test_dataset: Dataset):
        # TODO: improve caching so that multiple test sets can be used
        translation_file = os.path.join(self.dir_path, "data", f"translations.{self.tgt_language}")
        if not os.path.exists(translation_file):
            print(f"Could not find cached dataset at {translation_file}. Computing translations...")
            self.translate_test_data(test_dataset)
        
        translations = Path(translation_file).read_text().strip().split("\n")
        refs = [language_pair[self.tgt_language] for language_pair in test_dataset['translation']]
        bleu = sacrebleu.metrics.BLEU()
        return bleu.corpus_score(translations, [refs])

        
    def compute_chrf(self, test_dataset: Dataset):
        translation_file = os.path.join(self.dir_path, "data", f"translations.{self.tgt_language}")
        if not os.path.exists(translation_file):
            print(f"Could not find cached dataset at {translation_file}. Computing translations...")
            self.translate_test_data(test_dataset)
        
        translations = Path(translation_file).read_text().strip().split("\n")
        refs = [language_pair[self.tgt_language] for language_pair in test_dataset['translation']]
        return sacrebleu.corpus_chrf(translations, [refs])
    
    def print_stats(self, test_dataset: Dataset):
        print(f"Translation model {self.model_name} trained on {self.src_language}-{self.tgt_language}:")
        print(self.compute_bleu(test_dataset))
        print(f"{self.compute_chrf(test_dataset)}\n")