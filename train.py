#!/home/bizon/anaconda3/envs/ml/bin/python3


import sys
from huggingface_model import HuggingFaceTranslationModel
from model import TranslationModel, TranslationTrainingConfig
from transformers import (
    AutoModel,
    AutoTokenizer,
    MarianMTModel,
    MarianConfig,
    BartTokenizer,
    BartForConditionalGeneration,
    BartConfig,
    BartTokenizerFast,
    Seq2SeqTrainingArguments,
)
from datasets import Dataset
from data_utils import load_dataset

if __name__ == "__main__":
    # data = load_dataset("wmt15", "fr-en")
    # max_train_size = 2_500
    # print(f"Dataset loaded. Filtering train data to size {max_train_size}...")
    # data['train'] = Dataset.from_dict(data['train'][:max_train_size])

    # print(f"Dataset filtered. Loading model...")
    # hf_model = HuggingFaceTranslationModel.from_pretrained("models/hf/bart_2.5M-fr-en")
    # hf_translator = HuggingFaceTranslationModel(
    #     "hf/bart_2.5M",
    #     "fr",
    #     "en",
    #     tokenizer=BartTokenizer.from_pretrained("facebook/bart-base"),
    #     model = hf_model
    # )
    # smaller_model = TranslationModel.from_pretrained("models/2.5M-fr-en")
    # smaller_translator = TranslationModel(
    #     "2.5M",
    #     "fr",
    #     "en",
    #     src_tokenizer=AutoTokenizer.from_pretrained("dbmdz/bert-base-french-europeana-cased"),
    #     tgt_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased"),
    #     model = smaller_model
    # )

    # larger_model = TranslationModel.from_pretrained("models/5M_4epoch-fr-en")
    # larger_translator = TranslationModel(
    #     "5M_4epoch",
    #     "fr",
    #     "en",
    #     src_tokenizer=AutoTokenizer.from_pretrained("dbmdz/bert-base-french-europeana-cased"),
    #     tgt_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased"),
    #     model = larger_model
    # )

    # translators = [hf_translator, smaller_translator, larger_translator]

    # for translator in translators:
    #     translator.print_stats(data["test"])

    # data = load_dataset("datasets/ht-en", "ht-en")
    # ht_translator = TranslationModel(
    #     "smaller_finetuned_8epoch",
    #     "ht",
    #     "en",
    #     src_tokenizer=AutoTokenizer.from_pretrained("dbmdz/bert-base-french-europeana-cased"),
    #     tgt_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased"),
    #     model = TranslationModel.from_pretrained("models/2.5M-fr-en")
    # )
    # config_kwargs = {
    #     "batch_size": 32,
    #     "distributed": False,
    #     "num_epochs": 8,
    #     "accum_iter": 10,
    #     "base_lr": 1.0,
    #     "max_padding": 128,
    #     "warmup": 3000,
    #     "file_prefix": "model_",
    # }
    # ht_translator.train(data, TranslationTrainingConfig(**config_kwargs))
    commit_hash = sys.argv[1]

    ht_dataset = load_dataset("datasets/ht-en", "ht-en")
    hf_fr_en = HuggingFaceTranslationModel(
        "hf/fr-en-finetuned",
        "ht",
        "en",
        tokenizer=AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en"),
        model=AutoModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en"),
    )
    config = HuggingFaceTranslationModel(
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        commit_hash=commit_hash
    )
    hf_fr_en.train(ht_dataset, config)
