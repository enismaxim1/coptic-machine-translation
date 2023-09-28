import os
from pathlib import Path
import sacrebleu
from tqdm import tqdm
import transformers
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, MarianMTModel, MarianConfig, GenerationConfig




class HuggingFaceTranslationModel:
    def __init__(
        self,
        model_name: str,
        src_language: str,
        tgt_language: str,
        dataset,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
        max_input_length: int = 128,
        max_target_length: int = 128,
    ):
        self.model_name = model_name
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.dir_path = os.path.join("models", "hf", f"{model_name}-{self.src_language}-{self.tgt_language}")
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model = model
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self._load_model()


    def translate(self, src_sentence: str):
        inputs = self.tokenizer.encode(src_sentence, return_tensors="pt")
        outputs = self.model.generate(inputs)
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text
    
    def translate_test_data(self):
        print(f"Computing translations from {self.src_language}-{self.tgt_language}.")

        data_dir = os.path.join(self.dir_path, "data/")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        translation_file = f"{data_dir}translations.{self.tgt_language}"

        if os.path.exists(translation_file):
            print(f"Translation file already exists. Skipping computation.")
            return
        
        with open(translation_file, 'w') as translations:
            for language_pair in tqdm(self.dataset['test']['translation'], total=len(self.dataset['test'])):
                test_sentence = language_pair[self.src_language]
                translations.write(self.translate(test_sentence) + "\n")

    def compute_bleu(self):
        translation_file = os.path.join(self.dir_path, f"data/translations.{self.tgt_language}")
        if not os.path.exists(translation_file):
            raise FileNotFoundError(f"Could not find translations file at path {translation_file}.")
        
        translations = Path(translation_file).read_text().strip().split("\n")
        refs = [language_pair[self.tgt_language] for language_pair in self.dataset['test']['translation']]
        bleu = sacrebleu.metrics.BLEU()
        return bleu.corpus_score(translations, [refs])
    
    
    def compute_chrf(self):
        translation_file = os.path.join(self.dir_path, f"data/translations.{self.tgt_language}")
        if not os.path.exists(translation_file):
            raise FileNotFoundError(f"Could not find translations file at path {translation_file}.")
        
        translations = Path(translation_file).read_text().strip().split("\n")
        refs = [language_pair[self.tgt_language] for language_pair in self.dataset['test']['translation']]
        return sacrebleu.corpus_chrf(translations, [refs])
    
    def print_stats(self):
        print(f"Translation model {self.model_name} trained on {self.src_language}-{self.tgt_language}:")
        print(f"BLEU: {self.compute_bleu()}")
        print(f"CHRF: {self.compute_chrf()}\n")

    def preprocess_function(self, data):
        inputs = [ex[self.src_language] for ex in data["translation"]]
        targets = [ex[self.tgt_language] for ex in data["translation"]]
        model_inputs = self.tokenizer(
            inputs, max_length=self.max_input_length, truncation=True
        )

        # Setup the tokenizer for targets
        labels = self.tokenizer(
            text_target=targets, max_length=self.max_target_length, truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels
    
    def _load_model(self):
        if os.path.exists(self.dir_path):
            print(f"Found cached model parameters at {self.dir_path}. Loading parameters...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.dir_path)
        else:
            print(f"No cached model parameters found. Training model...")
            self._train()

    def _train(self):
        tokenized_datasets = self.dataset.map(self.preprocess_function, batched=True)
        batch_size = 16

        args = Seq2SeqTrainingArguments(
            self.dir_path,
            evaluation_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=1,
            predict_with_generate=True,
        )
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset = tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        trainer.train()

        trainer.save_model(self.dir_path)
