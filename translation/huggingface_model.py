import os
import transformers
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, MarianMTModel, MarianConfig, GenerationConfig
from datasets import Dataset
from base_model import BaseTranslationModel



class HuggingFaceTranslationModel(BaseTranslationModel):
    def __init__(
        self,
        model_name: str,
        src_language: str,
        tgt_language: str,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
        save_to_disk = True,
        max_input_length: int = 128,
        max_target_length: int = 128,
    ):
        super().__init__(model_name, src_language, tgt_language, model, save_to_disk)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length


    def translate(self, src_sentence: str):
        inputs = self.tokenizer.encode(src_sentence, return_tensors="pt")
        outputs = self.model.generate(inputs)
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text


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

    def train(self, dataset: Dataset):
        tokenized_datasets = dataset.map(self.preprocess_function, batched=True)
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
        if self.save_to_disk:
            trainer.save_model(self.dir_path)


    @classmethod
    def from_pretrained(cls, path):
        return AutoModelForSeq2SeqLM.from_pretrained(path)