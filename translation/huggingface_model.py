import json
import os
from attr import dataclass
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    MarianMTModel,
    MarianConfig,
    AutoTokenizer,
)
from datasets import Dataset
from base_model import BaseTranslationModel, GenerationConfig
from utils import get_git_revision_short_hash


@dataclass
class HuggingFaceTranslationModelTrainingConfig:
    job_hash: str
    evaluation_strategy: str = "epoch"
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    weight_decay: float = 0.01
    save_total_limit: int = 3
    num_train_epochs: int = 1
    predict_with_generate: bool = True
    commit_hash: str = get_git_revision_short_hash()

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)


class HuggingFaceTranslationModel(BaseTranslationModel):
    def __init__(
        self,
        model_name: str,
        src_language: str,
        tgt_language: str,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
        save_to_disk=True,
        max_input_length: int = 128,
        max_target_length: int = 128,
    ):
        super().__init__(model_name, src_language, tgt_language, model, save_to_disk)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        if save_to_disk:
            self.save_args()
            tokenizer.save_pretrained(self.dir_path)
            model.save_pretrained(self.dir_path)

    def translate(self, src_sentence: str, generation_config: GenerationConfig()):
        inputs = self.tokenizer.encode(src_sentence, return_tensors="pt")
        outputs = self.model.generate(
            inputs,
            max_length=generation_config.max_length,
            max_new_tokens=generation_config.max_new_tokens,
            min_length=generation_config.min_length,
            min_new_tokens=generation_config.min_new_tokens,
            early_stopping=generation_config.early_stopping,
            do_sample=generation_config.do_sample,
            num_beams=generation_config.num_beams,
            num_beam_groups=generation_config.num_beam_groups,
        )
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text
    

    def preprocess_function(self, data):
        inputs = [ex[self.src_language] for ex in data["translation"]]
        targets = [ex[self.tgt_language] for ex in data["translation"]]
        model_inputs = self.tokenizer(
            inputs, max_length=self.max_input_length, truncation=True, padding='max_length'
        )

        # Setup the tokenizer for targets
        labels = self.tokenizer(
            text_target=targets, max_length=self.max_target_length, truncation=True, padding='max_length'
        )

        model_inputs["labels"] = labels["input_ids"]

        # Add decoder input ids 
        model_inputs["decoder_input_ids"] = labels["input_ids"]

        return model_inputs


    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels


    def train(
        self, dataset: Dataset, train_config: HuggingFaceTranslationModelTrainingConfig
    ):
        tokenized_datasets = dataset.map(self.preprocess_function, batched=True)
        args = Seq2SeqTrainingArguments(
            self.dir_path,
            evaluation_strategy=train_config.evaluation_strategy,
            learning_rate=train_config.learning_rate,
            per_device_train_batch_size=train_config.per_device_train_batch_size,
            per_device_eval_batch_size=train_config.per_device_eval_batch_size,
            weight_decay=train_config.weight_decay,
            save_total_limit=train_config.save_total_limit,
            num_train_epochs=train_config.num_train_epochs,
            predict_with_generate=train_config.predict_with_generate,
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model, return_tensors="pt")
        trainer = Seq2SeqTrainer(
            self.model,
            args=args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        trainer.train()
        if self.save_to_disk:
            train_config.save(os.path.join(self.dir_path, "train_config.json"))
            trainer.save_model(self.dir_path)



    def save_args(self):
        filename = os.path.join(self.dir_path, "args.json")
        if os.path.exists(filename):
            return
        with open(filename, "w") as f:
            info = {
                "model_name": self.model_name,
                "src_language": self.src_language,
                "tgt_language": self.tgt_language,
                "max_input_length": self.max_input_length,
                "max_target_length": self.max_target_length,
            }
            json.dump(info, f)
        

    @classmethod
    def from_pretrained(cls, path):
        args = json.load(open(os.path.join(path, "args.json")))

        tokenizer = AutoTokenizer.from_pretrained(path)
        model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(path)

        return HuggingFaceTranslationModel(
            args["model_name"],
            args["src_language"],
            args["tgt_language"],
            tokenizer,
            model,
            save_to_disk=False,
            max_input_length=args["max_input_length"],
            max_target_length=args["max_target_length"],
        )
