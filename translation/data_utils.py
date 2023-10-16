import json
import os
import datasets
from datasets import Dataset, DatasetDict


def make_splits(
    dir_path: str,
    translation_task: str,
    train_split: float = 0.8,
    validation_split: float = 0.1,
    test_split: float = 0.1,
):
    """
    Split two parallel datasets into train, validation, and test sets. Datasets must be named as {src_language}.txt and {tgt_language}.txt.
    dir_path: path to directory containing source and target files.
    translation_task: string of the form "src_language-tgt_language".
    train_split: (Optional) fraction of dataset to use for training. Defaults to 0.8.
    valid_split: (Optional) fraction of dataset to use for validation. Defaults to 0.1.
    test_split: (Optional) fraction of dataset to use for testing. Defaults to 0.1.
    """
    # TODO: implement with streaming for very big datasets
    src_language, tgt_language = translation_task.split("-")
    src_path = os.path.join(dir_path, f"{src_language}.txt")
    tgt_path = os.path.join(dir_path, f"{tgt_language}.txt")
    if not os.path.exists(os.path.join(dir_path, src_path)):
        raise FileNotFoundError(f"{src_path} not found.")
    if not os.path.exists(os.path.join(dir_path, tgt_path)):
        raise FileNotFoundError(f"{tgt_path} not found.")
    src_lines = open(src_path, "r", encoding="utf-8").readlines()
    tgt_lines = open(tgt_path, "r", encoding="utf-8").readlines()
    if not len(src_lines) == len(tgt_lines):
        raise ValueError(
            f"Line count for src {src_path} ({len(src_lines)}) does not match line count for tgt {tgt_path} ({len(tgt_lines)})."
        )
    num_lines = len(src_lines)

    num_train, num_validation, num_test = (
        int(train_split * num_lines),
        int(validation_split * num_lines),
        int(test_split * num_lines),
    )
    train_dir, validation_dir, test_dir = (
        os.path.join(dir_path, "train"),
        os.path.join(dir_path, "validation"),
        os.path.join(dir_path, "test"),
    )
    for directory in [train_dir, validation_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            raise FileExistsError(
                f"{directory} already exists. Cannot generate splits."
            )

    for lines, language in zip([src_lines, tgt_lines], [src_language, tgt_language]):
        train_path = os.path.join(train_dir, f"{language}.txt")
        validation_path = os.path.join(validation_dir, f"{language}.txt")
        test_path = os.path.join(test_dir, f"{language}.txt")
        for filename in [train_path, validation_path, test_path]:
            if os.path.exists(train_path):
                raise FileExistsError(f"{filename} already exists.")

        with open(train_path, "w") as train:
            train.writelines(lines[:num_train])

        with open(validation_path, "w") as train:
            train.writelines(lines[num_train : num_train + num_validation])

        with open(test_path, "w") as train:
            train.writelines(lines[num_train + num_validation :])


def translations(src_filepath: str, tgt_filepath, src_language: str, tgt_language: str):
    """Load parallel data from files in data_dir."""
    if not os.path.exists(src_filepath):
        raise FileNotFoundError(f"{src_filepath} not found.")
    if not os.path.exists(tgt_filepath):
        raise FileNotFoundError(f"{tgt_filepath} not found.")

    with open(src_filepath, "r", encoding="utf-8") as src_file, open(
        tgt_filepath, "r", encoding="utf-8"
    ) as tgt_file:
        return [
            {src_language: src_sentence.strip(), tgt_language: tgt_sentence.strip()}
            for src_sentence, tgt_sentence in zip(src_file, tgt_file)
        ]


def set_translation_json(data_dir, src_language: str, tgt_language: str):
    if os.path.exists(os.path.join(data_dir, "translations.json")):
        raise FileExistsError(
            f"{os.path.join(data_dir, 'translations.json')} already exists."
        )

    src_filepath = os.path.join(data_dir, f"{src_language}.txt")
    tgt_filepath = os.path.join(data_dir, f"{tgt_language}.txt")
    """Load parallel data from files in data_dir into a json file."""
    if not os.path.exists(src_filepath):
        raise FileNotFoundError(f"{src_filepath} not found.")
    if not os.path.exists(tgt_filepath):
        raise FileNotFoundError(f"{tgt_filepath} not found.")

    with open(src_filepath, "r", encoding="utf-8") as src_file, open(
        tgt_filepath, "r", encoding="utf-8"
    ) as tgt_file:
        translations = [
            {
                "translation": {
                    src_language: src_sentence.strip(),
                    tgt_language: tgt_sentence.strip(),
                }
            }
            for src_sentence, tgt_sentence in zip(src_file, tgt_file)
            if src_sentence and tgt_sentence
        ]

        with open(os.path.join(data_dir, "translations.json"), "w") as f:
            json.dump(translations, f)


def load_dataset(data_dir: str, translation_task: str, split=None):
    """
    Load dataset from files in data_dir.
    data_dir: path to directory containing train, validation, and test files.
    translation_task: string of the form "src_language-tgt_language".
    split: optional string specifying which split to load.
    """

    splits = ["train", "validation", "test"] if not split else [split]
    data_files = {
        split: os.path.join(data_dir, split, f"translations.json") for split in splits
    }

    try:
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"{data_dir} not found.")
        data = datasets.load_dataset("json", data_files=data_files, split=split)
        for split in splits:
            data[split].data_dir = data_dir
        data.data_dir = data_dir
        return data

    except FileNotFoundError:
        dataset = datasets.load_dataset(data_dir, translation_task)
        for split in splits:
            dataset[split].data_dir = data_dir

    dataset.data_dir = data_dir
    return dataset
