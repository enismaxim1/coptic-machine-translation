## Neural Machine Translator

Code Credits: https://nlp.seas.harvard.edu/annotated-transformer/


A Python module for creating custom neural machine translators from a source to target language.

### Installing the Package
To install the `translation` package, you must first ensure that your virtual environment has all the required dependencies. TODO: add dependency management.
Install the package using pip install -e translation/.

### Creating a Translator
To create a translator, use the following code:
```
from models import TranslationModel
translator = TranslationModel('{src_language}', '{tgt_language}', dataset)
```
where `dataset` is a HuggingFace-style dataset containing a train, validation, and test split, each with a single 'translation' feature of source-target dictionaries of parallel sentences.

The first time this is run, the model will train on your system's GPUs. After it has trained, the results will be cached and future retrievals will not be needed.

To translate, use the `.translate` method on the `TranslationModel` with a source sentence argument:
`translated = translator.translate('{source sentence input}')`

### Creating a Dataset
If you have local parallel data files, then you can use the `load_dataset` function in the `src/data_utils.py` module. Store your data with the files 
```
parent/train/{src}.txt
parent/train/{tgt}.txt
parent/validation/{src}.txt
parent/validation/{tgt}.txt
parent/test/{src}.txt
parent/test/{tgt}.txt
```
Then, call the function like so: `load_datasets('parent', '{src}-{tgt}')` to generate a dataset in the proper form.












