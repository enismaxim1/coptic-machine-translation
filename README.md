## Neural Machine Translator

Code Credits: https://nlp.seas.harvard.edu/annotated-transformer/


A Python module for creating custom neural machine translators from a source to target language. To create a translator from a source to target language, add your train, validation, and test data to the path `models/{SRC}-{TGT}/data/` with file names `train.{SRC}`, `train.{TGT}`, `valid.{SRC}`, `valid.{TGT}`, `test.{SRC}`, and `test.{TGT}`. Next, simply create and train your translator with the following Python commands:

```
from models import TranslationModel
translator = TranslationModel('{SRC}', '{TGT}')
```

The first time this is run, the model will train on your system's GPUs. After it has trained, the results will be cached and future retrievals will not be needed.

To translate, use the `.translate` method on the `TranslationModel` with a source sentence argument:
`translated = translator.translate('{source sentence input}')`










