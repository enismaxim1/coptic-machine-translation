import pandas as pd
from collections import defaultdict
import random
import re
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class DictionaryTranslator():
    def __init__(self):
        dictionary_data = pd.read_csv("datasets/raw_dictionary.csv")
        
        self.cop_en_dict = defaultdict(list)
        self.en_cop_dict = defaultdict(list)

        for _, row in dictionary_data.iterrows():
            eng = row["eng"]
            translation = re.sub(r'\([^)]*\)', '', eng)
            translation = re.sub(r'\([^)]$', '', translation)

            eng = translation.split(",")
            for e in eng:
                stripped_e = e.strip()
                self.cop_en_dict[row["coptic"]].append(stripped_e)
                self.en_cop_dict[stripped_e].append(row["coptic"])
        # self.init_eng_cop()

    def _get_dictionry(self, src, tgt):
        if src == "cop" and tgt == "en":
            return self.cop_en_dict
        elif src == "en" and tgt == "cop":
            return self.en_cop_dict
        else:
            raise ValueError("Invalid source or target language")
    def translate(self, word, src="cop", tgt="en"):
        dictionary = self._get_dictionry(src, tgt)
        entry = dictionary[word]
        if len(entry) == 0:
            return "?"
        rand_idx = random.randint(0, len(entry) - 1)
        translation = entry[rand_idx]
        # remove parenthesised text
        return translation

    def translate_sentence(self, sentence, src="cop", tgt="en"):
        dictionary = self._get_dictionry(src, tgt)
        # remove punctuations
        sentence = sentence.replace(".", "").replace(",", "").replace("?", "").replace("!", "")
        words = sentence.split(" ")
        translation = [self.translate(word, src, tgt) for word in words]
        return " ".join(translation)

    # Function to vectorize text using word embeddings
    def vectorize_text(self, text):
        words = text.lower().split()
        word_vectors = [self.model[word] for word in words if word in self.model]
        if len(word_vectors) == 0:
            print("FUCK", text)
        return np.mean(word_vectors, axis=0)

    def init_eng_cop(self):
        english_coptic_dict = {
            k: v[0] for k, v in self.en_cop_dict.items()
        }
        # Load pre-trained word embeddings
        self.model = api.load("glove-wiki-gigaword-100")
        self.english_coptic_dict = english_coptic_dict
        
        # Vectorize all definitions
        self.definition_vectors = {definition: self.vectorize_text(definition) for definition in self.english_coptic_dict}

    def find_closest_coptic_word(self, english_word):
        english_vector = self.vectorize_text(english_word)
        similarities = {definition: cosine_similarity([english_vector], [vec])[0][0] for definition, vec in self.definition_vectors.items()}
        closest_definition = max(similarities, key=similarities.get)
        return self.english_coptic_dict[closest_definition]

# t = DictionaryTranslator()
# # Example usage
# english_word = "cake"
# coptic_word = t.find_closest_coptic_word(english_word)
# print(f"The closest Coptic word for '{english_word}' is '{coptic_word}'")
