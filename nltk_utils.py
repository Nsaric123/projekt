import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
import re

stemmer = PorterStemmer()

# Tokenizacija (split na riječi)
def tokenize(sentence):
    return re.findall(r"\b\w+\b", sentence.lower())

# Stemmanje riječi (npr. "studiraš" -> "studira")
def stem(word):
    return stemmer.stem(word.lower())

# Pretvaranje rečenice u bag-of-words vektor
def bag_of_words(tokenized_sentence, all_words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag
