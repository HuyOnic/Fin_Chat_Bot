import os
import re
import urllib.request
from underthesea import word_tokenize

stop_word = set()
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "res/vietnamese-stopwords.txt"), "r", encoding="utf-8") as f:
    for line in f:
        word = line.strip()
        if word:   
            stop_word.add(word)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", '', text)
    tokens = word_tokenize(text)
    tokens = [word.replace(' ', '_') for word in tokens if word not in stop_word]
    return ' '.join(tokens)

