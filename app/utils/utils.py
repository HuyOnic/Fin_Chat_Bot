import os
import re
from docling.document_converter import DocumentConverter
import numpy as np
from underthesea import word_tokenize


stop_word = set()
base_dir = os.path.dirname(os.path.abspath(__file__))
stop_word_file = os.path.join(base_dir, "../../data/vietnamese_stopwords.txt")
with open(stop_word_file, "r", encoding="utf-8") as f:
    for line in f:
        word = line.strip()
        if word:   
            stop_word.add(word)


def convert_pdf_to_markdown(path):
    converter = DocumentConverter()
    result = converter.convert(path)
    return result.document.export_to_markdown()


def get_files_in_directory(path):
    files = []
    for f in os.listdir(path):
        path_f = os.path.join(path, f)
        if os.path.isfile(path_f):
            files.append(path_f)
        elif os.path.isdir(path_f):
            files.extend(get_files_in_directory(path_f))

    return files

def normalize(scores):
    scores = np.array(scores, dtype=np.float32)
    mean = scores.mean()
    std = scores.std()

    lower = mean - 3 * std
    upper = mean + 3 * std

    if upper == lower:
        return np.ones_like(scores) * 0.5

    normalized = (scores - lower) / (upper - lower)
    normalized = np.clip(normalized, 0, 1)
    return normalized.tolist()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", '', text)
    tokens = word_tokenize(text)
    tokens = [word.replace(' ', '_') for word in tokens if word not in stop_word]
    return ' '.join(tokens)