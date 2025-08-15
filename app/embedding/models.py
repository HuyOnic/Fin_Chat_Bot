import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from app.embedding._utils import preprocess_text

load_dotenv() 

class DenseEmbedding:
    def __init__(self, model_name=os.getenv("DENSE_MODEL")):
        self.model = SentenceTransformer(model_name, cache_folder=os.path.join(os.path.dirname(os.path.realpath(__file__)),"models"))

    def encode(self, texts):
        if isinstance(texts, str):
            return self.model.encode(texts)
        elif isinstance(texts, list):
            return [e.tolist() for e in self.model.encode(texts)]
        else:
            raise ValueError("Input must be a string or a list of strings.")
        
    def get_dimension(self):
        return self.encode("test").shape[0]

class SparseEmbedding:
    def __init__(self, model_name=os.getenv("SPARSE_MODEL")):
        self.model = SparseTextEmbedding(model_name, cache_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), "models"))

    def passage_embed(self, texts):
        if isinstance(texts, str):
            texts = preprocess_text(texts)
            return next(self.model.passage_embed([texts])).as_object()
        elif isinstance(texts, list):
            texts = [preprocess_text(text) for text in texts]
            return [embedding.as_object() for embedding in self.model.passage_embed(texts)]
        else:
            raise ValueError("Input must be a string or a list of strings.")
