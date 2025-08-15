import os
from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastembed import SparseTextEmbedding
from app.utils.utils import preprocess_text
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

MODEL_DIR = os.getenv("EMBEDDING_MODEL", "./models/sentence_transformers")
SPARSE_MODEL = "Qdrant/bm25"
#MODEL_DIR = "BAAI/bge-m3"
print(f"Đang load model từ: {MODEL_DIR}")

try:
    bert_model = SentenceTransformer(MODEL_DIR, device="cpu")
    sparse_model = SparseTextEmbedding(SPARSE_MODEL, cache_dir="models",
                                                          cuda=False)
    print("✅ Embedding Model đã được load thành công!")
except Exception as e:
    print(f"Lỗi khi load model: {e}")
    raise


