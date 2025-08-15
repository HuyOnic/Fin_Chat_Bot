from app.db.qdrant import VectorStore
from app.embedding.models import DenseEmbedding, SparseEmbedding

dense_model = DenseEmbedding()
sparse_model = SparseEmbedding()  

COLLECTIONS = {
    "stock": VectorStore(collection_name="stockcode_hybrid_searchz", dense_model=dense_model, sparse_model=sparse_model),
    "news": VectorStore(collection_name="news_hybrid_searchz", dense_model=dense_model, sparse_model=sparse_model),
}


