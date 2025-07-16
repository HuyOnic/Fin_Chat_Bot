from app.chains.hybrid_rag_chain import build_hybrid_rag_chain
from app.retriever.hybrid_retriever import HybridRetriever
from app.retriever.dense_retriever import DenseRetriever
from app.retriever.sparse_retriever import SparseRetriever
from app.db.postgre import SessionLocal
from app.db.qdrant import client
from app.utils.vectorizer import bert_model

def answer_with_hybrid_rag(query):
    pg_session = SessionLocal()
    dense_retriever = DenseRetriever(client, bert_model)
    sparse_retriever = SparseRetriever(pg_session)
    retriever = HybridRetriever(dense_retriever, sparse_retriever)
    chain = build_hybrid_rag_chain(retriever)
    return chain.run(query)