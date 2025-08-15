from pydantic import BaseModel, Field
from langchain.tools import Tool
from app.db.qdrant import NEWS_COLLECTION_NAME, STOCKCODE_COLLECTION_NAME, get_documents_by_vector
from app.utils.vectorizer import convert_to_dense_vector, convert_to_sparse_vector


def search_stock_collection(query, top_k=5, threshold=0.2):
    dense_vector = convert_to_dense_vector(query)
    sparse_vector = convert_to_sparse_vector(query)
    return get_documents_by_vector(dense_vector, sparse_vector, top_k, threshold, STOCKCODE_COLLECTION_NAME)

def search_news_collection(query, top_k=5, threshold=0.2):
    dense_vector = convert_to_dense_vector(query)
    sparse_vector = convert_to_sparse_vector(query)
    return get_documents_by_vector(dense_vector, sparse_vector, top_k, threshold, NEWS_COLLECTION_NAME)

class Input(BaseModel):
    Query: str = Field(..., description="Câu hỏi truy vấn")
    top_k: int = Field(5, description="Số lượng tài liệu liên quan tối đa cần truy xuất")
    threshold: float = Field(0.3, description="Ngưỡng độ tin cậy cho kết quả.")

stock_rag_search_tool = Tool.from_function(
    func=search_stock_collection,
    name="search_stock_data",
    description="Thực hiện tìm kiếm trong tập tài liệu chứng khoán.",
    input_schema=Input
)

news_rag_search_tool = Tool.from_function(
    func=search_news_collection,
    name="search_news_data",
    description="Thực hiện tìm kiếm trong tập tài liệu tin tức.",
    input_schema=Input
)