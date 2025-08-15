from pydantic import BaseModel, Field
from langchain.tools import Tool
from app.rag.collections import COLLECTIONS


def search_stock_collection(query, top_k=5, threshold=0.2):
    docs = COLLECTIONS["stock"].hybrid_search(query, top_k, threshold)
    return [doc.payload for doc in docs]

def search_news_collection(query, top_k=5, threshold=0.2):
    docs = COLLECTIONS["news"].hybrid_search(query, top_k, threshold)
    return [doc.payload for doc in docs]

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