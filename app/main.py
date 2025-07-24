from fastapi import FastAPI, Query
from app.api.crawl import run_crawler
from app.api.preprocess import check_and_update_duplicates
from app.api.update_sentiment import update_sentiment
from app.api.chatbot_engine import ask_bot, chat_bot, rounting, sentiment_news
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, world!"}

@app.post("/run-pipeline")
def run_pipeline(
    crawl_source: str = Query("all", description="Nguồn dữ liệu: 'cafef', 'vietstock', hoặc 'all'"),
    days: int = Query(1, description="Số ngày cần crawl"),
    threshold: float = Query(0.85, description="Ngưỡng similarity để kiểm tra trùng lặp")
):
    try:
        # # 1️. Crawl dữ liệu
        print("Bắt đầu Crawl dữ liệu...")
        crawl_result = run_crawler(crawl_source, days)
        print("Crawl thành công!")

        # # 2️. Kiểm tra trùng lặp dữ liệu
        print("Bắt đầu kiểm tra trùng lặp dữ liệu...")
        preprocess_result = check_and_update_duplicates(threshold)
        print("Kiểm tra trùng lặp thành công!")

        # 3️. Cập nhật Sentiment Score
        print("Bắt đầu cập nhật Sentiment Score...")
        sentiment_result = update_sentiment()
        print("Cập nhật Sentiment thành công!")

        return {
            "message": "Pipeline chạy thành công",
            # "crawl_result": crawl_result,
            "preprocess_result": preprocess_result,
            "sentiment_result": sentiment_result
        }

    except Exception as e:
        print(f"Lỗi khi chạy pipeline: {e}")
        return {"error": str(e)}
    
@app.post("/ask")
def ask(question: str):
    return {"answer:", ask_bot(question)}

# Testing
# @app.post("/test_hybrid_rag")
# def test_hybrid_rag(question: str):
#     return {"result:", answer_with_hybrid_rag(question)}

@app.post("/test_chat")
def chat(req: ChatRequest):
    response = chat_bot(req.message)
    return {"message:", response}

@app.post("/test_router_agent")
def route_question(req: ChatRequest):
    try:
        response = rounting(req.message)
        return {"context": response}
    except Exception as e:
        return e
    
@app.post("/test_sentiment_analysis")
def test_sentiment_news(req: ChatRequest):
    try:
        prompt = sentiment_news(req.message)
        return {"message": prompt}
    except Exception as e:
        return e
    
