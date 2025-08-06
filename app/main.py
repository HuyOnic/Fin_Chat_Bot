from fastapi import FastAPI, Query
from app.api.crawl import run_crawler
from app.api.preprocess import check_and_update_duplicates
from app.api.update_sentiment import update_sentiment
from app.api.chatbot_engine import ask_bot, chat_bot, rounting, sentiment_news, sentiment_vn30f1m
from app.api.chatbot_engine import chat_pipeline
from app.db.postgre import insert_news
from pydantic import BaseModel
import requests, time, json
from typing import List 
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app.llm.llm_list import LOCAL_MODELS

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class Message(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str

class ChatInput(BaseModel):
    model: str = "Llama3_8B"
    messages: List[Message]
    temperature: float = 0.7

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc ["http://localhost:3000"] nếu bạn muốn giới hạn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def read_root():
    return {"message": "Hello trade world!"}

@app.post("/run-pipeline")
def run_pipeline(
    crawl_source: str = Query("all", description="Nguồn dữ liệu: 'cafef', 'vietstock', hoặc 'all'"),
    days: int = Query(1, description="Số ngày cần crawl"),
    threshold: float = Query(0.85, description="Ngưỡng similarity để kiểm tra trùng lặp")
):
    try:
        # # 1️. Crawl dữ liệu
        print("Bắt đầu Crawl dữ liệu...")
        all_data = run_crawler(crawl_source, days)
        if all_data: insert_news(all_data)

        # # 2️. Kiểm tra trùng lặp dữ liệu
        print("Bắt đầu kiểm tra trùng lặp dữ liệu...")
        preprocess_result = check_and_update_duplicates(all_data, threshold)
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
    return {"answer": ask_bot(question)}

# Testing
# @app.post("/test_hybrid_rag")
# def test_hybrid_rag(question: str):
#     return {"result:", answer_with_hybrid_rag(question)}

# @app.post("/test_chat")
# def chat(req: ChatRequest):
#     response = chat_bot(req.message)
#     return {"message:", response}

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Literal
import time
import uuid

app = FastAPI()

# Định nghĩa input và message format đúng chuẩn
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatInput(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[dict] = None
    user: Optional[str] = None

@app.get("/v1/models")
async def list_models():
    # Lấy danh sách các local models
    return JSONResponse(content={
        "object": "list",
        "data": LOCAL_MODELS
    })

@app.post("/v1/chat/completions")
async def chat_endpoint(input: ChatInput):
    user_message = input.messages[-1].content
    assistant_reply = rounting(user_message)

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())

    if input.stream:
        # Trả từng dòng dạng event-stream
        def event_stream():
            # chunk đầu (dòng dữ liệu)
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": input.model,
                "choices": [{
                    "delta": {"role": "assistant", "content": assistant_reply},
                    "index": 0,
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            # chunk kết thúc
            done_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": input.model,
                "choices": [{
                    "delta": {},
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(done_chunk)}\n\ndata: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Nếu không stream
    response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": input.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": assistant_reply
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 27,
            "completion_tokens": len(assistant_reply.split()),
            "total_tokens": 27 + len(assistant_reply.split())
        }
    }

    return JSONResponse(content=response)



@app.post("/v2/chat/completions")
async def chat_endpoint_v2(input: ChatInput):
    user_message = input.messages[-1].content
    assistant_reply = chat_pipeline(user_message)

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())

    if input.stream:
        # Trả từng dòng dạng event-stream
        def event_stream():
            # chunk đầu (dòng dữ liệu)
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": input.model,
                "choices": [{
                    "delta": {"role": "assistant", "content": assistant_reply},
                    "index": 0,
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            # chunk kết thúc
            done_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": input.model,
                "choices": [{
                    "delta": {},
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(done_chunk)}\n\ndata: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Nếu không stream
    response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": input.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": assistant_reply
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 27,
            "completion_tokens": len(assistant_reply.split()),
            "total_tokens": 27 + len(assistant_reply.split())
        }
    }

    return JSONResponse(content=response)
    
@app.post("/test_sentiment_analysis")
def test_sentiment_news(req: ChatRequest):
    try:
        prompt = sentiment_news(req.message)
        return {"message": prompt}
    except Exception as e:
        return e
    
@app.post("/sentiment_vn30f1m/chat/completions")
def test_sentiment_vn30f1m(req: ChatRequest):
    try:
        return {"text": sentiment_vn30f1m()}
    except Exception as e:
        return {"text": "Hỏi vớ hỏi vẩn"}
    
if __name__=="__main__":
    chat_bot("hello")
    
