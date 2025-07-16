import requests
from concurrent.futures import ThreadPoolExecutor
from app.db.postgre import get_all_pending_sentiment, update_sentiment_score, update_status
import unicodedata
import time
from dotenv import load_dotenv
import os
from pathlib import Path

# Load file .env
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

MAX_LEN = 10000
SENTIMENT_API = os.getenv("SENTIMENT_API_URL")
executor = ThreadPoolExecutor(max_workers=5)

def clean_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    cleaned_chars = [ch for ch in text if not unicodedata.category(ch).startswith('C')]
    return ''.join(cleaned_chars).strip()

def call_predict_api(text, max_retries=3, timeout=60):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Gọi API dự đoán lần {attempt}...")
            resp = requests.post(SENTIMENT_API, json={"text": text}, timeout=timeout)
            resp.raise_for_status()
            print(f"API trả về thành công ở lần {attempt}")
            return resp.json()
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            print(f"Lỗi khi gọi API lần {attempt}: {e}")
            if attempt < max_retries:
                print("Thử lại sau 5 giây...")
                time.sleep(5)
            else:
                print("Gọi API thất bại sau nhiều lần thử lại")
                raise

def process_sentiment(article):
    article_id, content = article
    print(f"Bắt đầu xử lý bài viết ID={article_id}")

    try:
        data = call_predict_api(content)

        if "score" in data:
            score = float(data["score"])
            print(f"Đã lấy được score: {score} cho bài viết ID={article_id}")
            update_sentiment_score(article_id, score)
            update_status(article_id, 2)
            print(f"Cập nhật status thành công cho bài viết ID={article_id}")
            return {"article_id": article_id, "score": score}
        else:
            print(f"Không tìm thấy score cho bài viết ID={article_id}")
            return {"article_id": article_id, "error": "Không có score"}

    except Exception as e:
        print(f"Lỗi khi xử lý bài viết ID={article_id}: {e}")
        return {"article_id": article_id, "error": str(e)}

def update_sentiment():
    print("Bắt đầu lấy danh sách các bài viết cần cập nhật sentiment score...")
    all_data = get_all_pending_sentiment()
    all_data = [data[:MAX_LEN] for data in all_data]

    if not all_data:
        print("Không có bài viết nào cần cập nhật.")
        return {"message": "Không có bài viết để cập nhật."}

    print(f"Tổng số bài viết cần xử lý: {len(all_data)}")
    
    results = list(executor.map(process_sentiment, all_data))

    print("Hoàn thành cập nhật sentiment score cho tất cả bài viết.")
    return {"message": "Hoàn thành cập nhật sentiment score", "results": results}
