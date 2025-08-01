import os, requests
from dotenv import load_dotenv
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

MODEL_DIR = os.getenv("SENTIMENT_MODEL_ID", "wonrax/phobert-base-vietnamese-sentiment")
sentiment_device = "cuda:1"

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Tổng số tham số: {total_params:,}")
    print(f"🧠 Số tham số có thể huấn luyện: {trainable_params:,}")
    return total_params, trainable_params

try:
    sentiment_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=3)
    count_parameters(sentiment_model)
    sentiment_model.to(sentiment_device)
    print("✅ Sentiment Model đã được load thành công!")
except Exception as e:
    print(f"Lỗi khi load model: {e}")
    raise

def sentiment_analysis(content_list: list):
    url = "http://localhost:8082/sentiment/predict"
    data = {"content_list": content_list}
    try:
        response = requests.post(url, data)
        return response.json()["scores"]
    except Exception as e:
        print("Lỗi khi gọi API sentiment:", e)

if __name__=="__main__":
    pass


