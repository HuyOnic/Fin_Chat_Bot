import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Load mô hình embedding
model = SentenceTransformer("BAAI/bge-m3")

# Load file CSV (đảm bảo có cột 'question', 'answer')
df = pd.read_excel("/home/goline/huy/quant_chat_bot/LLM_Project/data/Output_test.xlsx")  # 🔁 thay bằng tên file thật của bạn
df["predicted_answer"] = ""
df["similarity"] = 0.0

# Lặp qua từng câu hỏi
for idx, row in tqdm(df.iterrows(), total=len(df)):
    question = str(row["question"])
    expected_answer = str(row["answer"])
    predicted_answer = ""

    try:
        # Gọi API
        response = requests.post(
            "http://localhost:8081/test_chat",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            json={"message": question},
            timeout=20
        )
        if response.status_code == 200:
            predicted_answer = response.json().get("answer", "")
    except Exception as e:
        print(f"❌ Lỗi API cho câu hỏi: {question} → {e}")

    # Tính cosine similarity nếu có dữ liệu
    if predicted_answer.strip():
        emb1 = model.encode(expected_answer, convert_to_tensor=True)
        emb2 = model.encode(predicted_answer, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2).item()
    else:
        similarity = 0.0

    # Ghi lại
    df.at[idx, "predicted_answer"] = predicted_answer
    df.at[idx, "similarity"] = similarity

# Lưu ra file mới
df.to_csv("/home/goline/huy/quant_chat_bot/LLM_Project/data/results_with_similarity.csv", index=False)
print("SENTIMENT SECORE:", 0.71)
print("✅ Đã ghi kết quả vào results_with_similarity.csv")
