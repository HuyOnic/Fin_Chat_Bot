import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from openai import OpenAI
import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..")))
from app.api.chatbot_engine import rounting


summarize_model = OpenAI(
    # model=os.getenv("LLM_MODEL_NAME"),
    base_url=os.getenv("OPENAI_API_BASE_URL", "http://localhost:8000/v1"),
    api_key="EMPTY"
)
# Load mô hình embedding
model = SentenceTransformer("BAAI/bge-m3")

def summarize_text(context):
    response = summarize_model.chat.completions.create(
        model=os.getenv("LLM_MODEL_NAME"),
        messages=[
            {"role": "system", "content": "Bạn là một trợ lý AI giỏi tóm tắt các nội dung về tài chính."},
            {"role": "user", "content": f"Hãy tóm tắt ngắn gọn những số  liệu trong nội dung sau bằng Tiếng Việt, không sử dụng Tiếng Anh:\n\n{context}"}
        ],
        temperature=0.5,
        max_tokens=512
    )
    return response.choices[0].message.content

def senmatic_similarity_test():
    # Load file CSV (đảm bảo có cột 'question', 'answer')
    df = pd.read_csv("/home/goline/huy/quant_chat_bot/LLM_Project/data/output_test_v1.csv")  # 🔁 thay bằng tên file thật của bạn
    result_df = df[["intent", "question", "answer"]].copy().rename(columns={"answer": "expected_answer"})

    result_df["predicted_answer"] = ""
    result_df["summarize_expected_answer"] = result_df["expected_answer"].copy()
    result_df["similarity"] = 0.0

    # Lặp qua từng câu hỏi
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = str(row["question"])
        expected_answer = str(row["answer"])
        predicted_answer = ""

        predicted_answer = rounting(question)

        result_df.at[idx, "predicted_answer"] = predicted_answer

        # Tính cosine similarity nếu có dữ liệu
        if predicted_answer.strip():
            if len(expected_answer) > 200:
                expected_answer = summarize_text(expected_answer)
                result_df.at[idx, "summarize_expected_answer"] = expected_answer
            
            if len(predicted_answer) > 200:
                predicted_answer = summarize_text(predicted_answer)

            emb1 = model.encode(expected_answer, convert_to_tensor=True)
            emb2 = model.encode(predicted_answer, convert_to_tensor=True)
            similarity = util.cos_sim(emb1, emb2).item()
        else:
            similarity = 0.0

        # Ghi lại
        result_df.at[idx, "summarize_predicted_answer"] = predicted_answer
        result_df.at[idx, "similarity"] = similarity

    # Lưu ra file mới
    columns = ["intent", "question", "expected_answer", "predicted_answer", "summarize_expected_answer", "summarize_predicted_answer", "similarity"]
    result_df[columns].to_csv("/home/goline/huy/quant_chat_bot/LLM_Project/data/similarity_results.csv", index=False)
    print("SENTIMENT SECORE:", similarity)
    print("✅ Đã ghi kết quả vào data/similarity_results.csv")

if __name__=="__main__":
    senmatic_similarity_test()