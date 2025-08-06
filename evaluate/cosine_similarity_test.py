import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from openai import OpenAI
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.api.chatbot_engine import rounting # import routing vao script hien tai.

summarize_model = OpenAI(
    base_url=os.getenv("OPENAI_API_BASE_URL", "http://localhost:8000/v1"),
    api_key="EMPTY"
)
model = SentenceTransformer("BAAI/bge-m3")

def summarize_text(context):
    response = summarize_model.chat.completions.create(
        model=os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo"),
        messages=[
            {"role": "system", "content": "Bạn là một trợ lý AI giỏi tóm tắt các nội dung về tài chính."},
            {"role": "user", "content": f"Hãy tóm tắt ngắn gọn những số liệu trong nội dung sau bằng Tiếng Việt, không sử dụng Tiếng Anh:\n\n{context}"}
        ],
        temperature=0.5,
        max_tokens=512
    )
    return response.choices[0].message.content

# def senmatic_similarity_test():
#     df = pd.read_excel("data/Output_test.xlsx")
#     df = df.reset_index(drop=True)
#     result_df = df[["intent", "question", "answer"]].copy().rename(columns={"answer": "expected_answer"})

#     result_df["predicted_answer"] = ""
#     result_df["summarize_expected_answer"] = result_df["expected_answer"].copy()
#     result_df["summarize_predicted_answer"] = result_df["expected_answer"].copy()
#     result_df["similarity"] = 0.0

#     for idx, row in tqdm(df.iterrows(), total=len(df)):
#         question = str(row["question"])
#         expected_answer = str(row["answer"])
#         predicted_answer = ""

#         try:
#             response = requests.post(
#                 "http://localhost:8006/test_chat",
#                 headers={"accept": "application/json", "Content-Type": "application/json"},
#                 json={"message": question},
#                 timeout=20
#             )
#             if response.status_code == 200:
#                 predicted_answer = response.json().get("message", "")
#         except Exception as e:
#             print(f"❌ Lỗi API cho câu hỏi: {question} → {e}")

#         result_df.at[idx, "predicted_answer"] = predicted_answer

#         if predicted_answer and predicted_answer.strip():
#             if len(expected_answer) > 200:
#                 expected_answer = summarize_text(expected_answer)
#                 result_df.at[idx, "summarize_expected_answer"] = expected_answer

#             if len(predicted_answer) > 200:
#                 predicted_answer = summarize_text(predicted_answer)

#             emb1 = model.encode(expected_answer, convert_to_tensor=True)
#             emb2 = model.encode(predicted_answer, convert_to_tensor=True)
#             similarity = util.cos_sim(emb1, emb2).item()
#         else:
#             similarity = 0.0

#         result_df.at[idx, "summarize_predicted_answer"] = predicted_answer
#         result_df.at[idx, "similarity"] = similarity

#     columns = ["intent", "question", "expected_answer", "predicted_answer", "summarize_expected_answer", "summarize_predicted_answer", "similarity"]
#     result_df[columns].to_csv("data/similarity_results.csv", index=False)
#     print("✅ Đã ghi kết quả vào data/similarity_results.csv")
#     print("Similarity trung bình:", result_df["similarity"].mean())

def senmatic_similarity_test():
    df = pd.read_excel("data/Output_test.xlsx")
    df = df.reset_index(drop=True)
    result_df = df[["intent", "question", "answer"]].copy().rename(columns={"answer": "expected_answer"})

    result_df["predicted_answer"] = ""
    result_df["summarize_expected_answer"] = result_df["expected_answer"].copy()
    result_df["summarize_predicted_answer"] = result_df["expected_answer"].copy()
    result_df["similarity"] = 0.0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = str(row["question"])
        expected_answer = str(row["answer"])
        predicted_answer = ""

        try:
            predicted_answer = rounting(question) # goi truc tiep rounting thay vi goi api tu localhost.
        except Exception as e:
            print(f"❌ Lỗi trả lời cho câu hỏi: {question} → {e}")

        result_df.at[idx, "predicted_answer"] = predicted_answer

        if predicted_answer and predicted_answer.strip():
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

        result_df.at[idx, "summarize_predicted_answer"] = predicted_answer
        result_df.at[idx, "similarity"] = similarity

    columns = ["intent", "question", "expected_answer", "predicted_answer", "summarize_expected_answer", "summarize_predicted_answer", "similarity"]
    result_df[columns].to_csv("data/similarity_results.csv", index=False)
    print("✅ Đã ghi kết quả vào data/similarity_results.csv")
    print("Similarity trung bình:", result_df["similarity"].mean())


if __name__ == "__main__":
    senmatic_similarity_test()
