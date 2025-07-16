import os
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client, wrappers
from openai import OpenAI
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
import pandas as pd 
import requests
import re

client = Client()
name = "QA_dataset_1"

try:
    dataset = client.create_dataset(
        dataset_name=name, description="A sample dataset in LangSmith."
    )
except:
    dataset = client.read_dataset(dataset_name=name)
# Create examples
df = pd.read_excel("/home/goline/huy/quant_chat_bot/LLM_Project/data/gpt_test.xlsx")
df.dropna(subset=["question", "answer"], inplace=True)
print("number questions:", len(df))
# Tạo danh sách examples theo định dạng yêu cầu
examples = [
    {
        "inputs": {"question": row["question"]},
        "outputs": {"answer": row["answer"]}
    }
    for _, row in df.iterrows()
]
# Add examples to the dataset
client.create_examples(dataset_id=dataset.id, examples=examples)

openai_client = wrappers.wrap_openai(OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
))

def target(inputs: dict) -> dict:
    api_url = "http://localhost:8083/test_chat"
    data = {
        "message": inputs["question"]
    }
    response = requests.post(api_url, json=data)
        # Kiểm tra kết quả
    if response.status_code == 200:
        result = response.json()
        bot_message = result[1]    
        return {"answer": bot_message}
    else:
        print(f"Lỗi khi gọi API: {response.status_code} - {response.text}")
        return {"answer": "Không thể trả lời câu hỏi do lỗi hệ thống."}

judge_model = wrappers.wrap_openai(OpenAI(
    api_key="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"
))

def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    system_prompt = (
        "Bạn là một chuyên gia so sánh câu trả lời của mô hình với đáp án chuẩn bằng Tiếng Việt. Cho điểm từ 0 đến 5 dựa trên mức độ đúng, đầy đủ, và logic của câu trả lời so với đáp án chuẩn (câu trả lời sát với đáp án chuẩn thị score càng cao)"
        "Trả về cuối chuỗi score mà bạn chấm với dạng:"
        "\nscore:"
    )
    user_prompt = f"""\
    Trả lời bằng Tiếng Việt.
    Câu hỏi: {inputs["question"]}
    Câu trả lời của mô hình:
    {outputs["answer"]}
    Đáp án chuẩn:
    {reference_outputs["answer"]}
    Hãy cho điểm từ 0 đến 5 và giải thích ngắn gọn:"""

    response = judge_model.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=512,
    )

    content = response.choices[0].message.content.strip()
    # Tìm số điểm trong đoạn text, ví dụ: "score of 0.5 out of 1.0"
    pattern = r"score:\s*([0-9]+(?:\.[0-9]+)?)"
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        score = float(match.group(1))
    else:
        score = 0.0  # fallback nếu không parse được

    return {
        "key": "correctness",
        "score": score,
        "comment": content  # lưu lại câu trả lời để trace
    }

if __name__=="__main__":
    # experiment_results = client.evaluate(
    #     target,
    #     data=name,
    #     evaluators=[
    #         correctness_evaluator
    #     ],
    #     experiment_prefix="exp2",
    #     max_concurrency=2,
    # )
    report = {
        "inputs": [],
        "outputs": [],
        "reference_outputs": [],
        "comment": [],
        "score": []
    }
    for data in examples:
        inputs = data["inputs"]
        outputs = target(inputs)
        reference_outputs = data["outputs"]
        result = correctness_evaluator(inputs, outputs, reference_outputs)
        # print("Inputs:",inputs["question"])
        # print("Outputs:",outputs["answer"])
        # print("Reference Outputs:",reference_outputs["answer"])
        # print("Comment:", result["comment"])
        # print("Result:",result["score"])
        report["inputs"].append(inputs["question"])
        report["outputs"].append(outputs["answer"]) 
        report["reference_outputs"].append(reference_outputs["answer"])
        report["comment"].append(result["comment"])
        report["score"].append(result["score"])

    report_df = pd.DataFrame(report, columns=["inputs", "outputs", "reference_outputs", "comment", "score"])
    print(report_df.head())
    report_df.to_csv("/home/goline/huy/quant_chat_bot/LLM_Project/data/report.csv", index=False)
    total_llm_as_judge_scores = report_df["score"].sum()
    max_score = (5* len(report_df))
    print(f"Total LLM as judge scores: {total_llm_as_judge_scores:.2f}/{max_score}")