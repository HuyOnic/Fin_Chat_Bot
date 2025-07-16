from typing import Dict, Any
from langchain.evaluation import StringEvaluator
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from rouge_score.rouge_scorer import RougeScorer
import nltk
# nltk.download('punkt_tab')
import pandas as pd
import requests
""" A custom evaluator that computes BLEU and ROUGE scores for text outputs. """

class BLEUROUGEEvaluator(StringEvaluator):
    def __init__(self):
        self.rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    def _evaluate_strings(self, prediction: str, reference: str, **kwargs) -> Dict[str, Any]:
        # Tính BLEU
        ref_tokens = [word_tokenize(reference)]
        pred_tokens = word_tokenize(prediction)
        bleu_score = sentence_bleu(ref_tokens, pred_tokens)
        
        # Tính ROUGE
        rouge_scores = self.rouge_scorer.score(reference, prediction)
        
        return {
            "bleu": bleu_score,
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rouge2": rouge_scores["rouge2"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure,
            "avg_score": (bleu_score + rouge_scores["rougeL"].fmeasure) / 2  # Combined score
        }
if __name__ == "__main__":
    api_url = "http://localhost:8083/test_chat"

    df = pd.read_excel("/home/goline/huy/quant_chat_bot/LLM_Project/data/gpt_test.xlsx")
    df.dropna(subset=["question", "answer"], inplace=True)
    dataset = df[["question", "answer"]].rename(columns={"question":"input",
                                                        "answer":"output"}).to_dict(orient="records")
    # Sử dụng
    avg_bleu = 0.0
    avg_rouge1 = 0.0
    avg_rouge2 = 0.0
    avg_rougeL = 0.0
    avg_score = 0.0
    evaluator = BLEUROUGEEvaluator()
    for item in dataset:
        data = {
            "message": item["input"]
        }
        response = requests.post(api_url, json=data)
        # Kiểm tra kết quả
        if response.status_code == 200:
            result = response.json()
            bot_message = result[1]    
            print(f"Bot response: {bot_message}") 
            result = evaluator.evaluate_strings(
                prediction=bot_message,
                reference=item["output"]
            )
            avg_bleu += result["bleu"]
            avg_rouge1 += result["rouge1"]
            avg_rouge2 += result["rouge2"]
            avg_rougeL += result["rougeL"]
            avg_score += result["avg_score"]
        else:
            print(f"Lỗi khi gọi API: {response.status_code} - {response.text}")
    
    avg_count = len(dataset)
    print(f"Average BLEU: {avg_bleu / avg_count}")
    print(f"Average ROUGE-1: {avg_rouge1 / avg_count}")
    print(f"Average ROUGE-2: {avg_rouge2 / avg_count}")
    print(f"Average ROUGE-L: {avg_rougeL / avg_count}")
    print(f"Average Combined Score: {avg_score / avg_count}")
