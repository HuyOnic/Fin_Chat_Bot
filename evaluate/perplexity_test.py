import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import pandas as pd
from typing import List

model_name = "NlpHUST/gpt2-vietnamese"
pp_tokenizer = AutoTokenizer.from_pretrained(model_name)
pp_model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

def compute_ppl_list(sentences: List[str]) -> List[float]:
    perplexities = []
    for sentence in sentences:
        # Tokenize từng câu
        inputs = pp_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to("cuda")

        # Tính loss và perplexity
        with torch.no_grad():
            outputs = pp_model(input_ids, labels=input_ids)
            loss = outputs.loss
            ppl = math.exp(loss.item())
            perplexities.append(ppl)

    return perplexities


if __name__=="__main__":
    path = "/home/goline/huy/quant_chat_bot/LLM_Project/data/similarity_results.csv"
    df = pd.read_csv(path)
    df["perplexity"] = compute_ppl_list(df["predicted_answer"].to_list())
    df.to_csv(path, index=False)
