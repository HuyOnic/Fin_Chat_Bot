import torch 
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from dotenv import load_dotenv
load_dotenv()

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
print(LLM_MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

def get_llm_response(prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
    """
    Sinh câu trả lời từ mô hình LLaMA 3 7B.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            streamer=None,  # bỏ qua streaming nếu cần
            pad_token_id=tokenizer.eos_token_id
        )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Loại bỏ phần prompt đầu ra (nếu có)
    return output_text.replace(prompt, "").strip()

