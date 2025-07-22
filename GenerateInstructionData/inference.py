import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import pandas as pd
import re 

class InstructionGenerator:
    def __init__(self, model_id, document_path, output_path, limit):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
        self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048, pad_token_id=tokenizer.eos_token_id)

        self.documents = pd.read_csv(document_path)["content"].tolist()[:limit] if limit else pd.read_csv(document_path)["content"].tolist()
        self.output_path = output_path

    def extract_first_qa_pair(self, text):
        # Regex tổng quát cho mọi biến thể "Câu hỏi" / "Câu trả lời"
        pattern = r"""
            [\+\-\*\s]*        # Cho phép các dấu như +, -, *, khoảng trắng trước tiêu đề
            Câu\s*hỏi          # Câu hỏi / CÂU HỎI / câu hỏi
            \s*[:：]?\s*        # Dấu : hoặc ： (có thể có hoặc không)
            (.*?)              # Nội dung câu hỏi (non-greedy)
            \n?[\+\-\*\s]*     # Cho phép xuống dòng hoặc ký hiệu giữa 2 phần
            Câu\s*trả\s*lời    # Câu trả lời / viết hoa/thường
            \s*[:：]?\s*
            (.*?)(?:$|\n)      # Nội dung câu trả lời đến hết dòng/cuối văn bản
        """
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE | re.VERBOSE)
        if match:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            if question and answer:
                return question, answer
        return None, None

    def qa_document(self):
        def build_prompt(document: str):
            return f'''Đọc một cách cẩn thận tin tức sau đây, tự đưa ra 1 câu hỏi và 1 câu trả lời chính xác từ tin tức đó.
            
            Tin tức:
            \"\"\"{document}\"\"\"

            Đưa ra một cặp câu hỏi và câu trả lời chính xác từ tin tức trên. Trả về kết quả theo định dạng sau:
            + Câu hỏi:
            + Câu trả lời:
            '''

        with open(self.output_path, "a", encoding="utf-8") as f:
            for doc in tqdm(self.documents):
                prompt = build_prompt(doc)
                try:
                    output = self.pipeline(prompt)[0]["generated_text"]
                    # Loại bỏ phần prompt bị lặp lại nếu LLM echo
                    output = output.replace(prompt, "").strip()
                    question, answer = self.extract_first_qa_pair(output)
                    if question and answer:
                        json.dump({"instruction": question, "input": doc, "output": answer}, f, ensure_ascii=False)
                        f.write("\n")
                    else:
                        print("⚠️ Không tìm được cặp Q&A hợp lệ.")
                except Exception as e:
                    print(f"❌ Lỗi khi xử lý tài liệu: {e}")
        
        print(f"✅ Done. Results saved to {self.output_path}")    

    def task_phan_tich_tin_tuc(self):
        def build_prompt(document: str) -> str:
            return f'''Bạn là một chuyên gia trong lĩnh vực tài chính. Hãy phân tích xem tin tức sau có ảnh hưởng gì đến thị trường tài chính Việt Nam.
            Trong phần đưa ra kết luận, nhắc rõ ràng sự kiện xảy ra trong Tin tức, từ đó đưa ra kết luận, không sử dụng các từ ngữ chung chung mà phải đưa ra dẫn chứng rõ ràng

            Tin tức:
            """{document}"""

            bắt đầu câu trả lời bằng từ "Kết Luận:"
            '''
        
        with open(self.output_path, "a", encoding="utf-8") as f:
            for doc in tqdm(self.documents):
                prompt = build_prompt(doc)
                output = self.pipeline(prompt)[0]["generated_text"]
                
                output = output.replace(prompt,"")
                answer = re.search(r'Kết [Ll]uận[:：]?\s*(.*)', output, flags=re.DOTALL)
                if answer:
                    answer = answer.group(1).strip()
                    json.dump({"instruction": "phân tích tin tức", "input": doc, "output": answer}, f, ensure_ascii=False)
                    f.write("\n")

        print(f"✅ Done. Results saved to {self.output_path}")

if __name__=="__main__":
    #model_id = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct" 
    document_path = "/home/goline/huy/quant_chat_bot/sentiment_project/notebooks/news.csv"
    output_path = "/home/goline/huy/quant_chat_bot/sentiment_project/qa_output.jsonl"

    instruction_generator = InstructionGenerator(model_id=model_id, document_path=document_path, output_path=output_path, limit=None)
    #instruction_generator.task_phan_tich_tin_tuc()
    instruction_generator.qa_document()