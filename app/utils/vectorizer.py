import os
from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# MODEL_DIR = os.getenv("MODEL_DIR", "./models/sentence_transformers")
MODEL_DIR = "BAAI/bge-m3"
print(f"Đang load model từ: {MODEL_DIR}")

try:
    bert_model = SentenceTransformer(MODEL_DIR, device="cuda:1")
    print("Model đã được load thành công!")
except Exception as e:
    print(f"Lỗi khi load model: {e}")
    raise

def convert_to_vector(content_list):
    vectors = bert_model.encode(content_list, max_length=1024)
    return vectors

if __name__=="__main__":
    sentence1 = ["Giá cổ phiếu ACB"]
    sentence2 = ["Giá cổ phiếu ACB hôm nay là 982",
                "Giá cổ phiếu ACB (Ngân hàng Thương mại Cổ phần Á Châu) thời gian gần đây chịu ảnh hưởng bởi nhiều yếu tố kinh tế vĩ mô và tình hình ngành ngân hàng. Trong bối cảnh lãi suất có xu hướng giảm nhẹ và chính sách tiền tệ được nới lỏng, ACB được kỳ vọng có thể tăng trưởng tín dụng ổn định, hỗ trợ giá cổ phiếu. Tuy nhiên, áp lực cạnh tranh từ các ngân hàng lớn như Vietcombank hay BIDV cũng khiến thị giá ACB có thể biến động trong ngắn hạn. Các nhà đầu tư đang theo dõi sát sao báo cáo tài chính quý và chiến lược kinh doanh của ngân hàng để đưa ra quyết định phù hợp.",
                  "Theo phân tích kỹ thuật, giá cổ phiếu ACB hiện đang dao động trong vùng giá 20.000 - 25.000 đồng/cp, với thanh khoản khá tốt trên sàn HOSE. Một số chuyên gia nhận định nếu ACB duy trì được tỷ lệ NIM (biên lãi ròng) ổn định và kiểm soát tốt nợ xấu, giá cổ phiếu có thể phá vỡ ngưỡng kháng cự 25.000 đồng trong trung hạn. Tuy nhiên, rủi ro từ lạm phát hoặc biến động tỷ giá cũng có thể gây áp lực giảm giá. Nhà đầu tư nên cân nhắc yếu tố cơ bản và tâm lý thị trường trước khi giao dịch."]
    embedding1 = bert_model.encode(sentence1, batch_size=12, max_length=512)

    embedding2 = bert_model.encode(sentence2, batch_size=12, max_length=512)
    similarity = cosine_similarity(embedding1, embedding2)    
    print(similarity)


