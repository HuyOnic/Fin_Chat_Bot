from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import pandas as pd
from time import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

from app.db.qdrant import get_similar_vectors
from app.db.postgre import fetch_news_by_ids
from app.llm.prompt_builder import build_prompt, basic_system_prompt
from app.retriever.hybrid_retriever import HybridRetriever
from app.retriever.dense_retriever import DenseRetriever
from app.retriever.sparse_retriever import SparseRetriever
from app.db.postgre import fetch_newest_info
from app.db.qdrant import client
from app.llm.router_agent import RouterAgent
from app.llm.prompts import ANSWER_FINANCIAL_QUESTION_FROM_CONTEXT_PROMPT, ANSWER_FINANCIAL_QUESTION_PROMPT
from app.utils.vectorizer import convert_to_vector
from app.utils.sentimenizer import sentiment_analysis
from app.utils.sector_keywords import sector_keywords
from app.utils.chunking import extract_sector_sentences
load_dotenv()
print("OpenAI client:", os.getenv("OPENAI_API_BASE_URL"), "Key:", os.getenv("OPEN_API_KEY"))
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_NAME"),
    base_url=os.getenv("OPENAI_API_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.getenv("OPEN_API_KEY"),
    temperature=0.7
)
agent = RouterAgent(intent_tokenizer_ckt="models/phobert_tokenizer",
                    intent_model_ckt="models/intent_classifier",
                    entity_tokenizer_ckt="models/vit5_tokenizer",
                    entity_model_ckt="models/entity_classifier",
                    label_list_path="models/label_list_Balanced_Questions_Dataset.yaml",
                    api_call_lis_path="models/api_call_list.yaml",
                    use_onnx=False)
print("✅ Loaded Router Agent")
secCd_df = pd.read_csv("/home/goline/huy/quant_chat_bot/LLM_Project/data/stockcode_data/doanh_nghiep.csv")
print("✅ Loaded stock code knowledge")
# def answer_with_hybrid_rag(question: str):
#     pg_session = SessionLocal()
#     dense_retriever = DenseRetriever(client, bert_model)
#     sparse_retriever = SparseRetriever(pg_session)
#     hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)    
    
#     retrieved_docs = hybrid_retriever.retrieve(question, top_k=3)
#     prompt = build_fid_prompt(question, retrieved_docs)

    # return get_llm_response(prompt)

def ask_bot(question: str):
    start = time()
    vector = convert_to_vector(question)
    print("Convert vector time:", time()-start)

    start = time()
    results = get_similar_vectors(vector, top_k=3, threshold=0.70)
    ids = [r[0] for r in results]
    print("Get similar vector time:", time()-start)

    start = time()
    docs = fetch_news_by_ids(ids)
    print("Answer from document:",docs)
    print("Fetch news time:", time() - start)
    
    prompt = build_prompt(question, docs)

    start = time()
    # answer = get_llm_response(prompt)
    print("LLM answer time:", time() - start)
    return post_processing("")


def chat_bot(message: str) -> str:
    try:
        # 1. Chuyển đổi câu hỏi thành vector
        vector = convert_to_vector(message)
        if vector is None:
            raise ValueError("Không thể chuyển câu hỏi thành vector.")

        # 2. Tìm các vector tương tự
        similar_vectors = get_similar_vectors(vector, threshold=0.7) or []

        # 3. Chọn prompt phù hợp và context
        if not similar_vectors:
            prompt = ChatPromptTemplate.from_template(ANSWER_FINANCIAL_QUESTION_PROMPT)
            context = ""
        else:
            prompt = ChatPromptTemplate.from_template(ANSWER_FINANCIAL_QUESTION_FROM_CONTEXT_PROMPT)
            ids = [sv[0] for sv in similar_vectors]
            docs = fetch_news_by_ids(ids) or []
            context = "\n\n".join(doc.text for doc in docs)

        # 4. Tạo pipeline chain
        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # 5. Run chain
        response = rag_chain.invoke({"question": message, "context": context})
        return response

    except Exception as e:
        print(f"❌ Lỗi trong chat_bot: {e}")
        return "Hệ thống đang gặp lỗi, bạn vui lòng thử  lại sau"

    
def post_processing(answer):
    vietnamese_answer = answer.split("Translation:")[0]
    return vietnamese_answer

def route_fn(message: str):
    intent, secCd, contentType = agent.inference(message, "0", "0")
    try:
        prompt = sentiment_analysis_by_secCd(secCd.split(","))
    except Exception as e:
        print(e)
    return intent, secCd, contentType

def sentiment_news(message: str):
    intent, secCd, contentType = agent.inference(message, "0", "0")
    try:
        
        sentiment_prompt = ChatPromptTemplate.from_template(sentiment_analysis_by_secCd(secCd.split(",")))

        # 1. Chuyển đổi câu hỏi thành vector
        vector = convert_to_vector(message)
        if vector is None:
            raise ValueError("Không thể chuyển câu hỏi thành vector.")

        # 2. Tìm các vector tương tự
        similar_vectors = get_similar_vectors(vector, threshold=0.7) or []

        # 3. Chọn prompt phù hợp và context
        if not similar_vectors:
            prompt = ChatPromptTemplate.from_template(ANSWER_FINANCIAL_QUESTION_PROMPT)
            context = ""
        else:
            prompt = ChatPromptTemplate.from_template(ANSWER_FINANCIAL_QUESTION_FROM_CONTEXT_PROMPT)
            ids = [sv[0] for sv in similar_vectors]
            docs = fetch_news_by_ids(ids) or []
            context = "\n\n".join(doc.text for doc in docs)

    

        # 4. Tạo pipeline chain
        rag_chain = (
            # {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            prompt
            | llm
            | StrOutputParser()
        )

        # 5. Run chain
        response = rag_chain.invoke({"question": message, "context": sentiment_prompt})
        return response

    except Exception as e:
        print(e)

def sentiment_analysis_by_secCd(secCds: list):
    for secCd in secCds:        
        nhom_nganh = secCd_df[secCd_df['maDN']==secCd].loc[:, 'nhomNganh'].values[0].split("; ")[-1].replace("Nhom nganh", "")
        yesterday_timestamp = (datetime.now() - timedelta(days=20)).replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H%M")
        start = time()
        news = fetch_newest_info(int(yesterday_timestamp)) #lấy tin tức từ ngày hôm qua
        print("fetch_news:",time()-start)

        start = time()
        extracted_sector_sentences = extract_sector_sentences(news, sector_keywords)
        print("extract sector:", time()-start)

        selected_sentence = extracted_sector_sentences.get(nhom_nganh, None)
        
        if len(selected_sentence):
            sentences = selected_sentence["sentence"]
            sources = selected_sentence["source"]

            start = time()
            scores = sentiment_analysis(sentences)
            print('Sentiment analysis', time()-start)

            # BUILD PROMPS
            prompt = f"Theo phân tích dựa trên những tin tức gần đây nhất cho mã {secCd}"
            for i in range(len(scores)):
                score = scores[i].item()
                prompt+=f"\n📢 Nguồn tin {sources[i]} Sentiment Score:{score}"
                if score >= 0.5:
                    prompt+="\n=> Tin tức tích cực, giá của mã sẽ tăng"
                elif score <= -0.5:
                    prompt+="\n=> Tin tức tiêu cực, giá của mã sẽ giảm"
                else:
                    prompt+="\n=> Tin tức trung lập"
            return prompt

        else:
            print(f"Không tìm thấy dữ liệu tin tức mới nhất về ngành {nhom_nganh}")




if __name__=="__main__":
    secCds = ["ACB"]
    sentiment_analysis_by_secCd(secCds)









