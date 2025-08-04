from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import initialize_agent, AgentType
from langgraph.graph import StateGraph, END
import pandas as pd
import numpy as np
from time import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os, json, re

from app.db.qdrant import get_documents_by_vector
from app.llm.prompt_builder import build_prompt, basic_system_prompt
from app.retriever.hybrid_retriever import HybridRetriever
from app.retriever.dense_retriever import DenseRetriever
from app.retriever.sparse_retriever import SparseRetriever
from app.db.postgre import fetch_newest_info
from app.db.qdrant import client
from app.llm.router_agent import RouterAgent
from app.llm.prompts import *
from app.utils.vectorizer import convert_to_vector
from app.utils.sentimenizer import sentiment_analysis
from app.utils.sector_keywords import sector_keywords
from app.utils.chunking import extract_sector_sentences, split_sentences
from app.llm.tools import *


load_dotenv()
print("OpenAI client:", os.getenv("OPENAI_API_BASE_URL"), "Key:", os.getenv("OPEN_API_KEY"))

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_NAME"),
    base_url=os.getenv("OPENAI_API_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.getenv("OPEN_API_KEY"),
    temperature=0
)
mrktsec_quotes_detail_tool = mrktsec_quotes_detail_tool()
technical_price_list_tool = technical_price_list_tool()

ta_agent = initialize_agent(
    tools=[mrktsec_quotes_detail_tool, technical_price_list_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)
agent = RouterAgent(intent_tokenizer_ckt="models/phobert_tokenizer",
                    intent_model_ckt="models/intent_classifier",
                    entity_tokenizer_ckt="models/vit5_tokenizer",
                    entity_model_ckt="models/entity_classifier",
                    label_list_path="models/label_list_Balanced_Questions_Dataset.yaml",
                    api_call_lis_path="models/api_call_list.yaml",
                    use_onnx=False)

print("✅ Loaded Router Agent")
secCd_df = pd.read_csv("data/stockcode_data/doanh_nghiep.csv")
vn30_codes = pd.read_csv("data/vn30_weights.csv")
vn30_df = pd.merge(vn30_codes, secCd_df, how="left", on="maDN")
print("✅ Loaded stock code knowledge")

market_api_token = os.getenv("MARKET_API_TOKEN")
language = os.getenv("LANGUAGE")

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
    docs = get_documents_by_vector(vector, top_k=3, threshold=0.1)
    print("Answer from document:",docs)
    print("Fetch news time:", time() - start)
    
    prompt = build_prompt(question, docs)

    start = time()
    # answer = get_llm_response(prompt)
    print("LLM answer time:", time() - start)
    return post_processing("")

def retrieve(message):
    try:
        # 1. Chuyển đổi câu hỏi thành vector
        vector = convert_to_vector(message)
        if vector is None:
            raise ValueError("Không thể chuyển câu hỏi thành vector.")

        # 2. Truy vấn nội dung trong Qdrant
        docs = get_documents_by_vector(vector, top_k=3, threshold=0.1)

        # 3. Chọn prompt phù hợp và context
        if not docs:
            context = "Không truy vấn được nội dung cần tìm"
        else:
            context = "\n\n".join(doc["content"] for doc in docs)
        return context
    except Exception as e:
        print(e)

def chat_bot(message: str) -> str:
    try:
        # 1. Chuyển đổi câu hỏi thành vector
        vector = convert_to_vector(message)
        if vector is None:
            raise ValueError("Không thể chuyển câu hỏi thành vector.")

        # 2. Truy vấn nội dung trong Qdrant
        docs = get_documents_by_vector(vector, top_k=3, threshold=0.1)

        print("Tìm được:", len(docs))
        # 3. Chọn prompt phù hợp và context
        if not docs:
            prompt = ChatPromptTemplate.from_template(ANSWER_FINANCIAL_QUESTION_PROMPT)
            context = ""
        else:
            prompt = ChatPromptTemplate.from_template(ANSWER_FINANCIAL_QUESTION_FROM_CONTEXT_PROMPT)
            context = "\n\n".join(doc["content"] for doc in docs)

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

def chat_pipeline(message: str):
    intent, secCd, contentType = agent.inference(message, "0", "0")
    input = json.dumps({
        "question": message,
        "secCd": secCd,
        "contentType": contentType,
        "language": "VI",
        "jwt_token": os.getenv("MARKET_API_TOKEN")
    })
    output = ta_agent.run(input)
    print(output)

def rounting(message: str):
    try:
        if "sentiment" in message.lower():
            if "vn30" in message.lower():
                return sentiment_vn30f1m()
            else:
                response =  sentiment_news(message)
                return response
            
        intent, secCd, contentType = agent.inference(message, "0", "0")
        if intent=="account_info": # done
            context = get_display_account_info(contentType=contentType, language=language, jwt_token=market_api_token)
            prompt = ChatPromptTemplate.from_template(ANSWER_NOT_SUPPORT_PROMPT)
        
        elif intent=="compare_FA" or intent=="request_financial_info": #TA+FA #done
            context = get_financial_infomation(secCd=secCd, contentType=contentType, language=language, jwt_token=market_api_token)
            prompt = ChatPromptTemplate.from_template(ANSWER_TA_FA_QUESTION_FROM_CONTEXT_PROMPT)

        elif intent=="compare_securities" or intent=="technical_analysis": #TA+FA done
            # API financial_infomation bị lỗi, data: "---" và khi contentType=indexSection thì trả về {"status":"SUCCESS","arg":null,"data":{"statusCode":1,"errorCode":null,"message":null,"errorField":null,"data":null,"totalRecords":null}}
            # fi_res = get_financial_infomation(secCd=secCd, contentType=contentType, language=language, jwt_token=market_api_token)
            tp_res = get_technical_price_list(secCd=secCd, contentType="ALL", language=language, jwt_token=market_api_token) 
            context = f"""
            {tp_res}
            """
            prompt = ChatPromptTemplate.from_template(COMPARE_SECURITY_PROMPT)

        elif intent=="financial_analysis" or intent=="stock_insight": #TA+FA
            # API financial_infomation bị lỗi, data: "---"
            context = get_financial_analysis(secCd=secCd, contentType="indexSection", language=language, jwt_token=market_api_token)
            prompt = ChatPromptTemplate.from_template(ANSWER_TA_FA_QUESTION_FROM_CONTEXT_PROMPT)
        
        elif intent=="financial_valuation": #TA+FA
            # API financial_valuation bị lỗi '{"status":"SUCCESS","arg":null,"data":{"statusCode":1,"errorCode":null,"message":null,"errorField":null,"data":null,"totalRecords":null}}'
            context = get_financial_valuation(secCd=secCd, contentType=contentType, language=language, jwt_token=market_api_token)
            prompt = ChatPromptTemplate.from_template(ANSWER_TA_FA_QUESTION_FROM_CONTEXT_PROMPT)
        
        elif intent=="flashdeal_recommend": #TA+FA 
            #API flashdeal_recommend bị lỗi, trường rows trả về list rỗng
            context = get_flashdeal_recommend(marketCdList=secCd, contentType=contentType, language=language, jwt_token=market_api_token)  
            prompt = ChatPromptTemplate.from_template(ANSWER_TA_FA_QUESTION_FROM_CONTEXT_PROMPT)

        elif intent=="investment_efficiency": #Full Info done
            # chưa gọi được API investment_efficiency
            context = get_display_investment_efficiency(contentType=contentType, language=language, jwt_token=market_api_token)  
            prompt = ChatPromptTemplate.from_template(ANSWER_NOT_SUPPORT_PROMPT)

        elif intent=="margin_account_status": # done
            # chưa gọi được API margin_account_status
            context = get_margin_account_status(language=language, jwt_token=market_api_token)
            prompt = ChatPromptTemplate.from_template(ANSWER_NOT_SUPPORT_PROMPT)
        
        elif intent=="market_assessment": #TA+FA #done
            context = get_market_assessment(secCd=secCd, contentType=contentType, language=language, jwt_token=market_api_token)
            prompt = ChatPromptTemplate.from_template(ANSWER_TA_FA_QUESTION_FROM_CONTEXT_PROMPT)

        elif intent=="organization_info": #Full Info
            # API organization info bị lỗi, data:"---"
            context = get_organization_info(secCd=secCd, contentType=contentType, language=language, jwt_token=market_api_token)
            prompt = ChatPromptTemplate.from_template(ANSWER_TA_FA_QUESTION_FROM_CONTEXT_PROMPT)
        
        elif intent=="outperform_stock": #Full Info
            context = get_outperform_stock(contentType=contentType, language=language, jwt_token=market_api_token)
            prompt = ChatPromptTemplate.from_template(ANSWER_TA_FA_QUESTION_FROM_CONTEXT_PROMPT)
        
        elif intent=="questions_of_document":
            # Không cần API, trả lời bằng retrieve
            context= retrieve(message)
            prompt = ChatPromptTemplate.from_template(ANSWER_TA_FA_QUESTION_FROM_CONTEXT_PROMPT)

        elif intent=="sect_news": #Full Info
            # API sect_news bị lỗi {"status":"SUCCESS","arg":null,"data":{"statusCode":0,"errorCode":null,"message":null,"errorField":null,"data":[],"totalRecords":0}}
            context = get_sect_news(secCd=secCd, language=language, jwt_token=market_api_token)
            prompt = ChatPromptTemplate.from_template(ANSWER_TA_FA_QUESTION_FROM_CONTEXT_PROMPT)
        
        elif intent=="stock_price": #done
            context = get_mrktsec_quotes_detail(secCd=secCd, contentType=contentType, language=language, jwt_token=market_api_token)
            prompt = ChatPromptTemplate.from_template(STOCK_PRICE_PROMPT)
    
        elif intent=="top_index_contribution": #done
            # trường hợp hỏi 3 sàn, API chỉ trả về 1 sàn
            context = get_top_index_contribution(secCd=secCd, contentType=contentType, language=language, jwt_token=market_api_token)
            prompt = ChatPromptTemplate.from_template(ANSWER_TA_FA_QUESTION_FROM_CONTEXT_PROMPT)

        elif intent=="top_sec_index": #Full Info
            # API top_sec_index lỗi, trả về data: "---"
            context = get_top_sec_index(contentType=contentType, language=language, jwt_token=market_api_token)
            prompt = ChatPromptTemplate.from_template(ANSWER_TA_FA_QUESTION_FROM_CONTEXT_PROMPT)

        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        if str(context)=='{"status":"SUCCESS","arg":null}':
            response = "Xin lỗi Quý Khách, hiện tại chúng tôi không thể trả lời câu hỏi của Quý Khách."
        else:
            response = rag_chain.invoke({"question": message, "context": context})
        
        return response
    except Exception as e:
        print(e)

def sentiment_news(message: str):
    intent, secCd, contentType = agent.inference(message, "0", "0")
    try:
        sentiment_prompt = sentiment_analysis_by_secCd(secCd.split(","))
        return sentiment_prompt
        # 1. Chuyển đổi câu hỏi thành vector
        vector = convert_to_vector(message)
        if vector is None:
            raise ValueError("Không thể chuyển câu hỏi thành vector.")

        # 2. Truy vấn nội dung trong Qdrant
        docs = get_documents_by_vector(vector, top_k=3, threshold=0.1)

        # 3. Chọn prompt phù hợp và context
        if not docs:
            prompt = ChatPromptTemplate.from_template(ANSWER_FINANCIAL_QUESTION_PROMPT)
            context = ""
        else:
            prompt = ChatPromptTemplate.from_template(ANSWER_FINANCIAL_QUESTION_FROM_CONTEXT_PROMPT)
            context = "\n\n".join(doc["content"] for doc in docs)

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
        print("Lỗi khi chạy pipeline:", e)

def sentiment_analysis_by_secCd(secCds: list):
    for secCd in secCds:        
        nhom_nganh = secCd_df[secCd_df['maDN']==secCd].loc[:, 'nhomNganh'].values[0].split("; ")[-1].replace("Nhom nganh", "")
        yesterday_timestamp = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H%M")
        start = time()
        news = fetch_newest_info(int(yesterday_timestamp)) #lấy tin tức từ ngày hôm qua
        print("fetch_news:",time()-start)

        start = time()
        extracted_sector_sentences = extract_sector_sentences(news, sector_keywords)
        print("extract sector:", time()-start)

        selected_sentence = extracted_sector_sentences.get(nhom_nganh, None)
        print("Số sentence ảnh hưởng:", len(selected_sentence))
        if len(selected_sentence):
            sentences = selected_sentence["sentence"]
            sources = selected_sentence["source"]

            start = time()
            scores = sentiment_analysis(sentences)
            print("số scores: ",len(scores))
            # BUILD PROMPS
            prompt = f"Theo phân tích dựa trên những tin tức gần đây nhất cho mã {secCd}"
            for i in range(len(scores)):
                score = scores[i]
                prompt+=f"\n📢 Nguồn tin {sources[i]} Sentiment Score:{score}"
                if score >= 0.5:
                    prompt+="\n=> Tin tức tích cực, giá của mã sẽ tăng"
                elif score <= -0.5:
                    prompt+="\n=> Tin tức tiêu cực, giá của mã sẽ giảm"
                else:
                    prompt+="\n=> Tin tức trung lập"
            prompt += f"\nSentiment score trung bình cho mã {secCd} là: {sum(scores)/len(scores)}"
            return prompt

        else:
            print(f"Không tìm thấy dữ liệu tin tức mới nhất về ngành {nhom_nganh}")

def sentiment_vn30f1m():
    yesterday_timestamp = (datetime.now() - timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H%M")
    news = fetch_newest_info(yesterday_timestamp)
    print(f"Tìm thấy {len(news)} tin tức mới nhất. Đang tiến hành phân tích...")
    sentiment_results={}
    response = "Tính toán sentiment score của các mã dựa trên tin tức mới nhất:\n"
    for idx, row in vn30_df.iterrows():
        nhom_nganh = row["nhomNganh"].split("; ")[-1].replace("Nhom nganh", "")
        keywords = sector_keywords[nhom_nganh]
        impact_sentences = []
        for new in news:
            sentences = split_sentences(new["content"])
            impact_sentences.extend([sentence.lower() for sentence in sentences if any(keyword.lower() in sentence.lower() for keyword in keywords)])
        if len(impact_sentences):
            score = np.mean(vn30_df["Tỷ trọng"] * np.mean(sentiment_analysis(impact_sentences)))
        else:
            score = 0
        sentiment_results[row["maDN"]] = score
        response+=f'{row["maDN"]} : {score}\n'

    response += "=> Dựa trên những thông tin đã phân tích ở trên. Tôi đưa ra kết luận Sentiment Score VN30F1M hôm nay là:"+str(sum(sentiment_results.values()))
    return response

if __name__=="__main__":
    sentiment_vn30f1m()










