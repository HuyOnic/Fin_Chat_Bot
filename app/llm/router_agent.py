from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, pipeline
from langchain_community.llms import HuggingFacePipeline
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForSeq2SeqLM
import torch
from time import time
import os, sys
sys.path.append(os.path.join(os.path.abspath(__file__),".."))
from time import time
import pandas as pd
import re
import yaml
import json
from typing import List
import numpy as np

class RouterAgent():
    def  __init__(self, 
                  intent_tokenizer_ckt:str="tokenizers_checkpoint/phobert_tokenizer", 
                  intent_model_ckt:str="router_agent/results/checkpoint-5700",
                  #intent_model_ckt:str="router_agent/results/intent_onnx",  
                  entity_tokenizer_ckt:str="tokenizers_checkpoint/vit5_tokenizer", 
                  entity_model_ckt:str="router_agent/results/checkpoint-2000",
                  #entity_model_ckt:str="router_agent/results/entity_onnx",
                  label_list_path:str="router_agent/data/label_list_Balanced_Questions_Dataset.yaml",
                  api_call_lis_path:str="router_agent/data/api_call_list.yaml",
                  max_length:int=128,
                  use_onnx:bool=False):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if use_onnx:
            pass
        # if self.device == "cpu":
        #     core = Core()
            # Load OpenVINO models
            # self.intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_ckt).to(self.device)
            # self.intent_model = core.compile_model("router_agent/results/openvino_intent/model.xml", "CPU")
            # self.encoder_model = core.compile_model("router_agent/results/openvino_entity/encoder/encoder_model.xml", "CPU")
            # self.decoder_model = core.compile_model("router_agent/results/openvino_entity/decoder_with_past/decoder_with_past_model.xml", "CPU")
            # # Get I/O names
            # self.intent_input = self.intent_model.input(0)
            # self.intent_output = self.intent_model.output(0)

            # self.encoder_input = self.encoder_model.input(0)
            # self.encoder_output = self.encoder_model.output(0)

            # self.decoder_inputs = self.decoder_model.inputs
            # self.decoder_output = self.decoder_model.output(0)

        else:
            self.intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_ckt).to(self.device)
            self.entity_model = AutoModelForSeq2SeqLM.from_pretrained(entity_model_ckt).to(self.device)

        self.intent_tokenizer = AutoTokenizer.from_pretrained(intent_tokenizer_ckt)
        self.entity_tokenizer = AutoTokenizer.from_pretrained(entity_tokenizer_ckt)
        self.entity_tokenizer.add_tokens(["[", "]", "(", ")"])
        self.max_length=max_length
        self.use_onnx=use_onnx

        self.plat_codes = {"VNINDEX": "100", 
                           "HNX": "200",
                           "HSX": "200",
                           "HOSE": "200",
                           "UPCOMP": "300",
                           "UPCOM": "300"}
        
        self.label_list = yaml.safe_load(open(label_list_path, "r"))
        self.apli_call_list = yaml.safe_load(open(api_call_lis_path, "r"))
        # Not generate
        self.not_gen = ["account_info", "financial_valuation", "investment_efficiency", 
                        "margin_account_status", "market_assessment", "organization_info",
                        "outperform_stock", "request_financial_info", "sect_news", 
                        "stock_price", "top_index_contribution", "top_sec_index"]
        self.not_gen_indices = set([self.label_list.index(item) for item in self.not_gen])

        self.not_api = [self.label_list.index("margin_account_status")]

        # self.not_api_index = set([self.label_list.index(item) for item in self.not_api])
        self.compare_secirity_idx = self.label_list.index("compare_securities")
        self.sec_news_idx = self.label_list.index("sect_news")
        self.questions_of_document = self.label_list.index("questions_of_document")
    
    def run(self, question, session_id, user_id):
        assert isinstance(question, str), "input question should be a string"
        if not self.use_onnx:
            self.intent_model.eval()
            self.entity_model.eval()
        with torch.no_grad():
            # Intent classification
            start = time()

            intent_inputs = self.intent_tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            intent_inputs = {k: v.to(self.device) for k, v in intent_inputs.items()}  # Chuyển input sang GPU
            intent_outputs = self.intent_model(**intent_inputs)
            predicted_class_idx = torch.argmax(intent_outputs.logits, dim=-1)
            intent = self.label_list[predicted_class_idx]

            entity_inputs = self.entity_tokenizer(questions,  return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
            entity_output_ids = self.entity_model.generate(**entity_inputs)
            entity_decoded = self.entity_tokenizer.batch_decode(entity_output_ids, skip_special_tokens=True)
            contentType, secCd, top = self.entity_post_processing(question, entity_decoded)
            print("Run time:", time()-start)

            if predicted_class_idx==self.sec_news_idx or predicted_class_idx==self.questions_of_document:
                contentType=None
            if secCd=="":
                secCd=None
            if contentType=="":
                contentType=None
            
            if predicted_class_idx == self.compare_secirity_idx:
                contentType="indexSection"
            elif intent=="request_financial_info":
                contentType="financingSection"
            elif intent=="stock_insight":
                contentType="stock_insight"
            elif intent=="financial_analysis":
                contentType="ALL"
            elif intent=="outperform_stock":
                secCd = None
        # return intent, secCd, contentType
        return self.output_parser(intent=intent, 
                                 question=question, 
                                 contentType=contentType, 
                                 secCd=secCd, 
                                 session_id=session_id, 
                                 user_id=user_id)
    
    def inference(self, question, session_id, user_id):
        assert isinstance(question, str), "input question should be a string"
        if not self.use_onnx:
            self.intent_model.eval()
            self.entity_model.eval()
        with torch.no_grad():
            start = time()

            # Intent
            intent_inputs = self.intent_tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
            intent_outputs = self.intent_model(**intent_inputs)
            predicted_class_idx = torch.argmax(intent_outputs.logits, dim=-1).item()
            intent = self.label_list[predicted_class_idx]

            # Entity
            entity_inputs = self.entity_tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
            entity_output_ids = self.entity_model.generate(**entity_inputs, max_new_tokens=self.max_length)
            entity_decoded = self.entity_tokenizer.decode(entity_output_ids[0], skip_special_tokens=True, num_return_sequences=1)
            print("ENT decoded:", entity_decoded)
            contentType, secCd, _ = self.entity_post_processing(question, entity_decoded)
            print("Run time:", time()-start)

            if predicted_class_idx == self.sec_news_idx or predicted_class_idx == self.questions_of_document:
                contentType = None
            if secCd == "":
                secCd = None
            if contentType == "":
                contentType = None

            if predicted_class_idx == self.compare_secirity_idx:
                contentType = "indexSection"
            elif intent == "request_financial_info":
                contentType = "financingSection"
            elif intent == "stock_insight":
                contentType = "stock_insight"
            elif intent == "financial_analysis":
                contentType = "ALL"
            elif intent == "outperform_stock":
                secCd = None

        return intent, secCd, contentType

    def run_batch(self, questions: List[str], session_ids: List[str], user_ids: List[str]):
        start = time()
        results = []
        if not self.use_onnx:
            self.intent_model.eval()
            self.entity_model.eval()

        with torch.no_grad():
            infer_start=time()
            # Tokenize và dự đoán intent cho batch
            intent_inputs = self.intent_tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
            intent_outputs = self.intent_model(**intent_inputs)
            predicted_class_indices = torch.argmax(intent_outputs.logits, dim=-1)
            print("Intent time", time()-infer_start)
            entity_inputs = self.entity_tokenizer(questions,  return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
            entity_output_ids = self.entity_model.generate(**entity_inputs)
            entity_decoded = self.entity_tokenizer.batch_decode(entity_output_ids, skip_special_tokens=True)
            # entity_decoded = [self.generate_openvino_seq2seq(q, max_length=self.max_length) for q in questions]
            print("Inference time:", time()-infer_start)

            # Duyệt từng item trong batch để xử lý riêng lẻ
            for i in range(len(questions)):
                question = questions[i]
                session_id = session_ids[i]
                user_id = user_ids[i]

                intent_idx = predicted_class_indices[i].item()
                intent = self.label_list[intent_idx]
                contentType, secCd, _ = self.entity_post_processing(question, entity_decoded[i])

                # Các rule giống hàm `run()`
                if intent_idx == self.sec_news_idx or intent_idx == self.questions_of_document or contentType=="":
                    contentType = None
                if secCd == "":
                    secCd = None
                if intent_idx == self.compare_secirity_idx:
                    contentType = "indexSection"
                elif intent == "request_financial_info":
                    contentType = "financingSection"
                elif intent == "stock_insight":
                    contentType = "stock_insight"
                elif intent == "financial_analysis":
                    contentType = "ALL"
                elif intent == "outperform_stock":
                    secCd = None

                output = self.output_parser(intent, question, contentType, secCd, session_id, user_id)
                results.append(output)
            
        print("Total run time:", time() - start)
        return results
    
    def generate_openvino_seq2seq(self, input_text: str, max_length: int = 64) -> str:
        # Encode input
        input_ids = self.entity_tokenizer(input_text, return_tensors="np").input_ids
        encoder_hidden_states = self.encoder_model([input_ids])[self.encoder_output]

        # Init decoder input
        decoder_input_ids = np.array([[self.entity_tokenizer.pad_token_id]], dtype=np.int32)
        past_key_values = [np.zeros(inp.shape, dtype=np.float32) for inp in self.decoder_inputs[2:]]

        generated_ids = []

        for _ in range(max_length):
            inputs = {
                self.decoder_inputs[0].get_any_name(): decoder_input_ids,
                self.decoder_inputs[1].get_any_name(): encoder_hidden_states
            }
            for i, past in enumerate(past_key_values):
                inputs[self.decoder_inputs[i + 2].get_any_name()] = past

            outputs = self.decoder_model(inputs)
            logits = outputs[self.decoder_output]
            next_token_id = np.argmax(logits, axis=-1).astype(np.int32)

            if next_token_id.item() == self.entity_tokenizer.eos_token_id:
                break

            generated_ids.append(next_token_id.item())
            decoder_input_ids = next_token_id.reshape(1, 1)

            past_key_values = []
            for inp in self.decoder_inputs[2:]:
                shape = [d if isinstance(d, int) else 1 for d in inp.get_partial_shape()]
                past_key_values.append(np.zeros(shape, dtype=np.float32))

        return self.entity_tokenizer.decode(generated_ids, skip_special_tokens=True)


    def entity_post_processing(self, question, raw_text):
        content_types = re.findall(r'\((.*?)\)', raw_text)
        not_content_types = ['100', '200', '300']
        stock_codes = list(map(lambda x: x.replace(" ","").upper(), re.findall(r'\[(.*?)\]', raw_text)))
        for content_type in content_types:
            if content_type.strip() in not_content_types:
                stock_codes.append(content_type.strip())
        content_types = [c.strip() for c in content_types if c.strip() not in not_content_types]
        # if len(stock_codes)==0:
        #     stock_codes = re.findall(r'\b[A-Z]{3,5}\b', question)
        if ("UPCOMP" in raw_text) or ("UPCOM" in raw_text):
            stock_codes.append("UPCOMP")
        not_stock_codes = ["HSX", "VN", "ATO", "OTP", "ALL"]
        stock_codes = [stock_code for stock_code in stock_codes if stock_code not in not_stock_codes]
        return ",".join(content_types).replace(" ","").replace("ROE","roe"), ",".join(stock_codes), len(stock_codes)*3
    
    def output_parser(self, intent, question: str, contentType: str, secCd: str, session_id: str, user_id: str):
        if secCd is None:
            top = 0
            secCd = ""
        else:
            secCd = secCd.split(",")
            secCd = list(set([cd if cd not in self.plat_codes.keys() else self.plat_codes[cd] for cd in secCd]))
            secCd = ",".join(secCd)
            top = len(secCd)*3
        dict_template = {
            "compare_FA": 
            {
                "intent": intent,
                "parameters": {
                    "query": "",
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [json.loads(json.dumps({"query": question, "top": top}))],
                        "api_calls": [f"financial-information:{{\"duration\": \"{top//3}\", \"language\": \"VI\", \"secCd\": \"{secCd}\", \"contentType\": \"{contentType}\"}}"]
                    },
                    "generation": True,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id,
                "response_target": "user"
            },
            "request_financial_info": 
            {
                "intent": intent,
                "parameters": {
                    "query": "",
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [json.loads(json.dumps({"query": question, "top": top}))],
                        "api_calls": [
                            f"financial-information:{{\"duration\": \"{top//3}\", \"language\": \"VI\", \"secCd\":\"{secCd}\", \"contentType\": \"{contentType}\"}}"
                        ]
                    },
                    "generation": False,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id,
                "response_target": "user"
            },
            "compare_securities": 
            {
                "intent": intent,
                "parameters": {
                    "query": [{"query": question, "top": top}],
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [json.loads(json.dumps({"query": question, "top": top}))],
                        "api_calls":  [f"financial-information:{{\"contentType\": \"{contentType}\", \"duration\": \"{top//3}\", \"language\": \"VI\", \"secCd\": \"{secCd}\"}}", 
                                       f"technical-price-list:{{\"contentType\": \"{contentType}\", \"period\": \"1M\", \"language\": \"VI\", \"secList\": \"{secCd}\"}}"]
                    },
                    "generation": True,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id,
                "response_target": "user"
                },

            "stock_insight": {
                "intent": "stock_insight",
                "parameters": {
                    "query": "",
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [json.loads(json.dumps({"query": question, "top": top}))],
                        "api_calls": [
                            f"financial-analysis:{{\"contentType\": \"stock_insight\", \"duration\": \"1\", \"language\": \"VI\", \"secCd\": \"{secCd}\"}}", 
                            f"technical-price-list:{{\"contentType\": \"financial-analysis-stock_insight\", \"period\": \"1M\", \"language\": \"VI\", \"secList\": \"{secCd}\"}}",
                            f"sect-news:{{\"language\":\"VI\",\"page\":\"0\",\"pageSize\":\"3\",\"secCd\":\"{secCd}\"}}"
                        ]
                    },
                    "generation": True,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id
            },

            "margin_account_status": {
                "intent": "margin_account_status",
                "parameters": {
                    "query": "",
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [],
                        "api_calls": [f"displayMarginAccountStatus:{{\"language\":\"VI\"}}"]
                    },
                    "generation": False,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id
            },

            "questions_of_document": {
                "intent": "questions_of_document",
                "parameters": {
                    "query": "",
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [json.loads(json.dumps({"query": question, "top": top}))],
                        "api_calls": []
                    },
                    "generation": True,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id
            },

            "stock_price": 
            {
                "intent": intent,
                "parameters": {
                    "query": "",
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [],
                        "api_calls": [f"mrktsec-quotes-detail:{{\"language\": \"VI\", \"secCd\": \"{secCd}\", \"contentType\": \"{contentType}\"}}"]
                    },
                    "generation": False,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id,
                "response_target": "user"
            },

            "market_assessment": 
            {
                "intent": intent,
                "parameters": {
                    "query": "",
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [],
                        "api_calls": [f"market-assessment:{{\"language\": \"VI\", \"indexCdList\": \"{secCd}\", \"contentType\": \"{contentType}\"}}"]
                    },
                    "generation": False,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id,
                "response_target": "user"
            },

            "sect_news": 
            {
                "intent": intent,
                "parameters": {
                    "query": "",
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [],
                        "api_calls":[f"sect-news:{{\"language\":\"VI\",\"page\":\"0\",\"pageSize\":\"3\",\"secCd\":\"{secCd}\"}}"]
                    },
                    "generation": False,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id,
                "response_target": "user"
            },

            "top-index-contribution": 
            {
                "intent": intent,
                "parameters": {
                    "query": "",
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [],
                        "api_calls": [
                            f"top-index-contribution:{{\"language\": \"VI\", \"indexCdList\": \"{secCd}\", \"contentType\": \"{contentType}\"}}"
                        ]
                    },
                    "generation": False,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id,
                "response_target": "user"
            },

            "flashdeal_recommend": 
            {
                "intent": intent,
                "parameters": {
                    "query": "",
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [],
                        "api_calls": [
                            f"flashdeal_recommend:{{\"language\": \"VI\", \"page\":\"0\", \"pageSize\":\"10\", \"indexCdList\": \"{secCd}\", \"contentType\": \"{contentType}\"}}"
                        ]
                    },
                    "generation": False,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id,
                "response_target": "user"
            },

            "organization_info": 
            {
                "intent": intent,
                "parameters": {
                    "query": "",
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [],
                        "api_calls": [
                            f"organization_info:{{\"language\": \"VI\", \"secCd\": \"{secCd}\", \"contentType\": \"{contentType}\"}}"
                        ]
                    },
                    "generation": False,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id,
                "response_target": "user"
            },

            "top_sec_index": 
            {
                "intent": intent,
                "parameters": {
                    "query": "",
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [],
                        "api_calls": [
                            f"top_sec_index:{{\"top\":\"5\", \"language\": \"VI\", \"secCd\": \"{secCd}\", \"contentType\": \"{contentType}\"}}"
                        ]
                    },
                    "generation": False,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id,
                "response_target": "user"
            },        

            "outperform_stock": 
            {
                "intent": intent,
                "parameters": {
                    "query": "",
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [],
                        "api_calls": [
                            f"outperform-stock:{{\"language\": \"VI\", \"contentType\": \"{contentType}\"}}"
                        ]
                    },
                    "generation": False,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id,
                "response_target": "user"
            },   

            "financial_valuation": 
            {
                "intent": intent,
                "parameters": {
                    "query": "",
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [],
                        "api_calls": [f"financial-valuation:{{\"contentType\": \"{contentType}\", \"duration\": \"{top//3}\", \"language\": \"VI\", \"secCd\": \"{secCd}\"}}"]
                    },
                    "generation": False,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id,
                "response_target": "user"
            }, 

            "technical_analysis": 
            {
                "intent": intent,
                "parameters": {
                    "query": "",
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [],
                        "api_calls": [f'technical-price-list:{{\"period\": \"1M\", \"language\": \"VI\", \"typeList\": \"1,2\", \"secList\": \"{secCd}\", \"contentType\": \"{contentType}\"}}']
                    },
                    "generation": True,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id,
                "response_target": "user"
            },

            "financial_analysis": 
            {
                "intent": intent,
                "parameters": {
                    "query": "",
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [json.loads(json.dumps({"query": question, "top": top}))],
                        "api_calls":[f"financial-analysis:{{\"contentType\": \"{contentType}\", \"duration\": \"{top//3}\", \"language\": \"VI\", \"secCd\": \"{secCd}\"}}"]
                    },
                    "generation": True,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id,
                "response_target": "user"
            },

            "account_info": 
            {
                "intent": intent,
                "parameters": {
                    "query": "",
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [],
                        "api_calls":[f"displayAccountInfo:{{\"fromDate\": \"USER_FROM_DATE\", \"toDate\": \"USER_TO_DATE\", \"custNo\": \"USER_CUST_NO\", \"language\": \"VI\", \"contentType\": \"{contentType}\"}}"]
                    },
                    "generation": True,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id,
                "response_target": "user"
            },

            "investment_efficiency": 
            {
                "intent": intent,
                "parameters": {
                    "query": "",
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [],
                        "api_calls": [f"displayInvestmentEfficiency:{{\"fromDate\": \"USER_FROM_DATE\", \"toDate\": \"USER_TO_DATE\", \"custNo\": \"USER_CUST_NO\", \"language\": \"VI\", \"contentType\": \"{contentType}\"}}"]
                    },
                    "generation": False,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id,
                "response_target": "user"
            },

            "margin_account_status": 
            {
                "intent": intent,
                "parameters": {
                    "query": "",
                    "symbols": [],
                    "sector": None
                },
                "actions": {
                    "retrieval": {
                        "rag": [],
                        "api_calls":  ["displayMarginAccountStatus:{\"language\":\"VI\"}"]
                    },
                    "generation": False,
                    "raw_input": question
                },
                "session_id": session_id,
                "user_id": user_id,
                "response_target": "user"
            }
        }
        return json.dumps(dict_template[intent], ensure_ascii=False)
    
    def extract_content(self, raw_label):
        data = json.loads(raw_label)
        intent = data['intent']
        if intent=="technical_analysis":
            api_call_str = data['actions']['retrieval']['api_calls'][0] 
            json_part = api_call_str.split(':', 1)[1]
            api_call_data = json.loads(json_part)
            secCd = api_call_data['secList'] if "secList" in list(api_call_data.keys()) else None
            contentType = api_call_data['contentType'] if "contentType" in list(api_call_data.keys()) else None
            return intent, secCd, contentType 

        if len(data['actions']['retrieval']['api_calls']):
            api_call_str = data['actions']['retrieval']['api_calls'][0] 
            json_part = api_call_str.split(':', 1)[1]
            api_call_data = json.loads(json_part)
            secCd = api_call_data['secCd'] if "secCd" in list(api_call_data.keys()) else None
            contentType = api_call_data['contentType'] if "contentType" in list(api_call_data.keys()) else None
            return intent, secCd, contentType 
        return intent, None, None
    
def evaluate(agent, csv_path: str = "router_agent/data/Output_test.xlsx"):
    df = pd.read_excel(csv_path)
    df.dropna(inplace=True)
    questions = df['question'].tolist()
    labels = df['groundTruth'].tolist()
    intent_score = 0
    secCd_score = 0
    contentType_score = 0
    for question, label in zip(questions, labels):
        intent_label, secCd_label, contentType_label = agent.extract_content(label)
        intent_pred, secCd_pred, contentType_pred = agent.run(question)
        question = question.lower()
        if secCd_pred != secCd_label or contentType_pred!=contentType_label:
            print(question)
            # print("=>intent", intent_pred)
            # print("code",secCd_pred, secCd_label)
            print("content",contentType_pred, contentType_label)
        # print("pred:",contentType_pred,"label:",contentType_label)
        intent_score += 1 if intent_label==intent_pred else 0
        secCd_score += 1 if secCd_label==secCd_pred else 0
        contentType_score += 1 if contentType_label==contentType_pred else 0

    print(f"Intent Accuracy: {intent_score/len(questions)*100:.2f}%")
    print(f"secCd Accuracy: {secCd_score/len(questions)*100:.2f}%")
    print(f"contentType Accuracy: {contentType_score/len(questions)*100:.2f}%")

if __name__=="__main__":
    batch_size = 1000
    texts = ["cơ hội đầu tư cổ phiếu ACB" for _ in range(batch_size)]
    session_ids = ["0" for _ in range(batch_size)]
    user_ids = ["0" for _ in range(batch_size)]

    agent = RouterAgent()
    out = agent.run_batch(texts, session_ids, user_ids)
    exit()
    df = pd.read_excel("router_agent/data/Output_test.xlsx")
    df.dropna(inplace=True)
    questions = df['question'].tolist()
    labels = df['groundTruth'].tolist()
    intent_score = 0
    secCd_score = 0
    contentType_score = 0
    for question, label in zip(questions, labels):
        question = question.lower()
        intent_label, secCd_label, contentType_label = agent.extract_content(label)
        intent_pred, secCd_pred, contentType_pred = agent.run(question, "0", "1")
        if contentType_pred and contentType_label:
            contentType_pred = set(contentType_pred.split(","))
            contentType_label = set(contentType_label.split(","))
        if secCd_pred != secCd_label or contentType_pred!=contentType_label:
            print("=>question:", question)
            print("predicted_intent", intent_pred)
            print("predicted secCd:",secCd_pred, "actual secCd:", secCd_label)
            print("predicted content",contentType_pred, "actual content", contentType_label)
        # print("pred:",contentType_pred,"label:",contentType_label)
        intent_score += 1 if intent_label==intent_pred else 0
        secCd_score += 1 if secCd_label==secCd_pred else 0
        contentType_score += 1 if contentType_label==contentType_pred else 0

    print(f"Intent Accuracy: {intent_score/len(questions)*100:.2f}%")
    print(f"secCd Accuracy: {secCd_score/len(questions)*100:.2f}%")
    print(f"contentType Accuracy: {contentType_score/len(questions)*100:.2f}%")

