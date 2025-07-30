from fastapi import FastAPI
from dotenv import load_dotenv
from pathlib import Path
import os, sys

from app.llm.router_agent import RouterAgent
from sentence_transformers import SentenceTransformer
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

app = FastAPI()
router_agent = RouterAgent(intent_tokenizer_ckt="models/phobert_tokenizer",
                    intent_model_ckt="models/intent_classifier",
                    entity_tokenizer_ckt="models/vit5_tokenizer",
                    entity_model_ckt="models/entity_classifier",
                    label_list_path="models/label_list_Balanced_Questions_Dataset.yaml",
                    api_call_lis_path="models/api_call_list.yaml",
                    use_onnx=False)

embed_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL"), device="cpu")

@app.get("/")
def read_root():
    return {"message": "Model Service"}

@app.post("router_agent/predict")
def router_agent_predict(message: str):
    intent, secCd, contentType = router_agent.inference(message, "1", "1")
    return {
        "intent":intent,
        "secCd":secCd,
        "contentType":contentType
    }

@app.post("embedding")
def embedding(text: str):
    return embed_model.encode(text)
