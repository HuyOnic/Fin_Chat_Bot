import os
import logging
import torch
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import uvicorn

# --------------------------
# Configuration Setup
# --------------------------
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --------------------------
# Model Loading
# --------------------------
class ModelLoader:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.load_all_models()

    def load_all_models(self):
        """Load all required models with error handling"""
        try:
            self._load_router_agent()
            self._load_embedding_model()
            self._load_sentiment_model()
            logger.info("✅ All models loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to load models: {str(e)}")
            raise

    def _load_router_agent(self):
        from app.llm.router_agent import RouterAgent  # Local import to avoid circular dependency
        self.models["router_agent"] = RouterAgent(
            intent_tokenizer_ckt="models/phobert_tokenizer",
            intent_model_ckt="models/intent_classifier",
            entity_tokenizer_ckt="models/vit5_tokenizer",
            entity_model_ckt="models/entity_classifier",
            label_list_path="models/label_list_Balanced_Questions_Dataset.yaml",
            api_call_lis_path="models/api_call_list.yaml",
            use_onnx=False
        )

    def _load_embedding_model(self):
        model_name = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
        self.models["embed_model"] = SentenceTransformer(model_name).to(self.device)

    def _load_sentiment_model(self):
        model_id = os.getenv("SENTIMENT_MODEL_ID", "wonrax/phobert-base-vietnamese-sentiment")
        self.models["sentiment_tokenizer"] = AutoTokenizer.from_pretrained(model_id)
        self.models["sentiment_model"] = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)

# --------------------------
# FastAPI Application
# --------------------------
app = FastAPI(
    title="Model Serving API",
    description="API for NLP model serving including routing, embedding and sentiment analysis",
    version="1.0.0"
)

# Load models at startup
try:
    model_loader = ModelLoader()
except Exception as e:
    logger.critical(f"Failed to initialize models: {str(e)}")
    raise

# --------------------------
# Request Models
# --------------------------
class QuestionRequest(BaseModel):
    content: str

class ListRequest(BaseModel):
    items: List[str]

class RouterResponse(BaseModel):
    intent: str
    secCd: str
    contentType: str

class EmbeddingResponse(BaseModel):
    vectors: List[List[float]]

class SentimentResponse(BaseModel):
    scores: List[float]

# --------------------------
# Exception Handlers
# --------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return HTTPException(
        status_code=500,
        detail="Internal server error"
    )

# --------------------------
# API Endpoints
# --------------------------
@app.get("/", tags=["Health Check"])
def health_check():
    """Service health check endpoint"""
    return {"status": "healthy"}

@app.post("/router_agent/inference", response_model=RouterResponse, tags=["Routing"])
def router_agent_predict(request: QuestionRequest):
    """
    Predict intent and routing information for a question
    """
    try:
        router = model_loader.models["router_agent"]
        intent, secCd, contentType = router.inference(request.content, "1", "1")
        return {
            "intent": intent,
            "secCd": secCd,
            "contentType": contentType
        }
    except Exception as e:
        logger.error(f"Router agent error: {str(e)}")
        raise HTTPException(status_code=500, detail="Router agent prediction failed")

@app.post("/embedding/encode", response_model=EmbeddingResponse, tags=["Embeddings"])
def encode_texts(request: ListRequest):
    """
    Generate embeddings for a list of texts
    """
    try:
        embeddings = model_loader.models["embed_model"].encode(
            request.items,
            batch_size=32,
            show_progress_bar=False,
            convert_to_tensor=True
        )
        return {"vectors": embeddings.tolist()}
    except Exception as e:
        logger.error(f"Embedding generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Embedding generation failed")

@app.post("/sentiment/predict", response_model=SentimentResponse, tags=["Sentiment Analysis"])
def predict_sentiment(request: ListRequest):
    """
    Predict sentiment scores for a list of texts
    """
    try:
        tokenizer = model_loader.models["sentiment_tokenizer"]
        model = model_loader.models["sentiment_model"]
        
        inputs = tokenizer(
            request.items,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(model_loader.device)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.nn.functional.softmax(logits, dim=1)
        scores = probs[:, 0]*-1 + probs[:, 1]*0 + probs[:, 2]*1
        return {"scores": scores.tolist()}
    except Exception as e:
        logger.error(f"Sentiment prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Sentiment prediction failed")

# --------------------------
# Main Application Entry
# --------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8082,
        log_level="info",
        reload=False  # Set to True for development only
    )