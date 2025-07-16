from transformers.onnx import export
from pathlib import Path
from transformers.onnx.features import FeaturesManager
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

onnx_path = Path("sentiment_project/models/onnx/Llama-3.1-8B-Instruct.onnx")
model_kind, model_onnx_config = FeaturesManager.ccheck_supported_model_or_raise(model, feature="sequence-classification")



