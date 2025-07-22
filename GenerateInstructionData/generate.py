from vllm import LLM, SamplingParams
import os
os.environ["VLLM_DISABLE_FLASH_ATTN"] = "1"
prompts = [
    "Hello, my name is",
    "The president of the United States is"
    # "The capital of France is",
    # "The future of AI is",
]
model_id = "meta-llama/Meta-Llama-3-8B-Instruct" 
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)
llm = LLM(model=model_id, gpu_memory_utilization=0.65, max_num_batched_tokens=1024)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")