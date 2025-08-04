import time
LOCAL_MODELS = [
    {
        "id": "llama3-8b",
        "object": "model",
        "created": int(time.time()) - 86400,  # giả định tạo hôm qua
        "owned_by": "HuyOnic"
    },
    {
        "id": "mistral-7b-instruct",
        "object": "model",
        "created": int(time.time()) - 172800,
        "owned_by": "local"
    },
    {
        "id": "vinallm-chat-4",
        "object": "model",
        "created": int(time.time()) - 259200,
        "owned_by": "vinallm-team"
    }
]