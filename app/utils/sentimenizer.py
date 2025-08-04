import requests

def sentiment_analysis(content_list: list):
    url = "http://localhost:8082/sentiment/predict"
    payload = {
        "items": content_list
    }
    try:
        response = requests.post(url, json=payload)
        return response.json()["scores"]
    except Exception as e:
        print("Lỗi khi gọi API sentiment:", e)

if __name__=="__main__":
    pass


