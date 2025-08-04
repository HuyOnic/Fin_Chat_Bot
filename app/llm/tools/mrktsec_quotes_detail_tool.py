import requests, json
import re
from langchain.agents import Tool

def extract_text_only(html_items):
    return "\n".join([re.sub(r'<.*?>', '', item["data"]).strip() for item in html_items])

def get_mrktsec_quotes_detail(secCd, contentType, language, jwt_token):
    url = "https://api-ai.goline.vn/api/public/chat-management/test"
    params = {
        "api": f"http://10.10.3.31:7000/market/api/public/mrktsec-quotes-detail?secCd={secCd}&language={language}&contentType={contentType}"
    }
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-US,en;q=0.9",
        "authorization": "Basic",
        "cache-control": "no-cache",
        "clienttime": "20250715083305",
        "content-type": "application/json",
        "mac-address": "",
        "origin": "https://trade-demo.goline.vn",
        "pragma": "no-cache",
        "priority": "u=1, i",
        "referer": "https://trade-demo.goline.vn/",
        "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Microsoft Edge";v="138"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "token": jwt_token,
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
        "x-client-request-id": "feb1feea-59b3-4362-a0c1-44db62dc11b6",
        "x-master-account": "045C003127",
        "x-session_state": "3d4b43d3-21f7-40de-9a97-a441713724ff",
        "x-src-channel": "4",
        "x-version": "v.1.1.25.07011627",
        "Cookie": jwt_token
    }
    json_body = {
        "token": jwt_token
    }
    try:
        response = requests.get(url, headers=headers, params=params, json=json_body)
        response_text = json.loads(json.loads(response.text)["data"]["data"])["data"][0]["data"][1]["data"]
        context = f"Giá của mã {secCd}\n" + extract_text_only(response_text)
        return context
    
    except Exception as e:
        print("Lỗi khi gọi market API:", e)

def get_mrktsec_quotes_detail_wrapper(input: str):
    try:
        args = json.loads(input)
        return get_mrktsec_quotes_detail(
            secCd=args["secCd"],
            contentType=args["contentType"],
            language=args["language"],
            jwt_token=args["jwt_token"]
        )
    except Exception as e:
        return f"Lỗi khi xử lý input của mrktsec_quotes_detail {e}"

def mrktsec_quotes_detail_tool():
    return Tool(name="mrktsec_quotes_detail",
                func=get_mrktsec_quotes_detail_wrapper,
                description="Công cụ lấy giá và ngành hàng của mã chứng khoán")

if __name__=="__main__":
    secCd="BCG,ACB"
    language="VI"
    contentType="ALL"
    jwt_token="token_chatbot=eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJQSXNJZ1lKVDBIbGRMUTBzMUdEU2d0TC02M0dicTFKeTFNUGpvZS1GTjRzIn0.eyJleHAiOjE3NTI1NTc1ODUsImlhdCI6MTc1MjU0MzE4NSwianRpIjoiYTA2N2E1ZjEtYWU5Zi00OGVjLWIxMzYtZDY4YTQyYmFhNTMzIiwiaXNzIjoiaHR0cDovL2hvc3QuZG9ja2VyLmludGVybmFsOjkwMDAvYXV0aC9yZWFsbXMvdmdhaWEiLCJhdWQiOiJhY2NvdW50Iiwic3ViIjoiZjpiY2NjMjRmNy03ZGQ3LTRiMGUtYWZmOS05OGIwNjc0Njc2ZDc6MDQ1YzAwMzEyNyIsInR5cCI6IkJlYXJlciIsImF6cCI6InZnYWlhLWNsaWVudCIsInNlc3Npb25fc3RhdGUiOiIzZDRiNDNkMy0yMWY3LTQwZGUtOWE5Ny1hNDQxNzEzNzI0ZmYiLCJhY3IiOiIxIiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwiQ1VTVE9NRVIiLCJ1bWFfYXV0aG9yaXphdGlvbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoidmdhaWEtc2NvcGUgcHJvZmlsZSBlbWFpbCIsInNpZCI6IjNkNGI0M2QzLTIxZjctNDBkZS05YTk3LWE0NDE3MTM3MjRmZiIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwibmFtZSI6Ik1yLiAwNDVDMDAzMTI3IiwicHJlZmVycmVkX3VzZXJuYW1lIjoiMDQ1YzAwMzEyNyIsImdpdmVuX25hbWUiOiJNci4iLCJmYW1pbHlfbmFtZSI6IjA0NUMwMDMxMjciLCJlbWFpbCI6IjA0NUMwMDMxMjdAZ21haWwuY29tIn0.BdSITRpF7AtNE_0mEjBK-ybJ8SWzKq77uYJT0NL75tMxt4gp48fP_NB_BN9sxl6PTwPSnvB0uYxMsUVCdg1hvj8vkP3YlM06EadRNSqoLE3Gua2aC_4echjf8rB8SU8Dqvs8mdF2MlphX5qmSyoUBwqFwEalw5HazBPWXZXXBIgLHgqD2yP6ZegSyGHv8lyRm7QzH2EUrZ_7eMKEtAyWpBhin61qijyGQM5eTUcv5FxgQkR54AAZqQjhzllPPlsn2Uday-JP-t6C1-mSKIFUxtuar-ce09C_xZK9cm0NmQhYeK7lxeP9Vzi-g8oBsdApfpDZHpYs7ICIaFdXLAILQA"
    print(get_mrktsec_quotes_detail(secCd, contentType, language, jwt_token))


