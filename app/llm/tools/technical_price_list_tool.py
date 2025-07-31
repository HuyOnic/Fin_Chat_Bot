#http://10.10.3.31:7000/market/api/public/mrktsec-quotes-detail?secCd=SSI&contentType=lastPrice&language=VI
import requests
from pprint import pprint
import json
#API này đang bị lỗi, trả về '{"status":"SUCCESS","arg":null}'
import re

import re

def simplify_analysis(raw_data) -> str:
    def strip_html_tags(text):
        return re.sub(r'<[^>]+>', '', text).strip()

    context_parts = []

    for stock in raw_data:
        symbol = stock.get("id", "")
        oscillators = []
        moving_averages = []
        summaries = []
        current_section = None

        for item in stock.get("data", []):
            item_type = item.get("type")
            item_data = item.get("data")

            if item_type == "div" and isinstance(item_data, str):
                if "dao động" in item_data:
                    current_section = "oscillators"
                elif "trung bình động" in item_data:
                    current_section = "moving_averages"

            elif item_type == "table" and current_section:
                for row in item_data.get("rows", []):
                    indicator = row[0].get("data")
                    value = row[1]
                    action = row[2].get("data")
                    value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                    line = f"- {indicator}: {value_str} → {action}"
                    if current_section == "oscillators":
                        oscillators.append(line)
                    elif current_section == "moving_averages":
                        moving_averages.append(line)

            elif item_type == "html" and isinstance(item_data, str):
                summaries.append(f"- {strip_html_tags(item_data)}")

        # Build context string
        stock_context = []
        stock_context.append(f"Phân tích kỹ thuật mã {symbol}:")
        stock_context.append("Chỉ báo dao động:")
        stock_context.extend(oscillators if oscillators else ["- (Không có dữ liệu)"])
        stock_context.append("Chỉ báo trung bình động:")
        stock_context.extend(moving_averages if moving_averages else ["- (Không có dữ liệu)"])
        stock_context.append("Tổng kết:")
        stock_context.extend(summaries if summaries else ["- (Không có tổng kết)"])
        stock_context.append("=" * 50)

        context_parts.append("\n".join(stock_context))

    return "\n\n".join(context_parts)



def get_technical_price_list(secCd, contentType, language, jwt_token):
    url = "https://api-ai.goline.vn/api/public/chat-management/test"
    params = {
        "api": f'http://10.10.3.31:7000/market/api/public/technical-price-list?contentType={contentType}&period=1M&language={language}&secList={secCd}'
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
        return simplify_analysis(json.loads(json.loads(response.text)["data"]["data"])["data"])
    except Exception as e:
        print("Lỗi khi gọi market API:", e)

if __name__=="__main__":
    secCd="VCB,ACB,SSI"
    language="VI"
    contentType="ALL"
    jwt_token="token_chatbot=eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJQSXNJZ1lKVDBIbGRMUTBzMUdEU2d0TC02M0dicTFKeTFNUGpvZS1GTjRzIn0.eyJleHAiOjE3NTI1NTc1ODUsImlhdCI6MTc1MjU0MzE4NSwianRpIjoiYTA2N2E1ZjEtYWU5Zi00OGVjLWIxMzYtZDY4YTQyYmFhNTMzIiwiaXNzIjoiaHR0cDovL2hvc3QuZG9ja2VyLmludGVybmFsOjkwMDAvYXV0aC9yZWFsbXMvdmdhaWEiLCJhdWQiOiJhY2NvdW50Iiwic3ViIjoiZjpiY2NjMjRmNy03ZGQ3LTRiMGUtYWZmOS05OGIwNjc0Njc2ZDc6MDQ1YzAwMzEyNyIsInR5cCI6IkJlYXJlciIsImF6cCI6InZnYWlhLWNsaWVudCIsInNlc3Npb25fc3RhdGUiOiIzZDRiNDNkMy0yMWY3LTQwZGUtOWE5Ny1hNDQxNzEzNzI0ZmYiLCJhY3IiOiIxIiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwiQ1VTVE9NRVIiLCJ1bWFfYXV0aG9yaXphdGlvbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoidmdhaWEtc2NvcGUgcHJvZmlsZSBlbWFpbCIsInNpZCI6IjNkNGI0M2QzLTIxZjctNDBkZS05YTk3LWE0NDE3MTM3MjRmZiIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwibmFtZSI6Ik1yLiAwNDVDMDAzMTI3IiwicHJlZmVycmVkX3VzZXJuYW1lIjoiMDQ1YzAwMzEyNyIsImdpdmVuX25hbWUiOiJNci4iLCJmYW1pbHlfbmFtZSI6IjA0NUMwMDMxMjciLCJlbWFpbCI6IjA0NUMwMDMxMjdAZ21haWwuY29tIn0.BdSITRpF7AtNE_0mEjBK-ybJ8SWzKq77uYJT0NL75tMxt4gp48fP_NB_BN9sxl6PTwPSnvB0uYxMsUVCdg1hvj8vkP3YlM06EadRNSqoLE3Gua2aC_4echjf8rB8SU8Dqvs8mdF2MlphX5qmSyoUBwqFwEalw5HazBPWXZXXBIgLHgqD2yP6ZegSyGHv8lyRm7QzH2EUrZ_7eMKEtAyWpBhin61qijyGQM5eTUcv5FxgQkR54AAZqQjhzllPPlsn2Uday-JP-t6C1-mSKIFUxtuar-ce09C_xZK9cm0NmQhYeK7lxeP9Vzi-g8oBsdApfpDZHpYs7ICIaFdXLAILQA"
    pprint(get_technical_price_list(secCd, contentType, language, jwt_token))


