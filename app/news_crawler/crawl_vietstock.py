import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# URL của API để lấy nội dung trang
url = "https://vietstock.vn/StartPage/ChannelContentPage"

# Headers để giả lập trình duyệt
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Cookie": "ASP.NET_SessionId=yor4lfmsrelkty0rbnffeu43; _ga=GA1.1.1511410349.1731893017"
}

market_analysis = []
global_stocks = []
macro_economy = []

'''
channelID:
- 761: kinh tế vĩ mô
- 773: chứng khoán thế giới
- 1636: nhận định thị trường
'''
def vietstockCrawler(channelID, news_type, days):
    current_time = datetime.now()
    time_ago = current_time - timedelta(days=days)
    
    print("Crawling vietstock với channelID là:", channelID)
    news_list = []
    page = 1
    should_continue = True

    while should_continue:
        payload = {
            "channelID": channelID,
            "page": page
        }

        response = requests.post(url, headers=headers, data=payload)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            h4_tags = soup.find_all("h4")
            
            if not h4_tags:
                print(f"(Vietstock) Không còn bài viết nào ở trang {page}")
                break
            
            print(f"\n(Vietstock) Đang xử lý trang {page}:")
            for tag in h4_tags:
                title = tag.get_text(strip=True)
                link = tag.find('a')['href']
                
                # Xử lý link - thêm domain nếu không có http
                if not link.startswith('http'):
                    link = f"https://vietstock.vn{link}"
                
                # Lấy nội dung chi tiết và thời gian từ bài viết
                try:
                    article_response = requests.get(link, headers=headers)
                    if article_response.status_code == 200:
                        article_soup = BeautifulSoup(article_response.text, 'html.parser')
                        
                        # Lấy thời gian từ meta tag mới
                        meta_time = article_soup.find('meta', {'property': 'article:published_time'})
                        if meta_time:
                            time_str = meta_time.get('content')
                            # Chuyển đổi thời gian về định dạng YYYYMMDDHHmm
                            article_time = datetime.strptime(time_str.split('+')[0], "%Y-%m-%dT%H:%M:%S")
                            timestamp = article_time.strftime("%Y%m%d%H%M")
                        else:
                            print(f"(Vietstock) Không tìm thấy thời gian trong bài viết: {link}")
                            timestamp = "N/A"
                        
                        # Kiểm tra thời gian có nằm trong khoảng cho phép
                        if timestamp != "N/A":
                            article_time = datetime.strptime(timestamp, "%Y%m%d%H%M")
                            if article_time < time_ago:
                                print(f"Đã đạt đến dữ liệu cũ hơn {days} ngày. Dừng crawl.")
                                should_continue = False
                                break
                        
                        # Lấy nội dung bài viết
                        p_tags = article_soup.find_all('p', class_='pBody')
                        content = ' '.join(p.get_text() for p in p_tags)
                        content = ' '.join(content.split())
                    else:
                        print(f"(Vietstock) Lỗi khi lấy nội dung bài viết: {link}")
                        content = None
                        timestamp = "N/A"
                except Exception as e:
                    print(f"(Vietstock) Lỗi khi xử lý nội dung bài viết {link}: {str(e)}")
                    content = None
                    timestamp = "N/A"
                
                # Tạo dictionary cho mỗi bản tin
                news_item = {
                    'news_type': news_type,
                    'sec_cd': None,
                    'market_cd': None,
                    'content': content,
                    'score': None,
                    'manual_score': None,
                    'source': link,
                    'news_date': int(timestamp) if timestamp != "N/A" else None,
                    'status': 0,
                    'remarks': None
                }
                
                news_list.append(news_item)
            
            page += 1
        else:
            print(f"(Vietstock) Lỗi khi lấy dữ liệu trang {page}: {response.status_code}")
            break
    return news_list

# global_stocks = vietstockCrawler(channelID=773, news_type=1)
# market_analysis = vietstockCrawler(channelID=1636, news_type=1)
# macro_economy = vietstockCrawler(channelID=761, news_type=3)