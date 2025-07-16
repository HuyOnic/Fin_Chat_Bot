import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# Headers tối giản nhưng vẫn giữ thông tin quan trọng
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Cookie": "ASP.NET_SessionId=tivbnrprki4aqfqpbvp1tzkt; _ga=GA1.2.1653413231.1730861893;"
}

market_analysis = []
global_stocks = []
macro_economy = []

'''
channelID:
- 18832: tài chính quốc tế (tai-chinh-quoc-te)
- 18833: kinh tế vĩ mô (vi-mo-dau-tu)
- 18839: thị trường (thi-truong)
'''

def cafefCrawler(channelID, news_type, topic, days):
    current_time = datetime.now()
    time_ago = current_time - timedelta(days=days)
    
    print("Crawling cafef với topic là:", topic)
    main_url = f"https://cafef.vn/{topic}.chn"
    news_list = []

    # Đầu tiên, crawl trang chủ
    response = requests.get(main_url, headers=headers)

    should_continue = True
    if response.status_code == 200:
        print("(cafef) Request trang chủ thành công!")
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = soup.find_all('div', class_=['tlitem', 'box-category-item'])
        for item in news_items:
            # print("item:", item)
            # Tìm thời gian trong thẻ span có class time-ago
            time_span = item.find('span', class_='time-ago')
            time_str = time_span['title'] if time_span else None
            # print("\n\nTIME:", time_str,"\n\n")
            if time_str:
                try:
                    news_time = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
                    # So sánh với thời gian hiện tại để xem chênh lệch bao nhiêu ngày
                    time_diff = current_time - news_time
                    # So sánh với mốc trước để quyết định có dừng không
                    if news_time < time_ago:
                        print(f"Đã đạt đến dữ liệu cũ hơn {days} ngày. Dừng crawl.")
                        should_continue = False
                        break

                    # Lấy URL từ thẻ <a> đầu tiên trong item
                    link = item.find('a')['href']
                    # Thêm domain nếu là đường dẫn tương đối
                    if not link.startswith('http'):
                        link = f"https://cafef.vn{link}"
                    
                    # Request đến trang chi tiết để lấy nội dung
                    detail_response = requests.get(link, headers=headers)
                    if detail_response.status_code == 200:
                        detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
                        content_div = detail_soup.find('div', class_='detail-content afcbc-body')
                        content = content_div.get_text(strip=True) if content_div else item.text.strip()
                    else:
                        print(f"Lỗi khi lấy nội dung chi tiết: {detail_response.status_code}")
                        content = item.text.strip()
                    
                    # Tạo dict cho mỗi bản tin
                    news_data = {
                        'news_type': news_type,  # Tin thị trường chứng khoán
                        'sec_cd': None,  # Không có mã CK cụ thể
                        'market_cd': None,  # Chưa xác định thị trường
                        'content': content.replace("TIN MỚI", ""),  # Nội dung chi tiết từ trang bài viết
                        'score': None,  # Chưa có điểm
                        'manual_score': None,  # Chưa có điểm thủ công
                        'source': link,  # URL đầy đủ của bài viết
                        'news_date': int(news_time.strftime('%Y%m%d%H%M')),  # Format: YYYYMMDDHHmm
                        'status': 0,  # Trạng thái chưa tiền xử lý
                        'remarks': None,  # Chưa có ghi chú
                    }
                    
                    news_list.append(news_data)
                    
                except Exception as e:
                    print(f"Lỗi xử lý dữ liệu: {str(e)}")
    else:
        print(f"(cafef) Lỗi khi lấy dữ liệu trang chủ: {response.status_code}")

    # Sau đó, crawl các trang timeline
    page = 2
    while should_continue:
        url = f"https://cafef.vn/timelinelist/{channelID}/{page}.chn"
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            print(f"(cafef) Request timeline trang {page} thành công!")
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = soup.find_all('div', class_=['tlitem', 'box-category-item'])
            
            for item in news_items:
                # Tìm thời gian trong thẻ span có class time-ago
                time_span = item.find('span', class_='time-ago')
                time_str = time_span['title'] if time_span else None
                # print("\n\nTIME:", time_str,"\n\n")
                
                if time_str:
                    try:
                        news_time = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
                        time_diff = current_time - news_time
                        # So sánh với trước để quyết định có dừng không
                        if news_time < time_ago:
                            print(f"Đã đạt đến dữ liệu cũ hơn {days} ngày. Dừng crawl.")
                            should_continue = False
                            break

                        # Lấy URL từ thẻ <a> đầu tiên trong item
                        link = item.find('a')['href']
                        # Thêm domain nếu là đường dẫn tương đối
                        if not link.startswith('http'):
                            link = f"https://cafef.vn{link}"
                        
                        # Request đến trang chi tiết để lấy nội dung
                        detail_response = requests.get(link, headers=headers)
                        if detail_response.status_code == 200:
                            detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
                            content_div = detail_soup.find('div', class_='detail-content afcbc-body')
                            content = content_div.get_text() if content_div else item.text.strip()
                        else:
                            print(f"(cafef) Lỗi khi lấy nội dung chi tiết: {detail_response.status_code}")
                            content = item.text
                        content = content.replace("TIN MỚI", "")
                        content = content.strip() 
                        content = ' '.join(content.split())
                        # Tạo dict cho mỗi bản tin
                        news_data = {
                            'news_type': news_type,  # Tin thị trường CK
                            'sec_cd': None,  # Không có mã CK cụ thể
                            'market_cd': None,  # Không có mã sàn
                            'content': content,  # Nội dung chi tiết từ post
                            'score': None,  # Chưa có điểm
                            'manual_score': None,  # Chưa có điểm thủ công
                            'source': link,  # Link bài viết
                            'news_date': int(news_time.strftime('%Y%m%d%H%M')),  # Chuyển timestamp thành integer
                            'status': 0,  # Trạng thái mặc định là chưa tiền xử lý
                            'remarks': None,  # Chưa có ghi chú
                        }
                        
                        news_list.append(news_data)
                    except Exception as e:
                        print(f"(cafef) Lỗi xử lý dữ liệu: {str(e)}")
            
            page += 1
        else:
            print(f"(cafef) Lỗi khi lấy dữ liệu timeline trang {page}: {response.status_code}")
            break
    return news_list
 
# global_stocks = cafefCrawler(channelID=18832, news_type=1, topic="tai-chinh-quoc-te")
# market_analysis = cafefCrawler(channelID=18839, news_type=1, topic="thi-truong")
# macro_economy = cafefCrawler(channelID=18833, news_type=3, topic="vi-mo-dau-tu")