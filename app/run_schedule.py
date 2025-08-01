# Import hàm
from app.api.crawl import run_crawler
from app.db.postgre import insert_news

# # 1️. Crawl dữ liệu
print("Bắt đầu Crawl dữ liệu...")
all_data = run_crawler("all", 1)
if all_data: 
    insert_news(all_data)
print("Crawl thành công!")
