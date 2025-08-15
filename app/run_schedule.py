# Import hàm
from app.api.crawl import run_crawler
from app.db.postgre import insert_into_news_table 

# # 1️. Crawl dữ liệu
print("Bắt đầu Crawl dữ liệu...")
all_data = run_crawler("all", 1)
if all_data: 
    insert_into_news_table (all_data)
print("Crawl thành công!")
