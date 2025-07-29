# Import hàm
from app.api.crawl import run_crawler

# # 1️. Crawl dữ liệu
print("Bắt đầu Crawl dữ liệu...")
crawl_result = run_crawler("all", 1)
print("Crawl thành công!")