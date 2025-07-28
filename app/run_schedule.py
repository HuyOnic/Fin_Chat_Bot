import sys
import os

# Thêm đường dẫn gốc vào sys.path nếu cần
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "app")))

# Import hàm
from api.crawl import run_crawler

# # 1️. Crawl dữ liệu
print("Bắt đầu Crawl dữ liệu...")
crawl_result = run_crawler("all", 1)
print("Crawl thành công!")