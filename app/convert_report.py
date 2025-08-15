from app.db.postgre import get_all_pending_preprocess
from app.rag.collections import COLLECTIONS
from app.rag.indexer import insert_stock_data
from app.rag.indexer import insert_news_data


# Trạng thái hiện tại collection
print(COLLECTIONS["stock"].get_collection_info())

# Reset collection
COLLECTIONS["stock"].recreate_collection()

# Trạng thái hiện tại collection
print(COLLECTIONS["stock"].get_collection_info())

# Xử lý
insert_stock_data("data/raw_data")

# Trạng thái hiện tại collection
print(COLLECTIONS["stock"].get_collection_info())

# Xong
print("Done processing stockcode PDFs.")

##########################################################

# Kiểm tra số lượng dữ liệu cần xử lý
all_data = get_all_pending_preprocess()

# Trạng thái hiện tại collection
print(COLLECTIONS["news"].get_collection_info())

# Reset collection
COLLECTIONS["news"].recreate_collection()

# Trạng thái hiện tại collection
print(COLLECTIONS["news"].get_collection_info())

# Xử lý
insert_news_data(all_data, 0.85, True)

# Trạng thái hiện tại collection
print(COLLECTIONS["news"].get_collection_info())

# Xong
print("Done processing news.") 
