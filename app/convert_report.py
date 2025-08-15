from app.db.postgre import get_all_pending_preprocess
from app.rag.collections import COLLECTIONS
from app.api.process_stockcode_pdf import process_stockcode_pdf
from app.api.preprocess import check_and_update_duplicates


# Trạng thái hiện tại collection
print(COLLECTIONS["stock"].get_collection_info())

# Reset collection
COLLECTIONS["stock"].recreate_collection()

# Trạng thái hiện tại collection
print(COLLECTIONS["stock"].get_collection_info())

# Xử lý
process_stockcode_pdf("data/raw_data")

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
check_and_update_duplicates(all_data, 0.85)

# Trạng thái hiện tại collection
print(COLLECTIONS["news"].get_collection_info())

# Xong
print("Done processing news.") 
