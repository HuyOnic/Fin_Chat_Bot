from qdrant_client import QdrantClient
from app.db.postgre import get_all_pending_preprocess
from app.db.qdrant import STOCKCODE_COLLECTION_NAME, COLLECTION_NAME, create_collection
from app.api.process_stockcode_pdf import process_stockcode_pdf
from app.api.preprocess import check_and_update_duplicates

client = QdrantClient(host="localhost", port=6333)

# Trạng thái hiện tại collection
print(client.get_collection(STOCKCODE_COLLECTION_NAME))

# Reset collection
create_collection(STOCKCODE_COLLECTION_NAME)

# Trạng thái hiện tại collection
print(client.get_collection(STOCKCODE_COLLECTION_NAME))

# Xử lý
process_stockcode_pdf("data/raw_data")

# Trạng thái hiện tại collection
print(client.get_collection(STOCKCODE_COLLECTION_NAME))

# Xong
print("Done processing stockcode PDFs.")

##########################################################

client = QdrantClient(host="localhost", port=6333)

# Kiểm tra số lượng dữ liệu cần xử lý
all_data = get_all_pending_preprocess()

# Trạng thái hiện tại collection
print(client.get_collection(COLLECTION_NAME))

# Reset collection
create_collection(COLLECTION_NAME)

# Trạng thái hiện tại collection
print(client.get_collection(COLLECTION_NAME))

# Xử lý
check_and_update_duplicates(0.85)

# Trạng thái hiện tại collection
print(client.get_collection(COLLECTION_NAME))

# Xong
print("Done processing news.") 
