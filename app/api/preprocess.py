from ast import Continue
from app.db.postgre import get_all_pending_preprocess, update_status, insert_chunked_news
from app.db.qdrant import insert_vector, get_similar_vectors
from app.utils.vectorizer import convert_to_vector
from app.utils.chunking import chunking_document
from datetime import datetime
import uuid

def convert_news_date(news_date):
    if news_date in [None, 'None', '']:
        return datetime.now()
    try:
        return datetime.strptime(str(news_date), "%Y%m%d%H%M")
    except ValueError:
        # Nếu không đúng định dạng, vẫn trả về ngày hiện tại
        return datetime.now()

def check_and_update_duplicates(threshold: float):
    print(f"Bắt đầu kiểm tra trùng lặp (threshold={threshold})")

    all_data = get_all_pending_preprocess()
    all_data = [(row[0], row[1], convert_news_date(row[2]), row[3], row[4]) for row in all_data]

    if not all_data:
        print("Không có dữ liệu để kiểm tra")
        return {"message": "Không có dữ liệu để kiểm tra"}

    print(f"Tổng số bài viết cần kiểm tra: {len(all_data)}")

    results = []
    for current_id, current_content, current_date, source, status in all_data:
        print(f"Đang kiểm tra bài viết ID {current_id}")
        try:
            source_domain = source.split("//")[1].split("/")[0]
        except:
            continue
        # Chunking current document
        chunks = chunking_document(current_content)

        for chunk_idx, chunk in enumerate(chunks):
            vector = convert_to_vector([chunk])[0]
            similar_articles = get_similar_vectors(vector, top_k=3, threshold=threshold)

            if similar_articles:
                for duplicate_id, similarity_score in similar_articles:
                    update_status(current_id, 9)
                    results.append({
                        "current_id": current_id,
                        "duplicate_id": duplicate_id,
                        "similarity": similarity_score,
                        "status_updated": 9
                    })
            else:
                chunk_id = str(uuid.uuid4())
                # insert_chunked_news(chunk_id, current_id, chunk_idx, chunk)
                insert_vector(chunk_id, vector, current_id, chunk, source_domain, current_date, status)
                # update_status(current_id, 1)
                results.append({
                    "current_id": current_id,
                    "status": "saved_to_qdrant",
                    "status_updated": 1
                })

    print("Xử lý xong tất cả bài viết")
    return {"message": "Hoàn thành kiểm tra trùng lặp", "results": results}
