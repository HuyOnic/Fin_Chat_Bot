from ast import Continue
from app.db.postgre import get_last_news_id, update_status
from app.db.qdrant import insert_vector, get_similar_vectors
from app.utils.vectorizer import convert_to_dense_vector, convert_to_sparse_vector
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

def check_and_update_duplicates(all_data, threshold: float):
    if not all_data:
        print("Không có dữ liệu để kiểm tra")
        return {"message": "Không có dữ liệu để kiểm tra"}
    
    print(f"Bắt đầu kiểm tra trùng lặp (threshold={threshold})")
    
    if isinstance(all_data[0], tuple):
        all_data = [(row[0], row[1], convert_news_date(row[2]), row[3], row[4]) for row in all_data]
    else:
        len_data = len(all_data)
        id = get_last_news_id()
        if id is None:
            id = len_data

        all_data = [(id + 1 - len_data + i, 
                    data['content'],
                    data['news_date'],
                    data['source'],
                    data['status'],) for i, data in enumerate(all_data)]


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
            dense_vector = convert_to_dense_vector([chunk])[0]
            sparse_vector = convert_to_sparse_vector(chunk)
            similar_articles = get_similar_vectors(dense_vector, top_k=3, threshold=threshold)

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
                insert_vector(
                    article_id=chunk_id,
                    dense_vector=dense_vector, 
                    sparse_vector=sparse_vector,
                    payload_keys=["news_id", "content", "source", "news_date", "status"],
                    payload_values=[current_id, chunk, source_domain, current_date, status]
                )

                # update_status(current_id, 1)
                results.append({
                    "current_id": current_id,
                    "status": "saved_to_qdrant",
                    "status_updated": 1
                })

    print("Xử lý xong tất cả bài viết")
    return {"message": "Hoàn thành kiểm tra trùng lặp", "results": results}
