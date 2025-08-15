from app.rag.collections import COLLECTIONS
from app.rag._utils import get_files_in_directory, load_file_as_markdown, chunking, convert_date
from app.db.postgre import get_last_news_id, update_status


def insert_stock_data(input_dir):
    files = get_files_in_directory(input_dir)
    for f in files:
        if f.endswith(".pdf"):
            file_name = f.split("/")[-1][:-4]
            print(f"Processing file: {f}")
            markdown_content = load_file_as_markdown(f)
            chunks = chunking(markdown_content)
            for chunk in chunks:  
                COLLECTIONS["stock"].insert_data(["content", "source"], [chunk, file_name], [0, 1])


def insert_news_data(all_data, threshold: float, all_db_data: False):
    if not all_data:
        print("Không có dữ liệu để kiểm tra")
        return {"message": "Không có dữ liệu để kiểm tra"}
    
    print(f"Bắt đầu kiểm tra trùng lặp (threshold={threshold})")
    
    if all_db_data:
        all_data = [(row[0], row[1], convert_date(row[2]), row[3], row[4]) for row in all_data]
    else:
        # giả định trước đó đã add data vào database.
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
   
        chunks = chunking(current_content)

        for chunk_idx, chunk in enumerate(chunks):
            similar_articles = COLLECTIONS['news'].get_similar_vectors(chunk, top_k=3, threshold=threshold)

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
                COLLECTIONS["news"].insert_data(payload_keys=["news_id", "content", "source", "news_date", "status"],
                                                payload_values=[[current_id, chunk, source_domain, current_date, status]],
                                                embedding_indices=[1, 2])

                results.append({
                    "current_id": current_id,
                    "status": "saved_to_qdrant",
                    "status_updated": 1
                })

    print("Xử lý xong tất cả bài viết")
    return {"message": "Hoàn thành kiểm tra trùng lặp", "results": results}