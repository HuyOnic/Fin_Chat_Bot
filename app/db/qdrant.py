from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import os
import torch
import pandas as pd
load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
COLLECTION_NAME = "news_vectors"
STOCKCODE_COLLECTION_NAME = "stockcode_vectors"

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)

try:
    # Gửi yêu cầu lấy danh sách collection để kiểm tra kết nối
    response = client.get_collections()
    print(f"✅ Kết nối thành công Qdrant {QDRANT_HOST}:{QDRANT_PORT}. Danh sách collection:", response)
except Exception as e:
    print("❌ Lỗi kết nối đến Qdrant:", e)

def create_collection(collection_name=COLLECTION_NAME):
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE
            )
        )
        print(f"Đã tạo collection '{collection_name}'")
    except Exception as e:
        print(f"Lỗi khi tạo collection: {str(e)}")


def insert_vector(article_id, vector, payload_keys, payload_values, collection_name=COLLECTION_NAME):
    if not isinstance(vector, list):
        vector = vector.tolist()
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=article_id,
                vector=vector,
                payload=dict(zip(payload_keys, payload_values))
            )
        ]
    )

def get_similar_vectors(vector, top_k=3, threshold=0.85, collection_name=COLLECTION_NAME):
    if not isinstance(vector, list):
        vector = vector.tolist()
    search_result = client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=top_k
    )

    similar_articles = [
        (hit.id, hit.score) for hit in search_result if hit.score >= threshold
    ]

    return similar_articles


def get_documents_by_vector(vector, top_k=3, threshold=0.1, collection_name=COLLECTION_NAME):
    if not isinstance(vector, list):
        vector = vector.tolist()
    search_result = client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=top_k
    )

    docs = [hit.payload for hit in search_result if hit.score >= threshold]
    return docs


def view_collection_data(limit=20, collection_name=COLLECTION_NAME):
    # Lấy dữ liệu với scroll API
    records, next_page = client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True,    # Bao gồm metadata
        with_vectors=True,    # Bao gồm vector
    )

    print(f"\nHiển thị {len(records)} bản ghi từ collection '{collection_name}':")
    for idx, record in enumerate(records, 1):
        print(f"\nRecord {idx}:")
        print(f"ID: {record.id}")
        print(f"Payload: {record.payload}")
        print(f"Vector (first 5 dim): {record.vector[:5]}...")  # Chỉ hiển thị 5 chiều đầu
    
    return records

def qdrant2csv(limit=1000, collection_name=COLLECTION_NAME):
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=limit,  # Số lượng tối đa
        with_payload=True,
        with_vectors=True  # Bỏ qua vector nếu không cần
    )

    # Chuyển sang DataFrame
    data = []
    for point in points:
        row = {
            "id": point.id,
            "vector": point.vector,
            **point.payload  # Giả sử payload là dict
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Lưu thành CSV
    df.to_csv("qdrant_export.csv", index=False)
    print(f"Đã xuất {len(df)} bản ghi ra file qdrant_export.csv")

if __name__=="__main__":
    create_collection("fiinquant_documents")
    # vector = torch.rand(1024).numpy()
    # sim = get_similar_vectors(vector)
    # insert_vector(1, vector, "cafef")
    # print("inserted")
    #qdrant2csv()
    pass