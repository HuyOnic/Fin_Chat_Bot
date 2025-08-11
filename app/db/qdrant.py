from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import os
import torch
import pandas as pd
from collections import defaultdict
from app.utils.utils import normalize
load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
# COLLECTION_NAME = "news_vectors"
COLLECTION_NAME = "news_hybrid_search"
# STOCKCODE_COLLECTION_NAME = "stockcode_vectors"
STOCKCODE_COLLECTION_NAME = "stockcode_hybrid_search"

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
            vectors_config={
                "dense_vector": models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE,
                )
            },

            sparse_vectors_config={
                "sparse_vector": models.SparseVectorParams(
                    index=models.SparseIndexParams()
                )
            },
        )
        print(f"Đã tạo collection '{collection_name}'")
    except Exception as e:
        print(f"Lỗi khi tạo collection: {str(e)}")


def insert_vector(article_id, dense_vector, sparse_vector, payload_keys, payload_values, collection_name=COLLECTION_NAME):
    if not isinstance(dense_vector, list):
        dense_vector = dense_vector.tolist()

    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=article_id,
                vector={
                    "dense_vector": dense_vector,
                    "sparse_vector": sparse_vector,
                },
                payload=dict(zip(payload_keys, payload_values))
            )
        ]
    )


def _search_dense(query_vector, top_k, collection_name=COLLECTION_NAME):
    if not isinstance(query_vector, list):
        query_vector = query_vector.tolist()

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,  
        using="dense_vector",  
        with_payload=True,
        limit=top_k,
    )
    
    return results.points 

def search_dense(query_vector, top_k=3, threshold=0.3, collection_name=COLLECTION_NAME):
    results = _search_dense(query_vector, top_k, collection_name)
    return [point for point in results if point.score >= threshold]

def _search_sparse(sparse_vector_dict, top_k, collection_name=COLLECTION_NAME):
    sparse_vector = models.SparseVector(
        indices=sparse_vector_dict["indices"],
        values=sparse_vector_dict["values"]
    )

    results = client.query_points(
        collection_name=collection_name,
        query=sparse_vector,  
        using="sparse_vector",  
        with_payload=True,
        limit=top_k,
    )
    return results.points

def search_sparse(sparse_vector_dict, top_k=3, threshold=0.3, collection_name=COLLECTION_NAME):
    results = _search_sparse(sparse_vector_dict, top_k, collection_name)
    return [point for point in results if point.score >= threshold]

def hybrid_search(dense_vector, sparse_vector, top_k=3, threshold=0.3, alpha=0.7, collection_name=COLLECTION_NAME):
    dense_results = _search_dense(dense_vector, top_k * 3 + 1, collection_name)
    dense_scores = normalize([point.score for point in dense_results])
    for i, point in enumerate(dense_results):
        point.score = dense_scores[i]

    sparse_results = _search_sparse(sparse_vector, top_k * 3 + 1, collection_name)
    sparse_scores = normalize([point.score for point in sparse_results])
    for i, point in enumerate(sparse_results):
        point.score = sparse_scores[i]

    combined = defaultdict(lambda: {"payload": None, "dense": 0.0, "sparse": 0.0})
    for p in sparse_results:
        combined[p.id]["point"] = p
        combined[p.id]["sparse"] = p.score

    for p in dense_results:
        combined[p.id]["point"] = p
        combined[p.id]["dense"] = p.score

    final = []
    for id_, entry in combined.items():
        score = alpha * entry["dense"] + (1 - alpha) * entry["sparse"]
        if score >= threshold:
            entry['point'].score = score
            final.append(entry["point"])

    final = sorted(final, key=lambda d: d.score, reverse=True)[:top_k]
    return final

def get_similar_vectors(vector, top_k=3, threshold=0.85, collection_name=COLLECTION_NAME):
    if not isinstance(vector, list):
        vector = vector.tolist()

    search_result = search_dense(vector, top_k=top_k, threshold=threshold, collection_name=collection_name)

    similar_articles = [
        (hit.id, hit.score) for hit in search_result 
    ]

    return similar_articles


def get_documents_by_vector(dense_vector, sparse_vector, top_k=3, threshold=0.1, collection_name=COLLECTION_NAME):
    if not isinstance(dense_vector, list):
        dense_vector = dense_vector.tolist()

    search_result = hybrid_search(dense_vector, sparse_vector, top_k=top_k, threshold=threshold, collection_name=collection_name)

    docs = [hit.payload for hit in search_result]
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