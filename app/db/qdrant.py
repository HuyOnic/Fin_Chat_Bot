import os
from dotenv import load_dotenv
from collections import defaultdict
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, SparseVectorParams
from qdrant_client.models import SparseVector, HnswConfigDiff
from qdrant_client.models import SparseIndexParams
from app.db._utils import normalize

load_dotenv() 

QDRANT_CLIENT = QdrantClient(host=os.getenv("QDRANT_HOST"), port=os.getenv("QDRANT_PORT"), 
                             timeout=int(os.getenv("QDRANT_TIMEOUT")))

try:
    # Gửi yêu cầu lấy danh sách collection để kiểm tra kết nối
    response = QDRANT_CLIENT.get_collections()
    print(f"✅ Kết nối thành công Qdrant {os.getenv('QDRANT_HOST')}:{os.getenv('QDRANT_PORT')}. Danh sách collection:", response)
except Exception as e:
    print("❌ Lỗi kết nối đến Qdrant:", e) 

class VectorStore:
    def __init__(self, collection_name, dense_model, sparse_model):
        self.collection_name = collection_name
        self.dense_embedding_model = dense_model
        self.sparse_embedding_model = sparse_model

        if not QDRANT_CLIENT.collection_exists(self.collection_name):
            self._create_collection()

    def _create_collection(self):
        QDRANT_CLIENT.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense_vector": VectorParams(
                    size=self.dense_embedding_model.get_dimension(),
                    distance=Distance.COSINE,
                )
            },

            sparse_vectors_config={
                "sparse_vector": SparseVectorParams(
                    index=SparseIndexParams()
                )
            },

            hnsw_config=HnswConfigDiff(
                m=16,
            ),
        ) 

    def recreate_collection(self):
        if QDRANT_CLIENT.collection_exists(self.collection_name):
            QDRANT_CLIENT.delete_collection(self.collection_name)

        self._create_collection()

    def enable_hnsw_indexing(self):
        QDRANT_CLIENT.update_collection(
            collection_name=self.collection_name,
            hnsw_config=HnswConfigDiff(
                m=16,
            ),
        ) 

    def disable_hnsw_indexing(self):
        QDRANT_CLIENT.update_collection(
            collection_name=self.collection_name,
            hnsw_config=HnswConfigDiff(
                m=0,
            ),
        ) 

    def get_collection_info(self):
        return QDRANT_CLIENT.get_collection(self.collection_name)

    def _embed_contents(self, chunk_contents):
        dense_embeddings = self.dense_embedding_model.encode(chunk_contents)
        sparse_embeddings = self.sparse_embedding_model.passage_embed(chunk_contents)

        return dense_embeddings, sparse_embeddings
        
    def _upsert_data(self, points):
        BATCH_SIZE = 100
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]
            QDRANT_CLIENT.upsert(
                collection_name=self.collection_name,
                points=batch
            )

    def insert_data(self, payload_keys, payload_values, embedding_indices=[0]):
        if not payload_values:
            return
        
        if isinstance(payload_values[0], str):
            contents = payload_values
        elif isinstance(payload_values[0], list):
            contents = [
                "\n".join(str(v[i]) for i in embedding_indices)
                for v in payload_values
            ]
        else:
            raise ValueError("Unsupported payload_values format")

        dense_embeddings, sparse_embeddings = self._embed_contents(contents)

        self.upsert_data(
                dense_embeddings=dense_embeddings,
                sparse_embeddings=sparse_embeddings,
                payload_keys=payload_keys,
                payload_values=payload_values
            )

    def upsert_data(self, dense_embeddings, sparse_embeddings, payload_keys, payload_values):
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense_vector": dense_embeddings[i],
                    "sparse_vector": sparse_embeddings[i],
                },
                payload=dict(zip(payload_keys, payload_values[i] if isinstance(payload_values[i], list) else [payload_values[i]]))
            )
            for i in range(len(dense_embeddings))
        ]

        self._upsert_data(points)

    def _search_dense(self, query, top_k):
        query_vector = self.dense_embedding_model.encode(query)
    
        results = QDRANT_CLIENT.query_points(
            collection_name=self.collection_name,
            query=query_vector,  
            using="dense_vector",  
            with_payload=True,
            limit=top_k,
        )
        
        return results.points 
    
    def search_dense(self, query, top_k=3, threshold=0.3):
        results = self._search_dense(query, top_k)
        return [point for point in results if point.score >= threshold]
    
    def _search_sparse(self, query, top_k):
        sparse_vector_dict = self.sparse_embedding_model.passage_embed(query)

        sparse_vector = SparseVector(
            indices=sparse_vector_dict["indices"],
            values=sparse_vector_dict["values"]
        )

        results = QDRANT_CLIENT.query_points(
            collection_name=self.collection_name,
            query=sparse_vector,  
            using="sparse_vector",  
            with_payload=True,
            limit=top_k,
        )
        return results.points
    
    def search_sparse(self, query, top_k=3, threshold=0.3):
        results = self._search_sparse(query, top_k)
        return [point for point in results if point.score >= threshold]

    def hybrid_search(self, query, top_k=3, threshold=0.3, alpha=0.7):
        dense_results = self._search_dense(query, top_k * 3 + 1)
        dense_scores = normalize([point.score for point in dense_results])
        for i, point in enumerate(dense_results):
            point.score = dense_scores[i]

        sparse_results = self._search_sparse(query, top_k * 3 + 1)
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
        for _, entry in combined.items():
            score = alpha * entry["dense"] + (1 - alpha) * entry["sparse"]
            if score >= threshold:
                entry['point'].score = score
                final.append(entry["point"])

        final = sorted(final, key=lambda d: d.score, reverse=True)[:top_k]
        return final
    
    def get_similar_vectors(self, text, top_k=3, threshold=0.85):
        search_result = self.search_dense(text, top_k=top_k, threshold=threshold)

        similar_articles = [
            (hit.id, hit.score) for hit in search_result 
        ]

        return similar_articles
