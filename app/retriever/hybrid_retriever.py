from typing import List, Dict, Any

class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, alpha: float = 0.5):
        """
        :param dense_retriever: Đối tượng có phương thức .retrieve(query, top_k) → List
        :param sparse_retriever: Tương tự, nhưng cho keyword search
        :param alpha: trọng số pha trộn (0 = chỉ sparse, 1 = chỉ dense)
        """
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.alpha = alpha

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        dense_results = self.dense.retrieve(query, top_k * 2)
        sparse_results = self.sparse.retrieve(query, top_k * 2)

        combined = self._merge_scores(dense_results, sparse_results, top_k)
        return combined

    def _merge_scores(self, dense, sparse, top_k: int) -> List[Dict[str, Any]]:
        scores = {}
        doc_map = {}

        for doc in dense:
            doc_id = doc.id
            score = doc.score or 0
            scores[doc_id] = self.alpha * score
            doc_map[doc_id] = doc

        for doc in sparse:
            doc_id = doc[0]
            score = doc[2] or 0
            scores[doc_id] = scores.get(doc_id, 0) + (1 - self.alpha) * score
            doc_map[doc_id] = doc_map.get(doc_id) or doc

        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = [doc_map[doc_id] for doc_id, _ in sorted_ids[:top_k]]
        return top_docs
