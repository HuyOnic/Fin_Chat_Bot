class DenseRetriever:
    def __init__(self, qdrant_client, vectorizer):
        self.client = qdrant_client 
        self.vectorizer = vectorizer

    def retrieve(self, question, top_k=3, threshold=0.85):
        vector = self.vectorizer.encode(question)
        if not isinstance(vector, list):
            vector = vector.tolist()
        search_result = self.client.search(
            collection_name="news_vectors",
            query_vector=vector,
            limit=top_k
        )
        return search_result