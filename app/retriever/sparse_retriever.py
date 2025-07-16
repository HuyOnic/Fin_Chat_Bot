from sqlalchemy import text

class SparseRetriever:
    def __init__(self, pg_session):
        self.session = pg_session

    def retrieve(self, query: str, top_k: int = 5):
        sql = text("""
        SELECT id, content, ts_rank(to_tsvector(content), plainto_tsquery(:q)) AS rank
        FROM public.news
        WHERE to_tsvector(content) @@ plainto_tsquery(:q)
        ORDER BY rank DESC
        LIMIT :top_k;
        """)
        return self.session.execute(sql, {"q": query, "top_k": top_k}).fetchall()
