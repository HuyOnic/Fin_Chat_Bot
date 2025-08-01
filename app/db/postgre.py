import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import uuid
# Load biến môi trường từ file .env
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Tạo db nếu chưa tồn tại
def create_database():
    conn = get_pg_connection()
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute(f"SELECT 1 FROM pg_database WHERE datname='{DB_NAME}'")
    exists = cursor.fetchone()
    if not exists:
        cursor.execute(sql.SQL("CREATE DATABASE {};").format(sql.Identifier(DB_NAME)))
        print("Đã tạo cơ sở dữ liệu.")
    else:
        print("Cơ sở dữ liệu đã tồn tại.")
    cursor.close()
    conn.close()

# Tạo bảng news nếu chưa tồn tại
def create_news_table():    
    conn = get_pg_connection()
    cursor = conn.cursor()
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS news (
        id SERIAL PRIMARY KEY,
        news_type INTEGER NOT NULL CHECK (news_type IN (1, 2, 3, 4)),
        sec_cd VARCHAR(20),
        market_cd INTEGER,
        content TEXT NOT NULL,
        score NUMERIC(5, 2),
        manual_score NUMERIC(5, 2),
        source VARCHAR(255),
        news_date BIGINT NOT NULL,
        status INTEGER NOT NULL CHECK (status IN (0, 1, 2, 9)),
        remarks TEXT,
        reg_date_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    '''
    cursor.execute(create_table_query)
    conn.commit()
    cursor.close()
    conn.close()
    print("Bảng 'news' đã được khởi tạo.")

    
# Thêm data vào bảng news sau khi crawl
def insert_news(data_list):
    conn = get_pg_connection()
    cursor = conn.cursor()
    insert_query = '''
    INSERT INTO public.news (news_type, sec_cd, market_cd, content, source, news_date, status)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    '''
    for data in data_list:
        cursor.execute(insert_query, (
            data['news_type'],
            data['sec_cd'],
            data['market_cd'],
            data['content'],
            data['source'],
            data['news_date'],
            0 
        ))
    conn.commit()
    cursor.close()
    conn.close()
    print("Đã thêm dữ liệu tin tức.")
    

def get_all_pending_preprocess():
    conn = get_pg_connection()
    cursor = conn.cursor()
    select_query = '''
    SELECT id, content, news_date, source, status FROM news ORDER BY news_date ASC
    '''
    cursor.execute(select_query)
    all_data = cursor.fetchall()
    
    if all_data is not None:
        print(f"Tìm thấy {len(all_data)} bài viết để kiểm tra.")
    else:
        print("Không có bài viết nào cần kiểm tra.")
        cursor.close()
        conn.close()
        return None
    
    return all_data

def get_all_pending_sentiment():
    conn = get_pg_connection()
    cursor = conn.cursor()
    select_query = "SELECT id, content FROM news WHERE status = 1 ORDER BY news_date ASC"
    cursor.execute(select_query)
    all_data = cursor.fetchall()
    cursor.close()
    conn.close()
    return all_data
    
def update_status(current_id, status_number):
    conn = get_pg_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE news SET status = %s WHERE id = %s", (status_number, current_id,))
    conn.commit()
    cursor.close()
    conn.close()
    
def update_sentiment_score(article_id, score):
    conn = get_pg_connection()
    cursor = conn.cursor()
    update_query = "UPDATE news SET score = %s WHERE id = %s"
    cursor.execute(update_query, (score, article_id))
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Đã cập nhật sentiment score ({score}) cho bài viết ID {article_id}")

def get_pg_connection():
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)

def fetch_news_by_ids(ids: list[int]) -> list[dict]:
    """
    Truy vấn danh sách bản tin từ bảng `news` theo id.
    
    :param ids: List[int] các ID bài viết.
    :return: List[dict] chứa content và metadata.
    """
    if not ids:
        return []
    
    query = """
        SELECT id, content, news_type, sec_cd, market_cd, news_date, score, manual_score, source
        FROM news
        WHERE id = ANY(%s);
    """

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (ids,))
            results = cur.fetchall()

    columns = [
    "id", "content", "news_type", "sec_cd", "market_cd",
    "news_date", "score", "manual_score", "source"
    ]
    results = [dict(zip(columns, row)) for row in results]

    return results

def fetch_newest_info(news_date: int):
    query = """
        SELECT content, source, news_date
        FROM news 
        WHERE news_date >= %s
    """
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (news_date,))
            results = cur.fetchall()
    columns = ["content", "source", "news_date"]
    results = [dict(zip(columns, row)) for row in results]
    return results

def get_last_news_id():
    conn = get_pg_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM news ORDER BY id DESC LIMIT 1;")
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result[0] if result else None



if __name__=="__main__":
    pass