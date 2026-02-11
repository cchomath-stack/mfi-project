import psycopg2
from config import DB_URL

def create_new_table():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    try:
        # 1. 기존 테이블이 있다면 백업하거나 새로 생성 (여기선 새로 생성)
        print("Creating table: mcat2.question_image_embeddings")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS mcat2.question_image_embeddings (
                id SERIAL PRIMARY KEY,
                question_id UUID UNIQUE NOT NULL,
                image_embedding float8[], -- Vector data (float8[] for compatibility)
                ocr_text TEXT,
                meta_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. 성능을 위한 인덱스 추가
        cur.execute("CREATE INDEX IF NOT EXISTS idx_question_id ON mcat2.question_image_embeddings(question_id)")
        
        conn.commit()
        print("Table and indexes created successfully.")
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    create_new_table()
