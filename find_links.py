import psycopg2
from config import DB_URL

def find_links():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    try:
        # question_id가 포함된 다른 테이블들 검색
        cur.execute("""
            SELECT table_name 
            FROM information_schema.columns 
            WHERE column_name = 'question_id' AND table_schema = 'mcat2'
        """)
        tables = [row[0] for row in cur.fetchall()]
        print(f"Tables containing 'question_id': {tables}")
        
        # 각 테이블의 샘플 데이터 및 컬럼 확인
        for table in tables:
            if table == 'question_render': continue
            cur.execute(f"SELECT * FROM mcat2.{table} LIMIT 1")
            colnames = [desc[0] for desc in cur.description]
            print(f" - Table mcat2.{table} columns: {colnames}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    find_links()
