import psycopg2
from config import DB_URL

def sample_data():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    try:
        # question_render에서 샘플 5건 추출
        cur.execute("""
            SELECT question_id, preview_url, question_katex 
            FROM mcat2.question_render 
            ORDER BY created_at DESC 
            LIMIT 5
        """)
        rows = cur.fetchall()
        print("Sample data from mcat2.question_render:")
        for row in rows:
            print(f"ID: {row[0]}")
            print(f"URL: {row[1]}")
            print(f"KaTeX: {row[2][:100]}..." if row[2] else "KaTeX: None")
            print("-" * 20)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    sample_data()
