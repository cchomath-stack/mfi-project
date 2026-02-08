import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def check_schema():
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        # mcat2.problems_problem 테이블의 모든 컬럼 이름만 출력
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns 
            WHERE table_schema = 'mcat2' AND table_name = 'problems_problem'
            ORDER BY column_name;
        """)
        columns = cur.fetchall()
        print("--- Columns in mcat2.problems_problem ---")
        for col in columns:
            print(col[0])

if __name__ == "__main__":
    check_schema()
