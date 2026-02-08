import psycopg2

DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"

def check_pps():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # problems_problem_source 테이블의 실제 데이터 샘플과 컬럼명 확인
    try:
        cur.execute("SELECT * FROM mcat2.problems_problem_source LIMIT 1")
        row = cur.fetchone()
        cols = [desc[0] for desc in cur.description]
        print(f"problems_problem_source Columns: {cols}")
        print(f"Sample Data: {row}")
    except Exception as e:
        print(f"Error checking problems_problem_source: {e}")
        
    cur.close()
    conn.close()

if __name__ == "__main__":
    check_pps()
