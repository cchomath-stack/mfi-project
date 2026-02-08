import psycopg2

DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"

def check_raw_data():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # 1. 출처가 비어있는 문항들에 연결된 실제 source 데이터 확인
    cur.execute("""
        SELECT s.* 
        FROM mcat2.problems_source s 
        JOIN mcat2.problems_problem_with_source pws ON s.id = pws.source_id 
        WHERE pws.problem_id IN (SELECT problem_id FROM mcat2.problem_image_embeddings LIMIT 10)
    """)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    
    print(f"--- Data sample from problems_source (10 items) ---")
    for row in rows:
        data = dict(zip(cols, row))
        # 의미 있는 데이터만 출력
        print({k: v for k, v in data.items() if v is not None and v != ""})
        
    cur.close()
    conn.close()

if __name__ == "__main__":
    check_raw_data()
