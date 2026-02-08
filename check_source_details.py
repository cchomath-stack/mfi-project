import psycopg2

DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"

def check_source_details():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # 임베딩된 문항들에 연결된 출처 데이터의 여러 필드를 확인
    cur.execute("""
        SELECT s.id, s.title, s.origin_filename, s.exam_year, s.exam_month, s.category1
        FROM mcat2.problems_source s 
        JOIN mcat2.problems_problem_with_source pws ON s.id = pws.source_id 
        WHERE pws.problem_id IN (SELECT problem_id FROM mcat2.problem_image_embeddings LIMIT 20)
    """)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    
    print(f"--- Source Data Details (Sample) ---")
    for row in rows:
        data = dict(zip(cols, row))
        print(data)
        
    cur.close()
    conn.close()

if __name__ == "__main__":
    check_source_details()
