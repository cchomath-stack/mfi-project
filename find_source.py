import psycopg2

DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"

def find_source_path():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # 1. 샘플 데이터 하나 가져오기 (가장 최근 임베딩된 문항 중 하나)
    cur.execute("SELECT problem_id FROM mcat2.problem_image_embeddings LIMIT 1")
    sample_id = cur.fetchone()[0]
    print(f"Sample Problem ID: {sample_id}")
    
    # 2. 이 문항이 어디에 연결되어 있는지 모든 테이블 뒤지기
    cur.execute("""
        SELECT table_name, column_name 
        FROM information_schema.columns 
        WHERE table_schema = 'mcat2' AND column_name LIKE '%problem_id%'
    """)
    possible_tables = cur.fetchall()
    
    print("\n--- Searching for connections ---")
    for table, col in possible_tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM mcat2.{table} WHERE {col} = %s", (sample_id,))
            count = cur.fetchone()[0]
            if count > 0:
                print(f"Found in mcat2.{table}: {count} rows")
        except:
            pass
            
    cur.close()
    conn.close()

if __name__ == "__main__":
    find_source_path()
