import psycopg2

DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"

def diagnose():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # 1. 임베딩이 존재하는 문항 5개 가져오기
    cur.execute("SELECT problem_id FROM mcat2.problem_image_embeddings LIMIT 5")
    pids = [r[0] for r in cur.fetchall()]
    print(f"Testing Problem IDs: {pids}")
    
    for pid in pids:
        print(f"\n--- Diagnosing PID: {pid} ---")
        
        # Junction 확인
        cur.execute("SELECT * FROM mcat2.problems_problem_with_source WHERE problem_id = %s", (pid,))
        pws_row = cur.fetchone()
        if pws_row:
            print(f"  [Found in junction] source_id: {pws_row[4]}")
            sid = pws_row[4]
            
            # Source 확인
            cur.execute("SELECT id, title FROM mcat2.problems_source WHERE id = %s", (sid,))
            src_row = cur.fetchone()
            if src_row:
                print(f"  [Found in source] title: {src_row[1]}")
            else:
                print(f"  [NOT Found in source] No row with id {sid}")
        else:
            print("  [NOT Found in junction] No mapping in problems_problem_with_source")
            
            # 다른 가능성: problems_problem_source 테이블 (이전에 컬럼이 안보였지만 다시 확인)
            try:
                cur.execute(f"SELECT * FROM mcat2.problems_problem_source WHERE problem_id = %s", (pid,))
                pps_row = cur.fetchone()
                if pps_row:
                     print(f"  [Found in alternative junction problems_problem_source] row: {pps_row}")
            except:
                pass

    cur.close()
    conn.close()

if __name__ == "__main__":
    diagnose()
