import psycopg2

DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"

def check():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    tables = [
        'problems_problem', 
        'problems_problem_source', 
        'problems_problem_with_source', 
        'problems_source', 
        'problem_image_embeddings'
    ]
    
    for table in tables:
        print(f"\n--- {table} ---")
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_schema='mcat2' AND table_name='{table}'")
        for row in cur.fetchall():
            print(f" - {row[0]}")
            
    cur.close()
    conn.close()

if __name__ == "__main__":
    check()
