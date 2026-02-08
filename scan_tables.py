import psycopg2

DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"

def scan_tables():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='mcat2'")
    tables = [r[0] for r in cur.fetchall()]
    
    print(f"--- Tables in mcat2 and record counts ---")
    for table in tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM mcat2.{table}")
            count = cur.fetchone()[0]
            print(f" - {table}: {count} rows")
        except:
            print(f" - {table}: (Error or No permissions)")
            conn.rollback()
            
    cur.close()
    conn.close()

if __name__ == "__main__":
    scan_tables()
