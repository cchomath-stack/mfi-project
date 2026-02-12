import psycopg2
DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"
conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

print("--- question_metadata ---")
try:
    cur.execute("SELECT question_id, source_name FROM mcat2.question_metadata LIMIT 5")
    for r in cur.fetchall(): print(r)
except Exception as e:
    print(f"Error: {e}")
    conn.rollback()

print("\n--- question_sources ---")
try:
    cur.execute("SELECT * FROM mcat2.question_sources LIMIT 5")
    for r in cur.fetchall(): print(r)
except Exception as e:
    print(f"Error: {e}")
    conn.rollback()

cur.close()
conn.close()
