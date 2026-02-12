import psycopg2
DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"
conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

print("--- Samples from question_render ---")
cur.execute("SELECT question_id FROM mcat2.question_render LIMIT 3")
q_rows = cur.fetchall()
for r in q_rows: print(r)

print("\n--- Samples from questions ---")
cur.execute("SELECT id FROM mcat2.questions LIMIT 3")
p_rows = cur.fetchall()
for r in p_rows: print(r)

# Check mapping
if q_rows:
    qid = q_rows[0][0]
    cur.execute("SELECT origin_file_name FROM mcat2.questions WHERE id = %s", (qid,))
    res = cur.fetchone()
    print(f"\nDoes {qid} exist in questions? {res is not None} (File: {res[0] if res else 'N/A'})")

cur.close()
conn.close()
