import psycopg2
DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"
conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

# Get some sample IDs from question_render
cur.execute("SELECT question_id FROM mcat2.question_render LIMIT 10")
ids = [str(r[0]) for r in cur.fetchall()]

for qid in ids:
    cur.execute("SELECT source FROM mcat2.user_extracted_questions WHERE question_id = %s", (qid,))
    res = cur.fetchone()
    if res:
        print(f"Found match for {qid}: {res[0]}")
    else:
        # Try substring match
        short_id = qid[:6]
        cur.execute("SELECT question_id, source FROM mcat2.user_extracted_questions WHERE question_id LIKE %s", (f"%{short_id}%",))
        res = cur.fetchall()
        if res:
            print(f"Substring match for {short_id}: {res}")

cur.close()
conn.close()
