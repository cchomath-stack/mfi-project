import psycopg2
DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"
conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

# Get some sample IDs from question_render
cur.execute("SELECT question_id FROM mcat2.question_render LIMIT 10")
ids = [str(r[0]) for r in cur.fetchall()]

for qid in ids:
    cur.execute("""
        SELECT d.title
        FROM mcat2.question_sources qs
        JOIN mcat2.documents d ON qs.document_id = d.id
        WHERE qs.question_id = %s
    """, (qid,))
    res = cur.fetchone()
    if res:
        print(f"Doc Title for {qid}: {res[0]}")
    else:
        # Fallback to questions table
        cur.execute("SELECT origin_file_name FROM mcat2.questions WHERE id = %s", (qid,))
        res2 = cur.fetchone()
        if res2:
            print(f"Question Filename for {qid}: {res2[0]}")

cur.close()
conn.close()
