import psycopg2
DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"
conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

print("--- Samples from question_render ---")
cur.execute("SELECT question_id, preview_url FROM mcat2.question_render LIMIT 3")
q_rows = cur.fetchall()
for r in q_rows: print(r)

print("\n--- Samples from problems_problem ---")
cur.execute("SELECT id, problem_image_url FROM mcat2.problems_problem LIMIT 3")
p_rows = cur.fetchall()
for r in p_rows: print(r)

# Check if a question_id from question_render exists in problems_problem
if q_rows:
    qid = q_rows[0][0]
    cur.execute("SELECT id FROM mcat2.problems_problem WHERE id = %s", (qid,))
    res = cur.fetchone()
    print(f"\nDoes {qid} exist in problems_problem? {res is not None}")

cur.close()
conn.close()
