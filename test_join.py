import psycopg2
DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"
conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

# Get some sample IDs from question_render
cur.execute("SELECT question_id FROM mcat2.question_render LIMIT 5")
ids = [str(r[0]) for r in cur.fetchall()]
print(f"Testing IDs: {ids}")

# Try joining with problems_source
cur.execute("""
    SELECT q.question_id, s.title
    FROM mcat2.question_render q
    LEFT JOIN mcat2.problems_problem_with_source pws ON q.question_id = pws.problem_id
    LEFT JOIN mcat2.problems_source s ON pws.source_id = s.id
    WHERE q.question_id = ANY(%s::uuid[])
""", (ids,))

for r in cur.fetchall():
    print(r)

cur.close()
conn.close()
