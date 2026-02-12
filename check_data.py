import psycopg2
DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"
conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

print("--- user_extracted_questions ---")
cur.execute("SELECT question_id, source FROM mcat2.user_extracted_questions LIMIT 3")
for r in cur.fetchall(): print(r)

print("\n--- problems_source ---")
cur.execute("SELECT id, title FROM mcat2.problems_source LIMIT 3")
for r in cur.fetchall(): print(r)

print("\n--- problems_problem_with_source ---")
cur.execute("SELECT problem_id, source_id FROM mcat2.problems_problem_with_source LIMIT 3")
for r in cur.fetchall(): print(r)

cur.close()
conn.close()
