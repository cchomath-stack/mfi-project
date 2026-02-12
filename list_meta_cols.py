import psycopg2
DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"
conn = psycopg2.connect(DB_URL)
cur = conn.cursor()
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'question_metadata' AND table_schema = 'mcat2'")
for r in cur.fetchall(): print(r[0])
cur.close()
conn.close()
