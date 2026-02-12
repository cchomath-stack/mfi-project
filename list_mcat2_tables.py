import psycopg2
DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"
conn = psycopg2.connect(DB_URL)
cur = conn.cursor()
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'mcat2'")
print('\n'.join([r[0] for r in cur.fetchall()]))
cur.close()
conn.close()
