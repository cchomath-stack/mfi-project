import psycopg2

DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"

def inspect_db():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    print("--- Schemas ---")
    cur.execute("SELECT schema_name FROM information_schema.schemata;")
    for schema in cur.fetchall():
        print(f" - {schema[0]}")
        
    print("\n--- Tables in mcat2 ---")
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'mcat2';")
    for table in cur.fetchall():
        print(f" - {table[0]}")
        
    print("\n--- Check Permissions (CREATE) on mcat2 ---")
    cur.execute("SELECT has_schema_privilege('db_member4', 'mcat2', 'CREATE');")
    print(f" - Can create in mcat2: {cur.fetchone()[0]}")

    print("\n--- Check Permissions (CREATE) on public ---")
    cur.execute("SELECT has_schema_privilege('db_member4', 'public', 'CREATE');")
    print(f" - Can create in public: {cur.fetchone()[0]}")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    inspect_db()
