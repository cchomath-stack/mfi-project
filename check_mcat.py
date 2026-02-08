import psycopg2

DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"

def check_schemas_and_tables():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # 1. 스키마 목록 확인
    print("--- Schemas ---")
    cur.execute("SELECT schema_name FROM information_schema.schemata")
    schemas = [r[0] for r in cur.fetchall()]
    print(f"Schemas: {schemas}")
    
    # 2. mcat 및 mcat2 스키마의 테이블 확인
    for schema in ['mcat', 'mcat2']:
        if schema in schemas:
            print(f"\n--- Tables in {schema} ---")
            cur.execute(f"SELECT table_name FROM information_schema.tables WHERE table_schema='{schema}'")
            tables = [r[0] for r in cur.fetchall()]
            for t in tables:
                print(f" - {t}")
        else:
            print(f"\nSchema '{schema}' does not exist.")
            
    cur.close()
    conn.close()

if __name__ == "__main__":
    check_schemas_and_tables()
