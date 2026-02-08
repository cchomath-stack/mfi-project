import psycopg2

DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"

def fix_db():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # 1. mcat2 스키마 존재 확인 및 생성
    print("Checking/Creating schema mcat2...")
    cur.execute("CREATE SCHEMA IF NOT EXISTS mcat2;")
    
    # 2. web_users 테이블 생성
    print("Creating table mcat2.web_users...")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS mcat2.web_users (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        username VARCHAR(50) UNIQUE NOT NULL,
        hashed_password VARCHAR(255) NOT NULL,
        full_name VARCHAR(100),
        role VARCHAR(20) DEFAULT 'user',
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 3. 어드민 계정 확인 및 생성
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    admin_username = "admin"
    admin_password = "admin123!"
    hashed_password = pwd_context.hash(admin_password)
    
    cur.execute("SELECT id FROM mcat2.web_users WHERE username = %s", (admin_username,))
    if not cur.fetchone():
        cur.execute("""
        INSERT INTO mcat2.web_users (username, hashed_password, full_name, role)
        VALUES (%s, %s, %s, %s)
        """, (admin_username, hashed_password, "System Administrator", "admin"))
        print(f"Admin account '{admin_username}' created.")
    else:
        print(f"Admin account '{admin_username}' already exists.")
    
    conn.commit()
    
    # 4. 생성 확인
    cur.execute("SELECT count(*) FROM mcat2.web_users")
    count = cur.fetchone()[0]
    print(f"Total users in mcat2.web_users: {count}")
    
    cur.close()
    conn.close()
    print("Database fix completed.")

if __name__ == "__main__":
    fix_db()
