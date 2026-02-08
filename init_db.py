import psycopg2
from passlib.context import CryptContext
import uuid

# DB 연결 정보
DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"

# 비밀번호 해싱 설정
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def setup_db():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # 1. 테이블 생성
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
    
    # 2. 초기 어드민 계정 생성 (없을 경우에만)
    admin_username = "admin"
    admin_password = "admin123!" # 초기 비밀번호
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
    cur.close()
    conn.close()
    print("Database setup completed.")

if __name__ == "__main__":
    setup_db()
