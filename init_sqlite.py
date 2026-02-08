import sqlite3
import os
from passlib.context import CryptContext

# SQLite DB 경로
DB_PATH = "users.db"

# 비밀번호 해싱 설정
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def setup_sqlite():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # 1. 테이블 생성
    cur.execute("""
    CREATE TABLE IF NOT EXISTS web_users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        full_name TEXT,
        role TEXT DEFAULT 'user',
        is_active INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 2. 초기 어드민 계정 생성
    admin_username = "admin"
    admin_password = "admin123!"
    hashed_password = pwd_context.hash(admin_password)
    
    cur.execute("SELECT id FROM web_users WHERE username = ?", (admin_username,))
    if not cur.fetchone():
        import uuid
        cur.execute("""
        INSERT INTO web_users (id, username, hashed_password, full_name, role)
        VALUES (?, ?, ?, ?, ?)
        """, (str(uuid.uuid4()), admin_username, hashed_password, "System Administrator", "admin"))
        print(f"Admin account '{admin_username}' created in SQLite.")
    else:
        print(f"Admin account '{admin_username}' already exists in SQLite.")
    
    conn.commit()
    conn.close()
    print(f"SQLite database setup completed at {DB_PATH}.")

if __name__ == "__main__":
    setup_sqlite()
