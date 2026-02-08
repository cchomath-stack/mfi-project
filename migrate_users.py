import sqlite3
import os

DB_PATH = "C:\\Users\\user\\Desktop\\antigravity\\image_search_v2\\users.db"

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Error: {DB_PATH} not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("Checking current columns in web_users...")
    cursor.execute("PRAGMA table_info(web_users)")
    columns = [col[1] for col in cursor.fetchall()]

    # 추가할 컬럼들
    new_columns = [
        ("google_id", "TEXT"), # SQLite ADD COLUMN은 UNIQUE 제약 조건을 직접 추가할 수 없음
        ("email", "TEXT"),
        ("picture_url", "TEXT"),
        ("is_approved", "INTEGER DEFAULT 0") # 기본값 0 (승인 대기)
    ]

    for col_name, col_def in new_columns:
        if col_name not in columns:
            print(f"Adding column: {col_name}...")
            cursor.execute(f"ALTER TABLE web_users ADD COLUMN {col_name} {col_def}")
        else:
            print(f"Column {col_name} already exists.")

    # 기존 admin 계정은 자동으로 승인 처리
    print("Ensuring admin user is approved...")
    cursor.execute("UPDATE web_users SET is_approved = 1 WHERE username = 'admin'")

    conn.commit()
    conn.close()
    print("Migration completed successfully.")

if __name__ == "__main__":
    migrate()
