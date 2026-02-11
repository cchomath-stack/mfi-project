import psycopg2
from datetime import datetime

DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"

def check_status():
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        
        # 1. Total processed items
        cur.execute("SELECT COUNT(*) FROM mcat2.question_image_embeddings")
        processed_count = cur.fetchone()[0]
        
        # 2. Total items to process
        cur.execute("SELECT COUNT(*) FROM mcat2.question_render WHERE preview_url IS NOT NULL AND preview_url != ''")
        total_count = cur.fetchone()[0]
        
        # 3. Last updated item time
        cur.execute("SELECT MAX(updated_at) FROM mcat2.question_image_embeddings")
        last_updated = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        print(f"--- Indexing Status Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
        print(f"Processed: {processed_count:,} / {total_count:,} ({processed_count/total_count*100:.2f}%)")
        print(f"Remaining: {total_count - processed_count:,}")
        print(f"Last heartbeat: {last_updated}")
        
    except Exception as e:
        print(f"Error connecting to DB: {e}")

if __name__ == "__main__":
    check_status()
