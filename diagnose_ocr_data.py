import psycopg2
import config

def diagnose():
    try:
        conn = psycopg2.connect(config.DB_URL)
        cur = conn.cursor()
        
        print("--- Checking mcat2.question_image_embeddings ---")
        cur.execute("SELECT COUNT(*) FROM mcat2.question_image_embeddings")
        total = cur.fetchone()[0]
        print(f"Total rows in embeddings: {total}")
        
        print("\n--- Checking mcat2.question_render ---")
        cur.execute("SELECT COUNT(*) FROM mcat2.question_render WHERE preview_url IS NOT NULL AND preview_url != ''")
        total_render = cur.fetchone()[0]
        print(f"Total rows in render with preview_url: {total_render}")
        
        cur.execute("SELECT COUNT(*) FROM mcat2.question_image_embeddings WHERE ocr_text IS NOT NULL AND ocr_text != ''")
        with_ocr = cur.fetchone()[0]
        print(f"Rows with OCR text: {with_ocr}")
        
        if with_ocr > 0:
            print("\n--- Sample OCR text (First 5) ---")
            cur.execute("SELECT question_id, ocr_text FROM mcat2.question_image_embeddings WHERE ocr_text IS NOT NULL AND ocr_text != '' LIMIT 5")
            for row in cur.fetchall():
                print(f"ID: {row[0]} | OCR: {row[1][:50]}...")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    diagnose()
