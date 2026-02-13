import psycopg2
import config

def check_stats():
    try:
        conn = psycopg2.connect(config.DB_URL)
        cur = conn.cursor()
        
        # 전체 유효 문항 수
        cur.execute("SELECT COUNT(*) FROM mcat2.question_render WHERE preview_url IS NOT NULL AND preview_url != ''")
        total = cur.fetchone()[0]
        
        # OCR 데이터가 있는 수
        cur.execute("SELECT COUNT(*) FROM mcat2.question_image_embeddings WHERE ocr_text IS NOT NULL AND ocr_text != ''")
        has_ocr = cur.fetchone()[0]
        
        # OCR 데이터가 '?' 를 포함하거나 빈값인 경우 (실패 혹은 불완전)
        cur.execute("SELECT COUNT(*) FROM mcat2.question_image_embeddings WHERE ocr_text LIKE '%?%' OR ocr_text = ''")
        fail_ocr = cur.fetchone()[0]
        
        print(f"Total valid questions: {total}")
        print(f"Questions with OCR: {has_ocr}")
        print(f"Questions with failed/incomplete OCR: {fail_ocr}")
        print(f"Pending OCR (no record in embeddings table): {total - has_ocr - fail_ocr}")
        
        # 최근 5개 OCR 결과 샘플
        print("\n--- Recent OCR Samples ---")
        cur.execute("""
            SELECT q.question_id, e.ocr_text 
            FROM mcat2.question_render q
            JOIN mcat2.question_image_embeddings e ON q.question_id = e.question_id
            WHERE e.ocr_text IS NOT NULL AND e.ocr_text != ''
            ORDER BY e.updated_at DESC
            LIMIT 5
        """)
        for row in cur.fetchall():
            print(f"ID: {row[0]} | OCR: {row[1][:100]}...")
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_stats()
