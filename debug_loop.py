import psycopg2
import config

conn = psycopg2.connect(config.DB_URL)
cur = conn.cursor()

print("--- Items matching 'incomplete' criteria ---")
cur.execute("""
    SELECT q.question_id, q.preview_url, e.ocr_text
    FROM mcat2.question_render q
    LEFT JOIN mcat2.question_image_embeddings e ON q.question_id = e.question_id
    WHERE (e.question_id IS NULL OR e.ocr_text IS NULL OR e.ocr_text = '' OR e.ocr_text LIKE '%%?%%')
      AND q.preview_url IS NOT NULL AND q.preview_url != ''
    ORDER BY q.updated_at DESC
    LIMIT 5
""")

rows = cur.fetchall()
for r in rows:
    print(f"ID: {r[0]}")
    print(f"URL: {r[1]}")
    print(f"OCR: '{r[2]}'")
    print("-" * 20)

cur.close()
conn.close()
