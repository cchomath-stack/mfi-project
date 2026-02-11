import psycopg2
from config import DB_URL

def get_schema(table_name):
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'mcat2' AND table_name = '{table_name}'")
        cols = cur.fetchall()
        print(f"Schema for mcat2.{table_name}:")
        for col in cols:
            print(f" - {col[0]} ({col[1]})")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    get_schema("question_render")
    print("-" * 20)
    get_schema("user_extracted_questions")
    print("-" * 20)
    get_schema("problem_image_embeddings")
