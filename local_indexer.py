import os
import time
import torch
import numpy as np
import psycopg2
from PIL import Image
from io import BytesIO
import requests
from pix2text import Pix2Text
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# --- 설정 (서버 config.py 또는 직접 입력) ---
# 서버의 DB_URL을 그대로 사용하여 로컬에서 원격으로 데이터를 밀어넣습니다.
DB_URL = "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki"
MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f">>> Local Indexer started on {DEVICE}")
if DEVICE == "cpu":
    print(" [Warning] GPU (CUDA) not found. Speed will be very slow.")

# 1. 모델 로드
print(">>> Loading Models (Pix2Text, CLIP)...")
math_ocr = Pix2Text(languages=['en', 'ko'], mfr_config={'device': DEVICE})
clip_model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)

def get_db_conn():
    return psycopg2.connect(DB_URL)

def download_and_preprocess(qid_uuid, url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        # 고사양이므로 굳이 작게 리사이즈할 필요는 없으나 호환성을 위해 1024 유지
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        return str(qid_uuid), img
    except Exception as e:
        print(f" [Download Error] {qid_uuid}: {e}")
        return str(qid_uuid), None

def run_local_indexing():
    processed_count = 0
    batch_size = 20 # 5080이라면 배치 사이즈를 더 키워도 됩니다.
    
    print("\n>>> Start Indexing Loop...")
    
    try:
        while True:
            # 1. 대상 데이터 가져오기 (원격 DB에서 미처리 건 추출)
            conn = get_db_conn()
            cur = conn.cursor()
            cur.execute("""
                SELECT q.question_id, q.preview_url FROM mcat2.question_render q
                LEFT JOIN mcat2.question_image_embeddings e ON q.question_id = e.question_id
                WHERE e.question_id IS NULL
                  AND q.preview_url IS NOT NULL AND q.preview_url != '' LIMIT %s
            """, (batch_size,))
            rows = cur.fetchall()
            cur.close()
            conn.close()
            
            if not rows:
                print(">>> All items processed or no more items found.")
                break

            batch_start = time.time()
            
            # 2. 병렬 다운로드 (네트워크 병목 제거)
            with ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(lambda r: download_and_preprocess(r[0], r[1]), rows))
            
            valid_items = [(qid, img) for qid, img in results if img is not None]
            if not valid_items:
                continue

            qids = [x[0] for x in valid_items]
            imgs = [x[1] for x in valid_items]

            # 3. CLIP Embedding (Batch)
            inputs = clip_processor(images=imgs, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                out = clip_model.get_image_features(**inputs)
                feat = getattr(out, "image_embeds", getattr(out, "pooler_output", out))
                all_embs = feat.cpu().numpy()
                all_embs = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)

            # 4. Math OCR (Pix2Text)
            conn_u = get_db_conn()
            cur_u = conn_u.cursor()
            
            for i, qid in enumerate(qids):
                img = imgs[i]
                emb = all_embs[i]
                
                try:
                    # RTX 5080 파워!
                    outs = math_ocr.recognize(img)
                    ocr_text = "\n".join([out['text'] for out in outs])
                except Exception as e:
                    print(f" [OCR Error] {qid}: {e}")
                    ocr_text = ""

                # 5. DB 업로드 (원격 서버로 전송)
                cur_u.execute("""
                    INSERT INTO mcat2.question_image_embeddings (question_id, image_embedding, ocr_text, updated_at)
                    VALUES (%s, %s, %s, NOW()) ON CONFLICT (question_id) DO UPDATE 
                    SET image_embedding=EXCLUDED.image_embedding, ocr_text=EXCLUDED.ocr_text, updated_at=NOW()
                """, (qid, emb.tolist(), ocr_text))
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f" [Progress] {processed_count} items uploaded... (Last: {qid[:8]})")

            conn_u.commit()
            cur_u.close()
            conn_u.close()
            
            print(f" [Batch] {len(qids)} items done in {time.time()-batch_start:.2f}s")

    except KeyboardInterrupt:
        print("\n>>> Interrupted by user.")
    except Exception as e:
        print(f"\n>>> Critical Error: {e}")
    finally:
        print(f"\n>>> Local Indexing Finished. Total: {processed_count}")

if __name__ == "__main__":
    run_local_indexing()
