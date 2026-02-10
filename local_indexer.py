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
    batch_size = 20 # 5080 기준 (더 키워도 됨)
    
    print("\n>>> Start Indexing Loop...")
    
    try:
        # 0. 초기 통계 가져오기
        conn_init = get_db_conn()
        cur_init = conn_init.cursor()
        cur_init.execute("SELECT COUNT(*) FROM mcat2.question_render WHERE preview_url IS NOT NULL AND preview_url != ''")
        total_goal = cur_init.fetchone()[0]
        cur_init.execute("SELECT COUNT(*) FROM mcat2.question_image_embeddings")
        already_done = cur_init.fetchone()[0]
        cur_init.close(); conn_init.close()
        
        remaining_total = total_goal - already_done
        start_time = time.time()
        
        print(f" [Stats] Total Target: {total_goal:,} | Already Done: {already_done:,} | To Process: {remaining_total:,}")
        
        while True:
            # 1. 대상 데이터 가져오기
            conn = get_db_conn(); cur = conn.cursor()
            cur.execute("""
                SELECT q.question_id, q.preview_url FROM mcat2.question_render q
                LEFT JOIN mcat2.question_image_embeddings e ON q.question_id = e.question_id
                WHERE e.question_id IS NULL
                  AND q.preview_url IS NOT NULL AND q.preview_url != '' LIMIT %s
            """, (batch_size,))
            rows = cur.fetchall(); cur.close(); conn.close()
            
            if not rows:
                print("\n>>> [Success] All items processed or no more items found.")
                break

            batch_start = time.time()
            
            # 2. 병렬 다운로드
            with ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(lambda r: download_and_preprocess(r[0], r[1]), rows))
            
            valid_items = [(qid, img) for qid, img in results if img is not None]
            if not valid_items: continue

            qids = [x[0] for x in valid_items]; imgs = [x[1] for x in valid_items]

            # 3. CLIP Embedding
            inputs = clip_processor(images=imgs, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                out = clip_model.get_image_features(**inputs)
                feat = getattr(out, "image_embeds", getattr(out, "pooler_output", out))
                all_embs = feat.cpu().numpy()
                all_embs = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)

            # 4. Math OCR & DB Upload
            conn_u = get_db_conn(); cur_u = conn_u.cursor()
            for i, qid in enumerate(qids):
                try: 
                    outs = math_ocr.recognize(imgs[i])
                    ocr_text = "\n".join([out['text'] for out in outs])
                except: ocr_text = ""

                cur_u.execute("""
                    INSERT INTO mcat2.question_image_embeddings (question_id, image_embedding, ocr_text, updated_at)
                    VALUES (%s, %s, %s, NOW()) ON CONFLICT (question_id) DO UPDATE 
                    SET image_embedding=EXCLUDED.image_embedding, ocr_text=EXCLUDED.ocr_text, updated_at=NOW()
                """, (qid, all_embs[i].tolist(), ocr_text))
                processed_count += 1

            conn_u.commit(); cur_u.close(); conn_u.close()
            
            # 5. 실시간 진행 상황 출력 (남은 개수, %, ETA)
            elapsed = time.time() - start_time
            avg_speed = processed_count / elapsed # items per sec
            current_done = already_done + processed_count
            pct = (current_done / total_goal) * 100
            
            remaining_this_run = remaining_total - processed_count
            eta_seconds = remaining_this_run / avg_speed if avg_speed > 0 else 0
            eta_hours = eta_seconds / 3600
            
            print(f"\r [Progress] {current_done:,}/{total_goal:,} ({pct:.2f}%) | Remaining: {remaining_this_run:,} | Avg: {avg_speed*60:.1f} it/m | ETA: {eta_hours:.1f} hours", end="")
            
    except KeyboardInterrupt:
        print("\n>>> Interrupted by user.")
    except Exception as e:
        print(f"\n>>> Critical Error: {e}")
    finally:
        print(f"\n>>> Local Indexing Finished. Total newly processed: {processed_count}")

if __name__ == "__main__":
    run_local_indexing()
