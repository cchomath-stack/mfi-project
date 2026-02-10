import os
import time
import torch
import numpy as np
import psycopg2
from PIL import Image
from io import BytesIO
import requests
import google.generativeai as genai
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor

# 환경변수에서 Gemini API Key와 DB 주소를 가져옵니다.
RAW_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
GEMINI_API_KEY = RAW_KEY.strip().strip("'").strip('"')
DB_URL = os.getenv("DB_URL", "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki")
MODEL_ID = "openai/clip-vit-base-patch32"

# 서버는 CUDA가 불안정할 수 있으므로 자동 체크
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Gemini 설정
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"!!! [Gemini Init Error] {e}")
    model_gemini = None

# CLIP 설정
print(f">>> Loading CLIP on {DEVICE}...")
clip_model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)

def get_db_conn():
    return psycopg2.connect(DB_URL)

def download_and_preprocess(row):
    try:
        qid_uuid, url = row[0], row[1]
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return str(qid_uuid), img, resp.content
    except Exception as e:
        return str(row[0]), None, None

def run_server_indexing():
    processed_count = 0
    batch_size = 10 
    
    try:
        conn_init = get_db_conn(); cur_init = conn_init.cursor()
        cur_init.execute("SELECT COUNT(*) FROM mcat2.question_render WHERE preview_url IS NOT NULL AND preview_url != ''")
        total_goal = cur_init.fetchone()[0]
        cur_init.execute("SELECT COUNT(*), MAX(updated_at) FROM mcat2.question_image_embeddings")
        row_init = cur_init.fetchone()
        already_done = row_init[0] if row_init else 0
        last_update = row_init[1] if row_init and len(row_init) > 1 else "Never"
        cur_init.close(); conn_init.close()
        
        print("\n" + "="*60)
        print(f" [서버 가동: Gemini 고정밀 인덱싱]")
        print(f" - 진행 상태 : {already_done:,} / {total_goal:,} ({(already_done/total_goal*100 if total_goal>0 else 0):.2f}%)")
        print(f" - 마지막 갱신: {last_update}")
        print(f" - 연산 장치  : {DEVICE}")
        print("="*60 + "\n")
        
        if not GEMINI_API_KEY or "YOUR_GEMINI" in GEMINI_API_KEY:
            print("!!! [Error] GEMINI_API_KEY가 없습니다.")
            return

        print(f">>> 인덱싱을 시작합니다. (배치: {batch_size})")
        start_time = time.time()
        
        while True:
            conn = get_db_conn(); cur = conn.cursor()
            cur.execute("""
                SELECT q.question_id, q.preview_url FROM mcat2.question_render q
                LEFT JOIN mcat2.question_image_embeddings e ON q.question_id = e.question_id
                WHERE (e.question_id IS NULL OR e.ocr_text IS NULL OR e.ocr_text = '' OR e.ocr_text LIKE '%%?%%')
                  AND q.preview_url IS NOT NULL AND q.preview_url != '' LIMIT %s
            """, (batch_size,))
            rows = cur.fetchall(); cur.close(); conn.close()
            
            if not rows:
                print("\n>>> [완료] 인덱싱할 데이터가 더 이상 없습니다!")
                break

            # 1. 병렬 다운로드
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(download_and_preprocess, rows))
            
            valid_items = [r for r in results if r[1] is not None]
            if not valid_items:
                print(" [Batch] No valid images in this batch, skipping...")
                continue

            qids = [x[0] for x in valid_items]
            imgs = [x[1] for x in valid_items]
            raws = [x[2] for x in valid_items]

            # 2. CLIP (Vector)
            inputs = clip_processor(images=imgs, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                out = clip_model.get_image_features(**inputs)
                feat = getattr(out, "image_embeds", getattr(out, "pooler_output", out))
                all_embs = feat.cpu().numpy()
                if all_embs.ndim == 1: all_embs = np.expand_dims(all_embs, axis=0)
                all_embs = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)

            # 3. Gemini (OCR)
            conn_u = get_db_conn(); cur_u = conn_u.cursor()
            for i, qid in enumerate(qids):
                try: 
                    prompt = "이 수학 문제 이미지의 모든 텍스트를 한글로 정확히 읽어줘. 수학 공식은 반드시 LaTeX 형식($...$)을 사용해줘. 다른 설명은 하지 말고 인식된 텍스트만 출력해."
                    image_part = {"mime_type": "image/jpeg", "data": raws[i]}
                    
                    response = model_gemini.generate_content([prompt, image_part])
                    final_ocr = response.text.strip()
                    
                    cur_u.execute("""
                        INSERT INTO mcat2.question_image_embeddings (question_id, image_embedding, ocr_text, updated_at)
                        VALUES (%s, %s, %s, NOW()) ON CONFLICT (question_id) DO UPDATE 
                        SET image_embedding=EXCLUDED.image_embedding, ocr_text=EXCLUDED.ocr_text, updated_at=NOW()
                    """, (qid, all_embs[i].tolist(), final_ocr))
                    processed_count += 1
                except Exception as ex:
                    print(f"\n [OCR 실패] {qid}: {ex}")
                    time.sleep(0.5)

            conn_u.commit(); cur_u.close(); conn_u.close()
            
            # 모니터링
            elapsed = time.time() - start_time
            avg_speed = processed_count / elapsed if elapsed > 0 else 0
            current_done = already_done + processed_count
            eta = (total_goal - current_done) / avg_speed if avg_speed > 0 else 0
            print(f"\r >>> [진행중] 완료: {current_done:,}건 | 속도: {avg_speed*60:.1f} it/min | 잔여: {eta/3600:.1f}시간", end="", flush=True)

    except KeyboardInterrupt:
        print("\n>>> 중단되었습니다.")
    except Exception as e:
        import traceback
        print(f"\n>>> 치명적 오류 발생:")
        traceback.print_exc()

if __name__ == "__main__":
    run_server_indexing()
