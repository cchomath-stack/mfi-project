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

# --- [설정] 여기에 Gemini API Key를 입력하세요 ---
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
# -----------------------------------------------

# 서버 내부 DB 주소 (환경변수에서 가져오거나 직접 입력)
DB_URL = os.getenv("DB_URL", "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki")
MODEL_ID = "openai/clip-vit-base-patch32"

# 서버는 CUDA가 불안정할 수 있으므로 자동 체크
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Gemini 설정
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel('gemini-1.5-flash')

# CLIP 설정 (서버에서는 ONNX 대신 안전한 Torch 버전 사용)
print(f">>> Loading CLIP on {DEVICE}...")
clip_model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)

def get_db_conn():
    return psycopg2.connect(DB_URL)

def download_and_preprocess(qid_uuid, url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return str(qid_uuid), img, resp.content
    except Exception as e:
        return str(qid_uuid), None, None

def run_server_indexing():
    processed_count = 0
    batch_size = 10 # 대규모 처리 시 API 할당량(RPM)을 고려하여 조절
    
    try:
        conn_init = get_db_conn(); cur_init = conn_init.cursor()
        cur_init.execute("SELECT COUNT(*) FROM mcat2.question_render WHERE preview_url IS NOT NULL AND preview_url != ''")
        total_goal = cur_init.fetchone()[0]
        cur_init.execute("SELECT COUNT(*), MAX(updated_at) FROM mcat2.question_image_embeddings")
        already_done, last_update = cur_init.fetchone()
        cur_init.close(); conn_init.close()
        
        print("\n" + "="*60)
        print(f" [서버 가동: Gemini 고정밀 인덱싱]")
        print(f" - 진행 상태 : {already_done:,} / {total_goal:,} ({(already_done/total_goal*100):.2f}%)")
        print(f" - 마지막 갱신: {last_update}")
        print(f" - 연산 장치  : {DEVICE}")
        print("="*60 + "\n")
        
        if not GEMINI_API_KEY or "YOUR_GEMINI" in GEMINI_API_KEY:
            print("!!! [Error] GEMINI_API_KEY가 없습니다. 파일 상단의 KEY를 수정해주세요.")
            return

        print(">>> 인덱싱을 곧 시작합니다. (배치 사이즈: 10)")
        start_time = time.time()
        
        while True:
            conn = get_db_conn(); cur = conn.cursor()
            # 팁: 비어있거나 '?'가 포함된 데이터만 골라내기
            cur.execute("""
                SELECT q.question_id, q.preview_url FROM mcat2.question_render q
                LEFT JOIN mcat2.question_image_embeddings e ON q.question_id = e.question_id
                WHERE (e.question_id IS NULL OR e.ocr_text LIKE '%?%' OR e.ocr_text = '')
                  AND q.preview_url IS NOT NULL AND q.preview_url != '' LIMIT %s
            """, (batch_size,))
            rows = cur.fetchall(); cur.close(); conn.close()
            
            if not rows:
                print("\n>>> [완료] 인덱싱할 데이터가 더 이상 없습니다!")
                break

            # 1. 병렬 다운로드
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(lambda r: download_and_preprocess(r[0], r[1]), rows))
            
            valid_items = [(qid, img, raw) for qid, img, raw in results if img is not None]
            if not valid_items: continue

            qids = [x[0] for x in valid_items]
            imgs = [x[1] for x in valid_items]
            raws = [x[2] for x in valid_items]

            # 2. CLIP (Vector)
            inputs = clip_processor(images=imgs, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                out = clip_model.get_image_features(**inputs)
                feat = getattr(out, "image_embeds", getattr(out, "pooler_output", out))
                all_embs = feat.cpu().numpy()
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
                    time.sleep(1)

            conn_u.commit(); cur_u.close(); conn_u.close()
            
            # 모니터링
            elapsed = time.time() - start_time
            avg_speed = processed_count / elapsed if elapsed > 0 else 0
            current_done = already_done + processed_count
            eta = (total_goal - current_done) / avg_speed if avg_speed > 0 else 0
            
            print(f"\r >>> [서버 진행중] 완료: {current_done:,}건 | 속도: {avg_speed*60:.1f} it/min | 잔여: {eta/3600:.1f}시간", end="", flush=True)

    except KeyboardInterrupt:
        print("\n>>> 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n>>> 치명적 오류: {e}")

if __name__ == "__main__":
    run_server_indexing()
