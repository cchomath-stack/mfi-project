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
import traceback

# 환경변수 클렌징
def get_env_safe(key, default=""):
    val = os.getenv(key, default)
    return val.strip().strip("'").strip('"') if val else default

GEMINI_API_KEY = get_env_safe("GEMINI_API_KEY")
DB_URL = get_env_safe("DB_URL", "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki")
MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Gemini 초기화
try:
    genai.configure(api_key=GEMINI_API_KEY)
    # 가용 목록에서 확인된 최신 모델 gemini-2.0-flash로 교체
    model_gemini = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    print(f"!!! Gemini Initialization Error: {e}")
    exit(1)

# CLIP 초기화
print(f">>> [System] Loading CLIP on {DEVICE}...")
try:
    clip_model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)
except Exception as e:
    print(f"!!! CLIP Load Error: {e}")
    exit(1)

def get_db_conn():
    return psycopg2.connect(DB_URL)

def download_and_preprocess(row):
    """안전한 다운로드 및 전처리"""
    try:
        if not row or len(row) < 2:
            return None, None, None
        
        qid, url = row[0], row[1]
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        
        # 최적화: 너무 크면 줄임
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
        return str(qid), img, resp.content
    except Exception:
        # 에러 발생 시 qid만이라도 반환 (필터링 위함)
        q = str(row[0]) if row and len(row) > 0 else "unknown"
        return q, None, None

def run_server_indexing():
    processed_count = 0
    batch_size = 10 
    
    # 1. 시동 전 환경 체크
    if not GEMINI_API_KEY or "YOUR_GEMINI" in GEMINI_API_KEY:
        print("!!! [Error] GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        return

    try:
        # 진행률 파악
        conn = get_db_conn(); cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM mcat2.question_render WHERE preview_url IS NOT NULL AND preview_url != ''")
        total_p = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM mcat2.question_image_embeddings")
        done_p = cur.fetchone()[0]
        cur.close(); conn.close()
        
        print("\n" + "="*60)
        print(f" [MFi Gemini OCR Indexer v3.0]")
        print(f" - 진행도: {done_p:,} / {total_p:,} ({(done_p/total_p*100):.2f}%)")
        print(f" - 장치: {DEVICE} / 배치: {batch_size}")
        print("="*60 + "\n")

        start_time = time.time()
        
        while True:
            # 배치 데이터 가져오기
            conn = get_db_conn(); cur = conn.cursor()
            query = """
                SELECT q.question_id, q.preview_url FROM mcat2.question_render q
                LEFT JOIN mcat2.question_image_embeddings e ON q.question_id = e.question_id
                WHERE (e.question_id IS NULL OR e.ocr_text IS NULL OR e.ocr_text = '' OR e.ocr_text LIKE '%%?%%')
                  AND q.preview_url IS NOT NULL AND q.preview_url != '' 
                ORDER BY q.question_id ASC LIMIT %s
            """
            cur.execute(query, (batch_size,))
            batch_rows = cur.fetchall()
            cur.close(); conn.close()
            
            if not batch_rows:
                print("\n>>> [완료] 모든 데이터 인덱싱이 끝났습니다!")
                break

            # 2. 병렬 다운로드
            with ThreadPoolExecutor(max_workers=5) as executor:
                dl_results = list(executor.map(download_and_preprocess, batch_rows))
            
            # 유효한 것만 걸러내기
            valid = [r for r in dl_results if r[1] is not None]
            if not valid:
                print(f" [!] 이번 배치({len(batch_rows)}건)에 유효한 이미지가 없습니다. 다음으로 넘어갑니다.")
                continue

            qids = [v[0] for v in valid]
            imgs = [v[1] for v in valid]
            raws = [v[2] for v in valid]

            # 3. CLIP 임베딩 (Batch)
            try:
                inputs = clip_processor(images=imgs, return_tensors="pt", padding=True).to(DEVICE)
                with torch.no_grad():
                    out = clip_model.get_image_features(**inputs)
                    # CLIP 버전에 따른 대응
                    if isinstance(out, torch.Tensor):
                        feat = out
                    else:
                        feat = getattr(out, "image_embeds", getattr(out, "pooler_output", out))
                    
                    all_embs = feat.cpu().numpy()
                    if all_embs.ndim == 1: 
                        all_embs = np.expand_dims(all_embs, axis=0)
                    # Normalization
                    all_embs = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)
            except Exception as clip_err:
                print(f"\n!!! [CLIP 에러] {clip_err}")
                continue

            # 4. Gemini OCR (건별 처리)
            conn_u = get_db_conn(); cur_u = conn_u.cursor()
            for i in range(len(qids)):
                qid = qids[i]
                emb = all_embs[i]
                raw_data = raws[i]
                
                # 최대 3회 재시도 (할당량 초과 시 대기)
                for attempt in range(3):
                    try: 
                        prompt = """수학 전문가로서 이 이미지의 모든 수학적 내용과 텍스트를 인식해줘.
규칙:
1. 모든 수학 공식, 기호, 숫자는 반드시 LaTeX 형식($...$ 또는 $$...$$)으로 작성해. (예: $x^2 + y^2 = r^2$, $\frac{1}{2}$ 등)
2. 한글 문장과 단어도 빠짐없이 정확하게 읽어줘.
3. 다른 설명 없이 인식된 결과(LaTeX가 포함된 텍스트)만 출력해."""
                        image_part = {"mime_type": "image/jpeg", "data": raw_data}
                        
                        response = model_gemini.generate_content([prompt, image_part])
                        ocr_text = response.text.strip() if response and response.text else ""
                        
                        # DB 저장
                        cur_u.execute("""
                            INSERT INTO mcat2.question_image_embeddings (question_id, image_embedding, ocr_text, updated_at)
                            VALUES (%s, %s, %s, NOW()) ON CONFLICT (question_id) DO UPDATE 
                            SET image_embedding=EXCLUDED.image_embedding, ocr_text=EXCLUDED.ocr_text, updated_at=NOW()
                        """, (qid, emb.tolist(), ocr_text))
                        processed_count += 1
                        break # 성공 시 루프 탈출
                        
                    except Exception as ocr_err:
                        err_str = str(ocr_err)
                        if "429" in err_str or "quota" in err_str.lower():
                            wait_time = 30 * (attempt + 1)
                            print(f"\n [!] 할당량 초과 (Quota Exceeded). {wait_time}초 후 재시도합니다... ({qid[:8]})")
                            time.sleep(wait_time)
                        else:
                            print(f"\n [!] OCR 실패 ({qid}): {ocr_err}")
                            break # 다른 에러는 다음 항목으로
                
                # 유료 티어 속도 제한 해제
                time.sleep(0.1)

            conn_u.commit(); cur_u.close(); conn_u.close()
            
            # 실시간 상태 출력
            elapsed = time.time() - start_time
            speed = processed_count / (elapsed / 60) if elapsed > 0 else 0
            remain = total_p - (done_p + processed_count)
            eta = remain / (speed / 60) if speed > 0 else 0
            
            print(f"\r >>> [진행중] 처리: {processed_count:,}건 | 속도: {speed:.1f} it/min | 남은시간: {eta/3600:.1f}h", end="", flush=True)

    except KeyboardInterrupt:
        print("\n>>> 사용자에 의해 중단되었습니다.")
    except Exception:
        print("\n\n!!! [치명적 오류 발생] 아래 내용을 복사해서 알려주세요:")
        traceback.print_exc()

if __name__ == "__main__":
    run_server_indexing()
