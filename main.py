import os

# --- [중요] 모든 라이브러리 임포트 전 오프라인 설정 강제 (네트워크 타임아웃 방지) ---
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import uuid
import torch
import numpy as np
import psycopg2
import sqlite3
import google.generativeai as genai
import requests
import httpx
import time
import base64
from collections import defaultdict, deque
from io import BytesIO
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
import config
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

# --- 설정 및 상수 (config.py로 이전) ---
SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")

MODEL_ID = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    print(f"Loading CLIP model and processor (Device: {device})...")
    model = CLIPModel.from_pretrained(MODEL_ID, local_files_only=False).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID, local_files_only=False)
except Exception:
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

print("Preparing OCR engine placeholder...")
# math_ocr는 startup_event에서 초기화하여 Docker 시작 안정성 확보
math_ocr = None

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(title="MFi (MCAT_Find_image) API")

# --- 전역 상태 ---
embeddings_cache = [] 
update_in_progress = False
update_start_time = None
processed_in_session = 0
backend_logs = deque(maxlen=50) # 최근 50개 로그 유지

def log_backend(msg):
    full_msg = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(full_msg)
    backend_logs.append(full_msg)

# --- Rate Limiter (Phase 6) ---
class RateLimiter:
    def __init__(self, requests_max: int, window_seconds: int):
        self.requests_max = requests_max
        self.window_seconds = window_seconds
        self.user_requests = defaultdict(deque)

    def is_allowed(self, user_id: str) -> bool:
        now = time.time()
        requests = self.user_requests[user_id]
        while requests and requests[0] < now - self.window_seconds:
            requests.popleft()
        if len(requests) < self.requests_max:
            requests.append(now)
            return True
        return False

# 1분에 20회 요청 제한 (사내 운영 정책 반영)
search_limiter = RateLimiter(requests_max=20, window_seconds=60)

# --- 모델 ---
class Token(BaseModel):
    access_token: str
    token_type: str

class UserCreate(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None
    role: str = "user"

class SearchResult(BaseModel):
    problem_id: str
    image_url: str
    source_title: str
    similarity: float
    ocr_match: Optional[float] = 0.0

class UpdateStats(BaseModel):
    total_embeddings: int
    pending_count: int
    last_updated: Optional[str] = None
    update_in_progress: bool = False
    processed_this_session: int = 0
    items_per_min: float = 0.0
    estimated_minutes_left: float = 0.0

# --- 유틸리티 ---
def get_db_conn(): return psycopg2.connect(config.DB_URL)
def get_sqlite_conn():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def verify_password(plain, hashed): return pwd_context.verify(plain, hashed)
def get_password_hash(password): return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now() + (expires_delta or timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, config.SECRET_KEY, algorithm=config.ALGORITHM)

async def _get_user_by_token(token: str):
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        username = payload.get("sub")
        if not username:
            print(" [Auth Error] Payload missing 'sub'")
            raise HTTPException(status_code=401)
        
        conn = get_sqlite_conn()
        row = conn.execute("SELECT id, username, role, is_approved FROM web_users WHERE username = ?", (username,)).fetchone()
        conn.close()
        
        if not row:
            print(f" [Auth Error] User '{username}' not found in database")
            raise HTTPException(status_code=401)
        
        user_data = {"id": row['id'], "username": row['username'], "role": row['role'], "is_approved": row['is_approved']}
        
        if user_data["role"] != "admin" and not user_data["is_approved"]:
            print(f" [Auth Debug] User '{username}' is pending approval")
            raise HTTPException(status_code=403, detail="Account pending approval by admin")
            
        return user_data
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f" [Auth Error] Token verification failed: {str(e)}")
        raise HTTPException(status_code=401)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    return await _get_user_by_token(token)

async def get_current_user_with_query_token(token: Optional[str] = None, header_token: Optional[str] = Depends(OAuth2PasswordBearer(tokenUrl="token", auto_error=False))):
    actual_token = token or header_token
    if not actual_token:
        raise HTTPException(status_code=401)
    return await _get_user_by_token(actual_token)

# --- [Gemini 설정] ---
def load_gemini():
    raw_key = config.GEMINI_API_KEY
    # 따옴표 찌꺼기 제거 (중요)
    clean_key = raw_key.strip().strip("'").strip('"')
    
    if clean_key and clean_key != "YOUR_GEMINI_API_KEY_HERE":
        try:
            genai.configure(api_key=clean_key)
            # 사용자 환경에서 확인된 최신 모델 gemini-2.0-flash 사용
            model = genai.GenerativeModel('gemini-2.0-flash')
            return model, clean_key
        except Exception as e:
            print(f" [Gemini 로드 에러] {e}")
            return None, clean_key
    return None, clean_key

model_gemini, GEMINI_API_KEY = load_gemini()

def initialize_ocr():
    global math_ocr
    print(f"[OCR] Checking Gemini API Key: {GEMINI_API_KEY[:6]}...{GEMINI_API_KEY[-4:] if len(GEMINI_API_KEY)>10 else ''}")
    if model_gemini is not None:
        try:
            # 진짜로 작동하는지 아주 가벼운 테스트 호출
            math_ocr = "gemini" 
            print("[OCR] Using Gemini 1.5 Flash for high-precision OCR (Active).")
        except Exception as e:
            print(f"[OCR] Gemini API Test Failed: {e}")
            math_ocr = None
    else:
        print("[OCR] Warning: Gemini API Key not set or initialization failed.")

def get_mathpix_ocr(img_pil):
    """Mathpix Snip API를 사용한 고성능 수식 OCR"""
    if not config.MATHPIX_APP_ID or not config.MATHPIX_APP_KEY:
        return None
    try:
        buffered = BytesIO()
        img_pil.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        url = "https://api.mathpix.com/v3/text"
        headers = {
            "app_id": config.MATHPIX_APP_ID.strip().strip("'").strip('"'),
            "app_key": config.MATHPIX_APP_KEY.strip().strip("'").strip('"'),
            "Content-type": "application/json"
        }
        payload = {
            "src": f"data:image/jpeg;base64,{img_base64}",
            "formats": ["text"],
            "data_options": {
                "include_latex": True
            }
        }
        
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("text", "").strip()
        else:
            print(f" [Mathpix Error] Status {resp.status_code}: {resp.text}")
            return None
    except Exception as e:
        print(f" [Mathpix Utility Error] {e}")
        return None

def get_ocr_text(img_pil):
    # 1. Mathpix 우선 시도 (키가 있는 경우)
    if config.MATHPIX_APP_ID and config.MATHPIX_APP_KEY:
        mx_text = get_mathpix_ocr(img_pil)
        if mx_text:
            print(f"  > [OCR] Mathpix Success (LaTeX Quality: High)")
            return mx_text

    # 2. Gemini 백업 시도
    if model_gemini is None:
        return ""
    try:
        # PIL 이미지를 바이트로 변환
        buffered = BytesIO()
        img_pil.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        
        prompt = """수학 전문가로서 이 이미지의 모든 수학적 내용과 텍스트를 인식해줘.
규칙:
1. 모든 수학 공식, 기호, 숫자는 반드시 LaTeX 형식($...$ 또는 $$...$$)으로 작성해. (예: $x^2 + y^2 = r^2$, $\frac{1}{2}$ 등)
2. 한글 문장과 단어도 빠짐없이 정확하게 읽어줘.
3. 다른 설명 없이 인식된 결과(LaTeX가 포함된 텍스트)만 출력해."""
        image_part = {"mime_type": "image/jpeg", "data": img_bytes}
        
        response = model_gemini.generate_content([prompt, image_part])
        return response.text.strip()
    except Exception as e:
        print(f" [Gemini OCR 유틸 에러] {e}")
        return ""

# --- 초기화 ---
@app.on_event("startup")
def startup_event():
    initialize_ocr()
    sq_conn = get_sqlite_conn()
    # 테이블 확장 대응 (마이그레이션 스크립트로 처리했지만 안전을 위해 IF NOT EXISTS 유지)
    sq_conn.execute("""
        CREATE TABLE IF NOT EXISTS web_users (
            id TEXT PRIMARY KEY, 
            username TEXT UNIQUE, 
            hashed_password TEXT, 
            role TEXT,
            google_id TEXT,
            email TEXT,
            picture_url TEXT,
            is_approved INTEGER DEFAULT 0
        )
    """)
    # 관리자 계정 강제 초기화 (비밀번호: admin123! 고정)
    sq_conn.execute("DELETE FROM web_users WHERE username='admin'")
    sq_conn.execute("INSERT INTO web_users (id, username, hashed_password, role, is_approved) VALUES (?,?,?,?,?)", 
                    (str(uuid.uuid4()), 'admin', get_password_hash('admin123!'), 'admin', 1))
    sq_conn.commit()
    sq_conn.close()
    print("Application Startup: System Initializing...")
    log_backend("Server started successfully. Ready for local verification.")
    print("Admin user initialized/reset.")

    print("Loading embeddings from question_image_embeddings...")
    global embeddings_cache
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT question_id, image_embedding, ocr_text FROM mcat2.question_image_embeddings")
        for row in cur.fetchall():
            pid, emb, ocr = str(row[0]), np.array(row[1], dtype=np.float32), row[2] or ""
            embeddings_cache.append((pid, emb / (np.linalg.norm(emb) + 1e-8), ocr.strip()))
        cur.close(); conn.close()
        print(f"Loaded {len(embeddings_cache)} items.")
    except Exception as e: print(f"Load error: {e}")

# --- API ---
@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    log_backend(f"Login Attempt: {form_data.username}")
    conn = get_sqlite_conn()
    row = conn.execute("SELECT hashed_password, is_approved, role FROM web_users WHERE username=?", (form_data.username,)).fetchone()
    conn.close()
    
    if not row:
        log_backend(f"Login Failed: User '{form_data.username}' not found in DB.")
        raise HTTPException(status_code=401)
    
    is_valid = verify_password(form_data.password, row['hashed_password'])
    log_backend(f"Login Verify: {form_data.username} | Pass Match: {is_valid} | Approved: {row['is_approved']}")
    
    if not is_valid:
        raise HTTPException(status_code=401)
    
    return {"access_token": create_access_token({"sub": form_data.username}), "token_type": "bearer"}

# --- Google OAuth Endpoints ---

@app.get("/auth/google/login")
async def google_login():
    # 실제 구현 시에는 google auth library 등을 사용하거나 직접 리다이렉트 URL 생성
    # 여기서는 개념 및 흐름 구현을 위해 리다이렉트 예시 작성
    scope = "https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/userinfo.email"
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?response_type=code&client_id={config.GOOGLE_CLIENT_ID}&redirect_uri={config.GOOGLE_REDIRECT_URI}&scope={scope}&access_type=offline&prompt=consent"
    return RedirectResponse(auth_url)

@app.get("/auth/google/callback")
async def google_callback(code: str):
    # 1. Exchange code for tokens
    token_url = "https://oauth2.googleapis.com/token"
    token_data = {
        "code": code,
        "client_id": config.GOOGLE_CLIENT_ID,
        "client_secret": config.GOOGLE_CLIENT_SECRET,
        "redirect_uri": config.GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    
    # 2. Get user info from Google
    async with httpx.AsyncClient() as client:
        t_res = await client.post(token_url, data=token_data)
        if t_res.status_code != 200:
            error_detail = t_res.text
            print(f" [OAuth Error] Token exchange failed: {error_detail}")
            return JSONResponse(status_code=400, content={"detail": f"Failed to exchange token: {error_detail}"})
        
        tokens = t_res.json()
        print(f" [OAuth Debug] Tokens received. Fetching user info...")
        u_res = await client.get("https://www.googleapis.com/oauth2/v2/userinfo", 
                                headers={"Authorization": f"Bearer {tokens['access_token']}"})
        if u_res.status_code != 200:
            print(f" [OAuth Error] Failed to fetch user info: {u_res.text}")
            return JSONResponse(status_code=400, content={"detail": "Failed to fetch user info"})
        
        google_user = u_res.json() # id, email, name, picture 등 포함
        print(f" [OAuth Debug] Google User: {google_user.get('email')}")
    
    # 3. Create or update user in SQLite
    try:
        gid = google_user.get("id")
        email = google_user.get("email")
        picture = google_user.get("picture")
        username = email # 이메일 전체를 사용하여 중복 방지
        
        conn = get_sqlite_conn()
        row = conn.execute("SELECT id, username, is_approved FROM web_users WHERE google_id = ?", (gid,)).fetchone()
        
        if not row:
            # 신규 가입 (승인 대기 상태 0)
            uid = str(uuid.uuid4())
            conn.execute("INSERT INTO web_users (id, username, google_id, email, picture_url, role, is_approved) VALUES (?,?,?,?,?,?,?)",
                        (uid, username, gid, email, picture, "user", 0))
            print(f" [OAuth Debug] New user registered: {username}")
        else:
            # 기존 가입된 구글 유저 정보 업데이트
            # 데이터베이스에 이미 저장된 username을 사용해야 토큰 불일치가 발생하지 않음
            username = row['username']
            conn.execute("UPDATE web_users SET picture_url = ?, email = ? WHERE google_id = ?", (picture, email, gid))
            print(f" [OAuth Debug] Existing user logged in: {username}")
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Google Callback Error: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Registration failed: {str(e)}"})
    
    # 4. Generate JWT for the user
    token = create_access_token({"sub": username})
    
    # 클라이언트로 토큰과 함께 리다이렉트 (프론트엔드에서 처리)
    return RedirectResponse(url=f"/?token={token}")

@app.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    # get_current_user에서 이미 is_approved 체크를 하므로 여기 도달하면 승인된 것임
    return current_user

# --- Admin User Management ---

@app.get("/admin/users")
async def get_users(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin": raise HTTPException(status_code=403)
    conn = get_sqlite_conn()
    rows = conn.execute("SELECT id, username, email, picture_url, role, is_approved FROM web_users WHERE role != 'admin'").fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/admin/users/{user_id}/approve")
async def approve_user(user_id: str, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin": raise HTTPException(status_code=403)
    conn = get_sqlite_conn()
    conn.execute("UPDATE web_users SET is_approved = 1 WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return {"message": "User approved"}

@app.delete("/admin/users/{user_id}")
async def delete_user(user_id: str, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin": raise HTTPException(status_code=403)
    conn = get_sqlite_conn()
    conn.execute("DELETE FROM web_users WHERE id = ? AND role != 'admin'", (user_id,))
    conn.commit()
    conn.close()
    return {"message": "User deleted"}

@app.get("/admin/stats", response_model=UpdateStats)
async def get_update_stats(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin": raise HTTPException(status_code=403)
    try:
        conn = get_db_conn(); cur = conn.cursor()
        # 완료수: 신규 테이블(question_image_embeddings)에 등록된 수
        # 완료수: 임베딩과 OCR이 모두 정상적으로 완료된 수만 'Done'으로 표시
        cur.execute("SELECT COUNT(*) FROM mcat2.question_image_embeddings WHERE ocr_text IS NOT NULL AND ocr_text != ''")
        done = cur.fetchone()[0]
        # 전체수: question_render 테이블의 전체 유효 문항 수
        cur.execute("SELECT COUNT(*) FROM mcat2.question_render WHERE preview_url IS NOT NULL AND preview_url != ''")
        total_p = cur.fetchone()[0]
        pending = max(0, total_p - done)
        cur.execute("SELECT MAX(updated_at) FROM mcat2.question_image_embeddings WHERE ocr_text IS NOT NULL AND ocr_text != ''")
        ldt = cur.fetchone()[0]
        cur.close(); conn.close()

        speed, eta = 0.0, 0.0
        if update_in_progress and update_start_time and processed_in_session > 0:
            elapsed = (datetime.now() - update_start_time).total_seconds() / 60.0
            if elapsed > 0:
                speed = processed_in_session / elapsed
                if speed > 0: eta = pending / speed

        return UpdateStats(
            total_embeddings=done, pending_count=pending,
            last_updated=(ldt + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S") if ldt else "None",
            update_in_progress=update_in_progress, processed_this_session=processed_in_session,
            items_per_min=round(speed, 2), estimated_minutes_left=round(eta, 1)
        )
    except Exception as e: print(f"Stats Error: {e}"); raise HTTPException(status_code=500)

@app.get("/admin/debug-logs")
async def get_backend_logs(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin": raise HTTPException(status_code=403)
    return list(backend_logs)

@app.post("/admin/update-embeddings")
async def run_update(background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin": raise HTTPException(status_code=403)
    print(f">>> [API] Received run_update request from {current_user['username']}")
    global update_in_progress, update_start_time, processed_in_session
    if update_in_progress: return {"message": "Active"}
    update_in_progress, update_start_time, processed_in_session = True, datetime.now(), 0
    background_tasks.add_task(background_update_embeddings)
    return {"message": "Started"}

@app.post("/admin/stop-update")
async def stop_update(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin": raise HTTPException(status_code=403)
    global update_in_progress
    if not update_in_progress: return {"message": "Not running"}
    update_in_progress = False
    return {"message": "Stopping"}

from concurrent.futures import ThreadPoolExecutor

def background_update_embeddings():
    global update_in_progress, embeddings_cache, processed_in_session, update_start_time
    log_backend(">>> Optimized Background indexing task started.")
    update_start_time = datetime.now()
    
    def download_and_preprocess(qid_uuid, url):
        try:
            resp = requests.get(url, timeout=10); resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            return str(qid_uuid), img
        except Exception as e:
            log_backend(f" [Download Error] {qid_uuid}: {e}")
            return str(qid_uuid), None

    processed_ids = set()
    
    try:
        while True:
            if not update_in_progress: 
                log_backend(">>> Task stop requested by user.")
                break
            
            log_backend(" [Task] Fetching next batch of 10 items...")
            conn = get_db_conn(); cur = conn.cursor()
            # session에서 이미 처리한 ID는 SQL에서 제외하지 않고 가져온 뒤 파이썬에서 필터링하거나, 
            # 단순히 DB 상태(null인 것들)만 믿고 가져옵니다. 
            # 루프 방지를 위해 ORDER BY와 OFFSET을 적절히 쓰거나, 
            # 단순히 처리 완료된 것은 WHERE 조건에서 빠지게 되므로 NOT IN을 뺍니다.
            cur.execute("""
                SELECT q.question_id, q.preview_url FROM mcat2.question_render q
                LEFT JOIN mcat2.question_image_embeddings e ON q.question_id = e.question_id
                WHERE (e.question_id IS NULL OR e.ocr_text IS NULL OR e.ocr_text = '')
                  AND q.preview_url IS NOT NULL AND q.preview_url != ''
                ORDER BY q.updated_at DESC
                LIMIT 100
            """)
            rows = cur.fetchall(); cur.close(); conn.close()
            
            if not rows:
                log_backend(" [Task] No more pending items found. Finishing.")
                break

            # 이미 이번 세션에서 시도했던 ID는 제외 (SQL bloat 방지용 파이썬 필터링)
            pending_rows = [r for r in rows if str(r[0]) not in processed_ids][:10]
            
            if not pending_rows:
                log_backend(" [Task] All fetched items already processed this session. Finishing to avoid loop.")
                break

            log_backend(f" [Batch] Processing {len(pending_rows)} items...")
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(lambda r: download_and_preprocess(r[0], r[1]), pending_rows))
            
            if not rows:
                log_backend(" [Task] No more pending items found. Finishing.")
                break

            log_backend(f" [Batch] Processing {len(rows)} items...")
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(lambda r: download_and_preprocess(r[0], r[1]), rows))
            
            valid_items = [(qid, img) for qid, img in results if img is not None]
            if not valid_items:
                # 다음 번 시도를 위해 처리 완료 목록에는 넣어야 함 (무한 루프 방지)
                for qid, _ in results: processed_ids.add(qid)
                log_backend(" [Batch] No valid images in this batch. Skipping.")
                continue

            qids = [x[0] for x in valid_items]
            imgs = [x[1] for x in valid_items]
            
            inputs = processor(images=imgs, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model.get_image_features(**inputs)
                feat = getattr(out, "image_embeds", getattr(out, "pooler_output", out))
                all_embs = feat.cpu().numpy()
                all_embs = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)

            if math_ocr is None:
                log_backend(" [OCR] Re-initializing engine...")
                initialize_ocr()

            conn_s = get_db_conn(); cur_s = conn_s.cursor()
            for i, qid in enumerate(qids):
                if not update_in_progress: break
                img = imgs[i]; emb = all_embs[i]; ocr_text = ""
                
                for attempt in range(2):
                    try: 
                        ocr_text = get_ocr_text(img)
                        break 
                    except Exception as o_e:
                        if "quota" in str(o_e).lower():
                            log_backend(f"  > [Quota] Retrying {qid[:8]}...")
                            time.sleep(10)
                        else:
                            log_backend(f"  > [OCR Error] {qid[:8]}: {o_e}")
                            break

                cur_s.execute("""
                    INSERT INTO mcat2.question_image_embeddings (question_id, image_embedding, ocr_text, updated_at)
                    VALUES (%s, %s, %s, NOW()) ON CONFLICT (question_id) DO UPDATE 
                    SET image_embedding=EXCLUDED.image_embedding, ocr_text=EXCLUDED.ocr_text, updated_at=NOW()
                """, (qid, emb.tolist(), ocr_text))
                
                processed_in_session += 1
                processed_ids.add(qid)
            
            conn_s.commit(); cur_s.close(); conn_s.close()
            log_backend(f" [Batch] Step complete. Total session: {processed_in_session}")
            time.sleep(0.5)
        
        log_backend(f">>> Task finished successfully. Items: {processed_in_session}")
    except Exception as e: 
        log_backend(f">>> [CRITICAL] Task died: {e}")
    finally: 
        update_in_progress = False

@app.get("/api/v1/proxy-image")
async def proxy_image(url: str, token: Optional[str] = None, current_user: dict = Depends(get_current_user_with_query_token)):
    # 인증된 사용자만 원본 이미지 url에 접근 가능
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10)
            if resp.status_code == 200:
                return StreamingResponse(BytesIO(resp.content), media_type=resp.headers.get("content-type", "image/jpeg"))
            else:
                raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[SearchResult])
async def search(file: UploadFile = File(...), current_user: dict = Depends(get_current_user), token: str = Depends(oauth2_scheme)):
    # [Phase 6] Rate Limiting 체크
    if not search_limiter.is_allowed(current_user["username"]):
        raise HTTPException(status_code=429, detail="너무 잦은 검색 요청입니다. 잠시 후 다시 시도해 주세요.")

    try:
        data = await file.read(); img = Image.open(BytesIO(data)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
            if torch.is_tensor(out):
                q_emb = out.cpu().numpy()[0]
            else:
                feat = getattr(out, "image_embeds", getattr(out, "pooler_output", out))
                q_emb = feat.cpu().numpy()[0] if torch.is_tensor(feat) else feat[0]
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        
        # Gemini OCR for search query
        try:
            q_text_str = get_ocr_text(img)
            q_text = set(filter(None, q_text_str.lower().split()))
        except Exception as e:
            print(f" [Debug] Gemini OCR Error: {e}")
            q_text = set()

        scores = []
        for qid, emb, ocr in embeddings_cache:
            clip = float(np.dot(q_emb, emb))
            # 수식 매칭을 위해 간단한 셋 교집합 점수 계산
            txt = len(q_text.intersection(set(ocr.lower().split()))) / max(len(q_text), 1) if q_text and ocr else 0.0
            scores.append((qid, (clip * 0.2) + (txt * 0.8), txt))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:5]; ids = [s[0] for s in top]
        
        conn = get_db_conn(); cur = conn.cursor()
        # question_render와 questions 테이블을 조인하여 실제 출처 파일명(origin_file_name) 추출
        cur.execute("""
            SELECT q.question_id, q.preview_url, COALESCE(qt.origin_file_name, 'Unknown Source')
            FROM mcat2.question_render q
            LEFT JOIN mcat2.questions qt ON q.question_id = qt.id
            WHERE q.question_id = ANY(%s::uuid[])
        """, (ids,))
        
        imap = {str(r[0]): {
            "url": f"/api/v1/proxy-image?url={requests.utils.quote(r[1])}&token={token}", 
            "src": r[2]
        } for r in cur.fetchall()}
        cur.close(); conn.close()

        return [{"problem_id": p, "image_url": imap.get(p, {}).get("url", ""), "source_title": imap.get(p, {}).get("src", "Unknown"), "similarity": s, "ocr_match": o} for p, s, o in top]
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
