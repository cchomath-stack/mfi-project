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
from pix2tex.cli import LatexOCR
import requests
import httpx
import time
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
SQLITE_DB_PATH = "users.db"

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
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES))
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

# --- 초기화 ---
@app.on_event("startup")
def startup_event():
    global math_ocr
    print("Initializing Math OCR (pix2tex)...")
    try:
        math_ocr = LatexOCR()
        print("Math OCR initialized.")
    except Exception as e:
        print(f"Math OCR Init Error: {e}")
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
    conn = get_sqlite_conn()
    row = conn.execute("SELECT hashed_password, is_approved, role FROM web_users WHERE username=?", (form_data.username,)).fetchone()
    conn.close()
    if not row or not verify_password(form_data.password, row['hashed_password']):
        raise HTTPException(status_code=401)
    
    # 일반 유저인데 승인되지 않은 경우 토큰 발급은 하되, 이후 API에서 403 처리됨 (또는 여기서 차칭 가능)
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
        # 완료수: 실제 OCR+임베딩이 끝난 수
        cur.execute("SELECT COUNT(*) FROM mcat2.problem_image_embeddings WHERE ocr_text IS NOT NULL AND ocr_text != ''")
        done = cur.fetchone()[0]
        # 전체수: 이미지가 존재하는 전체 문항
        cur.execute("SELECT COUNT(*) FROM mcat2.problems_problem WHERE problem_image_url IS NOT NULL AND problem_image_url != ''")
        total_p = cur.fetchone()[0]
        pending = max(0, total_p - done)
        cur.execute("SELECT MAX(updated_at) FROM mcat2.problem_image_embeddings")
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

@app.post("/admin/update-embeddings")
async def run_update(background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin": raise HTTPException(status_code=403)
    global update_in_progress, update_start_time, processed_in_session
    if update_in_progress: return {"message": "Active"}
    update_in_progress, update_start_time, processed_in_session = True, datetime.now(), 0
    background_tasks.add_task(background_update_embeddings)
    return {"message": "Started"}

def background_update_embeddings():
    global update_in_progress, embeddings_cache, processed_in_session
    print("\n>>> New Background indexing task started (question_render).")
    try:
        while True:
            conn = get_db_conn(); cur = conn.cursor()
            # question_render에는 있으나 embeddings에는 없는 데이터 50개씩 처리
            cur.execute("""
                SELECT q.question_id, q.preview_url FROM mcat2.question_render q
                LEFT JOIN mcat2.question_image_embeddings e ON q.question_id = e.question_id
                WHERE e.question_id IS NULL
                  AND q.preview_url IS NOT NULL AND q.preview_url != '' LIMIT 50
            """)
            rows = cur.fetchall(); cur.close(); conn.close()
            if not rows or not update_in_progress: break

            for qid_uuid, url in rows:
                qid = str(qid_uuid)
                try:
                    resp = requests.get(url, timeout=10); resp.raise_for_status()
                    image_data = resp.content
                    image = Image.open(BytesIO(image_data)).convert("RGB")
                    
                    # 1. CLIP Embedding
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = model.get_image_features(**inputs)
                        feat = getattr(out, "image_embeds", getattr(out, "pooler_output", out))
                        emb = feat.cpu().numpy()[0] if torch.is_tensor(feat) else feat[0]
                        emb = emb / (np.linalg.norm(emb) + 1e-8)
                    
                    # 2. 고정밀 Math OCR (pix2tex)
                    try: 
                        ocr_text = math_ocr(image)
                        print(f" [OCR Debug] {qid[:8]}: {ocr_text}")
                    except Exception as o_e:
                        print(f" [OCR Error] {o_e}")
                        ocr_text = ""

                    # 3. DB 저장 (mcat2.question_image_embeddings)
                    conn_s = get_db_conn(); cur_s = conn_s.cursor()
                    cur_s.execute("""
                        INSERT INTO mcat2.question_image_embeddings (question_id, image_embedding, ocr_text, updated_at)
                        VALUES (%s, %s, %s, NOW()) ON CONFLICT (question_id) DO UPDATE 
                        SET image_embedding=EXCLUDED.image_embedding, ocr_text=EXCLUDED.ocr_text, updated_at=NOW()
                    """, (qid, emb.tolist(), ocr_text))
                    conn_s.commit(); cur_s.close(); conn_s.close()

                    # 4. 캐시 업데이트
                    found = False
                    for i, (cid, cemb, ctxt) in enumerate(embeddings_cache):
                        if cid == qid: embeddings_cache[i] = (qid, emb, ocr_text); found = True; break
                    if not found: embeddings_cache.append((qid, emb, ocr_text))

                    processed_in_session += 1
                    print(f" [Progress] {processed_in_session} items processed. (QID: {qid[:8]})")
                except Exception as e: print(f" [Error] {qid[:8]}: {e}")
        print(">>> Task finished.")
    except Exception as e: print(f">>> Critical Error: {e}")
    finally: update_in_progress = False

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
        
        try:
            # 쿼리 이미지 Math OCR (pix2tex)
            q_text_str = math_ocr(img)
            q_text = set(filter(None, q_text_str.lower().split()))
            print(f" [Debug] Query Math-OCR Text: {q_text_str}")
        except Exception as e:
            print(f" [Debug] Math-OCR Error: {e}")
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
        # question_render와 user_extracted_questions를 조인하여 메타데이터 추출
        cur.execute("""
            SELECT q.question_id, q.preview_url, COALESCE(u.source, 'Unknown Source')
            FROM mcat2.question_render q
            LEFT JOIN mcat2.user_extracted_questions u ON q.question_id::text = u.question_id
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
