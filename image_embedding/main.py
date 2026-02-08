import os
import sys
import time
import requests
import psycopg2
from psycopg2 import extras
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from datetime import datetime

# 환경 변수 로드
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
# 안정성을 위해 가장 표준적인 CLIP 모델로 변경
MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")

class ImageEmbedder:
    def __init__(self):
        self.conn = None
        self.connect_db()
        self.init_table() # DB 연결 직후 테이블 생성 (모델 로딩 전)

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 모델 및 프로세서 로딩 중: {MODEL_NAME}...")
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
            self.model.eval()
            print(f"모델 로딩 완료. (장치: {self.device})")
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            print("힌트: 인터넷 연결을 확인하거나, 이전에 설치된 라이브러리 충돌일 수 있습니다.")
            sys.exit(1)

    def connect_db(self):
        try:
            self.conn = psycopg2.connect(DATABASE_URL)
            self.conn.autocommit = True
        except Exception as e:
            print(f"데이터베이스 연결 실패: {e}")
            sys.exit(1)

    def init_table(self):
        with self.conn.cursor() as cur:
            # CREATE TABLE IF NOT EXISTS를 사용하여 기존 데이터 보존
            cur.execute("""
                CREATE TABLE IF NOT EXISTS mcat2.problem_image_embeddings (
                    id SERIAL PRIMARY KEY,
                    problem_id UUID UNIQUE NOT NULL,
                    system_id VARCHAR(255),
                    image_embedding FLOAT8[] NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_problem_id ON mcat2.problem_image_embeddings(problem_id);
            """)

    def get_stats(self):
        with self.conn.cursor() as cur:
            # 마지막 실행 날짜
            cur.execute("SELECT MAX(updated_at) FROM mcat2.problem_image_embeddings")
            last_date = cur.fetchone()[0]
            
            # 현재 저장된 총 임베딩 수
            cur.execute("SELECT COUNT(*) FROM mcat2.problem_image_embeddings")
            total_count = cur.fetchone()[0]
            
            # 신규 문항 수 (mcat2.problems_problem 테이블에서 problem_image_url이 있고, 아직 임베딩이 없는 것)
            cur.execute("""
                SELECT COUNT(*) 
                FROM mcat2.problems_problem p
                LEFT JOIN mcat2.problem_image_embeddings e ON p.id = e.problem_id
                WHERE e.problem_id IS NULL AND p.problem_image_url IS NOT NULL AND p.problem_image_url != ''
            """)
            new_count = cur.fetchone()[0]
            
            return last_date, total_count, new_count

    def process_batch(self):
        last_date, total_count, new_count = self.get_stats()
        
        print("\n" + "="*40)
        print(f" 마지막 실행: {last_date if last_date else '기록 없음'}")
        print(f" 완료된 문항: {total_count:,} 개")
        print(f" 신규 문항수: {new_count:,} 개")
        print("="*40 + "\n")
        
        if new_count == 0:
            print("새로 처리할 문항이 없습니다.")
            return

        # 신규 문항 가져오기
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT p.id, p.problem_image_url 
                FROM mcat2.problems_problem p
                LEFT JOIN mcat2.problem_image_embeddings e ON p.id = e.problem_id
                WHERE e.problem_id IS NULL AND p.problem_image_url IS NOT NULL AND p.problem_image_url != ''
            """)
            
            pbar = tqdm(total=new_count, desc="임베딩 생성 중")
            
            while True:
                rows = cur.fetchmany(10) # 10개씩 배치 처리
                if not rows:
                    break
                
                batch_data = []
                for pid, url in rows:
                    try:
                        # 이미지 다운로드
                        response = requests.get(url, timeout=10)
                        if response.status_code == 200:
                            img = Image.open(BytesIO(response.content)).convert("RGB")
                            
                            # 임베딩 생성 (CLIP 방식)
                            with torch.no_grad():
                                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                                outputs = self.model.get_image_features(**inputs)
                                
                                # 모델 버전에 따라 출력이 객체일 경우 pooler_output 추출
                                if hasattr(outputs, "pooler_output"):
                                    image_features = outputs.pooler_output
                                elif isinstance(outputs, (list, tuple)):
                                    image_features = outputs[0]
                                else:
                                    image_features = outputs
                                    
                                # L2 정규화
                                image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
                                embedding = image_features[0].cpu().numpy().tolist()
                            
                            batch_data.append((pid, MODEL_NAME, embedding))
                        else:
                            print(f"\n[오류] 이미지 로드 실패 (ID: {pid}, URL: {url})")
                    except Exception as e:
                        print(f"\n[오류] 처리 중 에러 발생 (ID: {pid}): {e}")
                    
                    pbar.update(1)
                
                # DB 저장
                if batch_data:
                    self.save_embeddings(batch_data)
            
            pbar.close()
            
            # 최종 통계 출력
            final_last_date, final_total_count, final_new_count = self.get_stats()
            print("\n" + "="*40)
            print(" [작업 완료 보고]")
            print(f" 마지막 update : {final_last_date if final_last_date else '기록 없음'}")
            print(f" 현재 imbedding 완료수 : {final_total_count:,} 개")
            print(f" 추가할 imbedding 수 : {final_new_count:,} 개")
            print("="*40 + "\n")
            
            print("모든 작업이 완료되었습니다.")

    def save_embeddings(self, data):
        with self.conn.cursor() as cur:
            extras.execute_values(
                cur,
                "INSERT INTO mcat2.problem_image_embeddings (problem_id, system_id, image_embedding) VALUES %s ON CONFLICT (problem_id) DO NOTHING",
                data
            )

if __name__ == "__main__":
    embedder = ImageEmbedder()
    embedder.process_batch()
