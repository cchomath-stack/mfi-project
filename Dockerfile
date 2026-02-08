# 1. Base Image - NVIDIA CUDA 지원 파이썬 이미지 사용
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# 2. 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY . .

# 6. 포트 노출
EXPOSE 8000

# 7. 서버 실행
CMD ["uvicorn", "main.py:app", "--host", "0.0.0.0", "--port", "8000"]
