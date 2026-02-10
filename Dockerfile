# 1. Base Image - NVIDIA CUDA + cuDNN 지원 이미지 사용 (ONNX 가속 필수)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV PYTHONUNBUFFERED=1

# 2. 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# GPU 라이브러리 경로 강제 설정 (CUDA 11.8 및 cuDNN)
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

RUN pip3 install --upgrade pip setuptools wheel

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# [CRITICAL] 일반 onnxruntime과 충돌 방지: 기존 것 삭제(없어도 통과) 후 GPU 전용 설치
RUN (pip3 uninstall -y onnxruntime onnxruntime-gpu || true) && \
    pip3 install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY . .

# 6. 포트 노출
EXPOSE 8000

# 7. 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
