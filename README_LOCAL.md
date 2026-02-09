# 🚀 로컬 고속 인덱서 (Local Indexer) 사용 가이드

RTX 5080 등 고사양 로컬 PC를 활용하여 6만 건의 데이터를 서버보다 훨씬 빠르게 처리할 수 있습니다.

## 1. 사전 준비 (로컬 PC)
- **Python 설치**: 3.9 ~ 3.11 버전 추천
- **NVIDIA Driver**: 최신 버전으로 업데이트
- **CUDA Toolkit**: 11.8 이상 권장

## 2. 필수 라이브러리 설치
로컬 터미널(CMD 또는 PowerShell)에서 아래 명령어를 실행하세요.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pix2text psycopg2-binary requests pillow transformers onnxruntime-gpu
```

## 3. 실행 방법
1. 서버에서 `git pull`을 받아 `local_indexer.py` 파일을 로컬 컴퓨터로 가져옵니다. (또는 제가 드린 코드를 복사해서 파일을 만드셔도 됩니다.)
2. `local_indexer.py` 내의 `DB_URL`이 현재 서버 주소와 맞는지 확인합니다.
3. 터미널에서 실행합니다.
```bash
python local_indexer.py
```

## 4. 특징
- **병렬 다운로드**: 네트워크 대기 시간을 최소화합니다.
- **GPU 배치 처리**: RTX 5080의 강력한 성능을 100% 활용합니다.
- **원격 업로드**: 결과값만 서버 DB로 즉시 전송하므로, 서버를 켜둔 채로 도움을 줄 수 있습니다.

---
**주의**: 로컬에서 돌리는 동안에도 서버의 Admin 페이지에서 업데이트를 동시에 누를 수 있습니다. (중복 방지 로직이 있어 안전합니다.)
