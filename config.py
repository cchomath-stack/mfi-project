import os

# Google OAuth Credentials
# 클라우드 배포 시에는 서버 환경 변수에서 값을 가져옵니다.
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "892646063722-nf6ja53jafafipqgk72qlrvlgkj0qjp1.apps.googleusercontent.com")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "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")
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")

# JWT Settings
SECRET_KEY = os.environ.get("SECRET_KEY", "antigravity_secret_key_change_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

# Database Settings
DB_URL = os.environ.get("DB_URL", "postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki")
