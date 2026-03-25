import os
from dotenv import load_dotenv

load_dotenv()

# Database Configuration (Easy to change for migration)
DATABASE_TYPE = os.getenv("DATABASE_TYPE", "sqlite")  # sqlite, postgresql, mysql

if DATABASE_TYPE == "sqlite":
    DATABASE_URL = "sqlite:///chat_history.db"
elif DATABASE_TYPE == "postgresql":
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/potato_db")
elif DATABASE_TYPE == "mysql":
    DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://user:password@localhost/potato_db")
else:
    DATABASE_URL = "sqlite:///chat_history.db"

# API Configuration
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# CORS Configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# RAG Configuration
ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "True").lower() == "true"
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", 5))