# api_main.py
import sys
from fastapi import FastAPI

# 添加项目根目录到Python路径
sys.path.insert(0, '..')

from api.router.search_api import router as search_router

app = FastAPI(title="QA Vector Search API")
app.include_router(search_router)

@app.get("/health")
def health():
    return {"status": "ok"}
