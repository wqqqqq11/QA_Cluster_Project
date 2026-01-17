# router/search_api.py
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.vector_db import QAVectorDB
from src.utils import get_weaviate_url

router = APIRouter(tags=["search"])

# ---- 1) 全局单例 ----
_VECTOR_DB: Optional[QAVectorDB] = None


def get_vector_db() -> QAVectorDB:
    """
    懒加载 QAVectorDB（避免每个请求都重复加载模型）
    """
    global _VECTOR_DB
    if _VECTOR_DB is None:
        weaviate_url = os.getenv("WEAVIATE_URL", get_weaviate_url())
        
        _VECTOR_DB = QAVectorDB(weaviate_url=weaviate_url, device="cpu")
    return _VECTOR_DB


# ---- 2) 响应结构 ----
class SearchItem(BaseModel):
    similarity_score: float = Field(..., description="0-100，越高越相似")
    source_dataset: str
    question: str
    answer: str
    # image_url: str
    image_url: str = Field("", description="相关的图片URL")  # 使用默认空字符串而不是Optional[str]

class CategoryBlock(BaseModel):
    category_name: str
    items: List[SearchItem]

class SearchResponse(BaseModel):
    search_info: Dict[str, Any]
    categories: List[CategoryBlock]


def build_search_json(
    query: str,
    source_filter: Optional[str],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    json_data: Dict[str, Any] = {
        "search_info": {
            "query": query,
            "source_filter": source_filter if source_filter else "all",
            "timestamp": datetime.now().isoformat(),
            "total_results": len(results),
        },
        "categories": [],
    }

    # 先用 dict 聚合，再转 list
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for r in results:
        category_name = r.get("cluster_name", "未知分类")
        grouped.setdefault(category_name, [])

        distance = r.get("_additional", {}).get("distance")
        similarity_score = round((1 - float(distance)) * 100, 2) if distance is not None else 0.0

        grouped[category_name].append(
            {
                "similarity_score": similarity_score,
                "source_dataset": r.get("source_dataset", ""),
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
                "image_url": r.get("image_url", ""),
            }
        )

    #  categories: [ {category_name, items}, ... ]
    for category_name, items in grouped.items():
        json_data["categories"].append(
            {
                "category_name": category_name,
                "items": items,
            }
        )

    return json_data


@router.get("/api/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Query text"),
    limit: int = Query(5, ge=1, le=50, description="Top K"),
    source: Optional[str] = Query(None, description="Optional: tianmao / overseas"),
    cluster_id: Optional[int] = Query(None, description="Optional cluster filter"),
):
    """
    最小检索接口：
    - q：查询文本
    - limit：返回条数（建议<=50）
    - source：数据源过滤（tianmao/overseas）
    - cluster_id：聚类过滤
    """

    try:
        db = get_vector_db()
        results = db.search(query=q, limit=limit, source_filter=source, cluster_filter=cluster_id)
        logging.info(results[0].keys())
        logging.info(results[0].get("image_url"))

        return build_search_json(q, source, results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

