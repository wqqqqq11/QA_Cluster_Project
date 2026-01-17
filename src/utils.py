# utils.py
import os
import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_qa_excel(xlsx_path: str, column_mapping: dict = None) -> pd.DataFrame:
    """
    加载QA Excel文件，支持不同的列名映射
    
    Args:
        xlsx_path: Excel文件路径
        column_mapping: 列名映射字典，格式: {"问题": "实际问题列名", "回答": "实际回答列名"}
    
    Returns:
        标准化后的DataFrame，包含"问题"和"回答"列
    """
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Excel not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path, engine="openpyxl")
    
    # 如果没有提供映射，使用默认的列名
    if column_mapping is None:
        column_mapping = {"问题": "问题", "回答": "回答"}
    
    # 检查映射的列是否存在
    question_col = column_mapping.get("问题", "问题")
    answer_col = column_mapping.get("回答", "回答")
    
    if question_col not in df.columns:
        raise ValueError(f"Excel文件中缺少问题列: {question_col}. 可用列: {list(df.columns)}")
    if answer_col not in df.columns:
        raise ValueError(f"Excel文件中缺少回答列: {answer_col}. 可用列: {list(df.columns)}")
    
    # 创建标准化的DataFrame
    standardized_df = df.copy()
    
    # 如果列名不是标准名称，则重命名
    if question_col != "问题":
        standardized_df["问题"] = standardized_df[question_col]
    if answer_col != "回答":
        standardized_df["回答"] = standardized_df[answer_col]
    
    # 确保字符串类型
    standardized_df["问题"] = standardized_df["问题"].astype(str)
    standardized_df["回答"] = standardized_df["回答"].astype(str)
    
    return standardized_df


def _clean_text(s: str) -> str:
    """
    Minimal, safe cleaning for customer service QA:
    - strip spaces
    - collapse multiple spaces/newlines
    - remove invisible chars
    """
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u200b", " ").replace("\ufeff", " ")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_questions(df: pd.DataFrame, question_col: str = "问题") -> List[str]:
    questions = [_clean_text(x) for x in df[question_col].tolist()]
    # Optional: handle empty questions
    questions = [q if q else "[EMPTY_QUESTION]" for q in questions]
    return questions


def build_text_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int = 64,
    normalize: bool = True,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Build semantic embeddings using SentenceTransformer.

    Requirements:
      pip install sentence-transformers torch

    Returns:
      vectors: np.ndarray shape (n, dim)
      model_info: dict for logging/preview
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required. Install with:\n"
            "  pip install sentence-transformers torch\n"
        ) from e

    # device auto
    if device is None:
        # SentenceTransformer will auto-pick cuda if available, but we keep info
        device = "cuda" if _torch_cuda_available() else "cpu"

    # 检查是否存在本地模型路径
    local_model_path = f"/home/projects/QA_Cluster_Project/models--sentence-transformers--{model_name}"
    if os.path.exists(local_model_path):
        print(f"[Utils] 从本地路径加载模型: {local_model_path}")
        model = SentenceTransformer(local_model_path, device=device)
    else:
        print(f"[Utils] 从Hugging Face加载模型: {model_name}")
        model = SentenceTransformer(model_name, device=device)

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    model_info = {
        "backend": "sentence-transformers",
        "model_name": model_name,
        "device": device,
        "normalize": normalize,
        "dim": int(vectors.shape[1]) if len(vectors.shape) == 2 else None,
        "count": int(vectors.shape[0]),
    }
    return vectors, model_info


def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def save_numpy_vectors(vectors: np.ndarray, out_path: str) -> None:
    # vectors should be float32 for storage efficiency
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    np.save(out_path, vectors)


def save_feature_preview_excel(
    df: pd.DataFrame,
    questions: List[str],
    vectors: np.ndarray,
    out_path: str,
    model_info: Dict,
    question_col: str = "问题",
    preview_dims: int = 8,
) -> None:
    """
    Export a human-checkable excel:
      - original metadata columns
      - cleaned question
      - vector_dim, vector_norm
      - first N dims preview: v0..vN
    """
    preview_df = df.copy()
    preview_df[f"{question_col}_clean"] = questions

    # norms (if normalized, should be close to 1.0)
    norms = np.linalg.norm(vectors, axis=1)
    preview_df["vector_dim"] = vectors.shape[1]
    preview_df["vector_norm"] = norms

    # preview first dims
    d = min(preview_dims, vectors.shape[1])
    for i in range(d):
        preview_df[f"v{i}"] = vectors[:, i]

    # attach model info as top rows in a separate sheet
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        preview_df.to_excel(writer, index=False, sheet_name="dataset1_preview")
        info_df = pd.DataFrame([model_info])
        info_df.to_excel(writer, index=False, sheet_name="embedding_info")


def quick_similarity_sanity_check(
    questions: List[str],
    vectors: np.ndarray,
    topk: int = 5,
    sample_n: int = 5,
) -> None:
    """
    Quick check embeddings are meaningful:
    - Randomly pick a few questions
    - Retrieve topk most similar (cosine, because vectors normalized)
    """
    if vectors.ndim != 2 or len(questions) != vectors.shape[0]:
        print("[WARN] similarity check skipped: shape mismatch")
        return

    n = vectors.shape[0]
    if n < 2:
        print("[WARN] similarity check skipped: not enough samples")
        return

    # cosine similarity for normalized vectors: sim = dot
    rng = np.random.default_rng(42)
    idxs = rng.choice(n, size=min(sample_n, n), replace=False)

    print("\n[SanityCheck] Similarity retrieval preview:")
    for idx in idxs:
        q = questions[idx]
        sims = vectors @ vectors[idx]  # (n,)
        # exclude itself
        sims[idx] = -1.0
        top_idx = np.argsort(-sims)[:topk]

        print("\n----------------------------------------")
        print(f"[Query] ({idx}) {q}")
        for rank, j in enumerate(top_idx, start=1):
            print(f"  Top{rank}: sim={sims[j]:.4f} | ({j}) {questions[j]}")

def choose_best_k_by_silhouette(
    vectors: np.ndarray,
    k_min: int = 5,
    k_max: int = 30,
    sample_size: int = 4000,
    random_state: int = 42,
) -> Tuple[int, pd.DataFrame]:
    """
    Use silhouette score to choose best K.
    - To speed up, optionally subsample for silhouette calculation.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    n = vectors.shape[0]
    rng = np.random.default_rng(random_state)

    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        X_eval = vectors[idx]
    else:
        X_eval = vectors

    records = []
    best_k = None
    best_score = -1.0

    for k in range(k_min, k_max + 1):
        if k <= 1 or k >= X_eval.shape[0]:
            continue

        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=1024,
            n_init="auto",
            max_iter=200,
        )
        labels = km.fit_predict(X_eval)

        # silhouette requires >1 cluster and not all points in one cluster
        if len(set(labels)) < 2:
            score = -1.0
        else:
            score = float(silhouette_score(X_eval, labels, metric="cosine"))

        records.append({"k": k, "silhouette_cosine": score})
        if score > best_score:
            best_score = score
            best_k = k

    eval_df = pd.DataFrame(records).sort_values("k")
    if best_k is None:
        # fallback: pick a safe mid value
        best_k = max(k_min, 2)

    return best_k, eval_df


def run_minibatch_kmeans(
    vectors: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    batch_size: int = 1024,
    max_iter: int = 200,
):
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        n_init="auto",
        max_iter=max_iter
    )
    labels = km.fit_predict(vectors)
    return labels, km


def save_dataset_cluster_excel(
    df: pd.DataFrame,
    questions: list,
    labels: np.ndarray,
    out_path: str,
    question_col: str = "问题",
    answer_col: str = "回答",
) -> None:
    """
    Save dataset1 cluster result:
    - original columns
    - 问题_clean
    - cluster_id
    """
    out_df = df.copy()
    out_df[f"{question_col}_clean"] = questions
    out_df["cluster_id"] = labels.astype(int)
    out_df.to_excel(out_path, index=False, engine="openpyxl")


def save_cluster_summary_excel(
    df: pd.DataFrame,
    questions: list,
    vectors: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    top_examples: int = 8,
) -> None:
    """
    Produce a human-review summary per cluster:
    - cluster_id
    - count
    - representative questions (closest to centroid)
    """
    # centroid = mean of member vectors
    label_ids = np.unique(labels)
    rows = []

    for cid in label_ids:
        idx = np.where(labels == cid)[0]
        count = len(idx)
        X = vectors[idx]
        centroid = X.mean(axis=0)

        # vectors are normalized; similarity dot
        sims = X @ centroid
        order = np.argsort(-sims)[:min(top_examples, count)]
        reps = [questions[idx[i]] for i in order]

        rows.append({
            "cluster_id": int(cid),
            "count": int(count),
            "representative_questions": "\n".join(reps)
        })

    summary_df = pd.DataFrame(rows).sort_values(["count", "cluster_id"], ascending=[False, True])

    # also export distribution sheet
    dist_df = summary_df[["cluster_id", "count"]].copy()

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="cluster_summary")
        dist_df.to_excel(writer, index=False, sheet_name="cluster_distribution")

# ========= Stage2-B: 跨数据集类别体系整合 =========

def build_cluster_texts(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    question_col: str = "问题_clean",
    topn: int = 10,
):
    """
    每个 cluster → 一个文本（代表该类的语义）
    """
    cluster_texts = {}

    for cid, g in df.groupby(cluster_col):
        qs = g[question_col].tolist()[:topn]
        cluster_texts[cid] = "；".join(qs)

    return cluster_texts


def merge_clusters_by_similarity(
    cluster_vectors: np.ndarray,
    threshold: float = 0.85,
):
    """
    基于 cosine similarity 的簇合并
    返回：cluster_index -> global_cluster_id
    """
    sim = cluster_vectors @ cluster_vectors.T
    n = sim.shape[0]

    global_ids = [-1] * n
    gid = 0

    for i in range(n):
        if global_ids[i] != -1:
            continue
        global_ids[i] = gid
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                global_ids[j] = gid
        gid += 1

    return global_ids


def save_global_cluster_result(
    df_all: pd.DataFrame,
    out_path: str,
):
    """
    保存最终全局聚类结果
    """
    df_all.to_excel(out_path, index=False, engine="openpyxl")

def _clean_answer_text(a: str) -> str:
    if a is None:
        return ""
    a = str(a).replace("\u200b", " ").replace("\ufeff", " ")
    a = a.strip()
    a = re.sub(r"\s+\n", "\n", a)
    a = re.sub(r"\n{3,}", "\n\n", a)
    return a


def save_cluster_answer_review_excel(
    df_clustered: pd.DataFrame,
    out_path: str,
    cluster_col: str = "cluster_id",
    question_col: str = "问题_clean",
    answer_col: str = "回答",
    top_questions: int = 8,
    top_answers: int = 12,
    answer_sep: str = "\n---\n",
) -> None:
    """
    Task1-Step(3): Answer grouping & review export (dataset1 only)
    Output Excel with 2 sheets:
      1) cluster_answer_view: per-cluster rollup for manual review
      2) cluster_answer_raw: row-level raw QA with cluster_id
    """
    required = {cluster_col, question_col, answer_col}
    missing = [c for c in required if c not in df_clustered.columns]
    if missing:
        raise ValueError(f"Missing required columns in clustered df: {missing}")

    # Raw sheet (keep only essential columns but preserve readability)
    raw_df = df_clustered.copy()
    raw_df[answer_col] = raw_df[answer_col].apply(_clean_answer_text)

    # View sheet (grouped) - 显示每个聚类的所有问题和回答
    rows = []
    for cid, g in raw_df.groupby(cluster_col):
        qs = [str(x) for x in g[question_col].tolist() if str(x).strip()]
        ans = [str(x) for x in g[answer_col].tolist() if str(x).strip()]

        # 显示所有问题（不限制数量）
        all_questions = "\n".join(qs[:top_questions]) if top_questions < len(qs) else "\n".join(qs)

        # 显示所有回答（不限制数量）  
        all_answers = answer_sep.join(ans[:top_answers]) if top_answers < len(ans) else answer_sep.join(ans)

        rows.append({
            "cluster_id": int(cid),
            "qa_count": int(len(g)),
            "all_questions": all_questions,
            "all_answers": all_answers,
            "cluster_name": "",  # 预留聚类标签字段，将由agent.py填充
        })

    view_df = pd.DataFrame(rows).sort_values(["qa_count", "cluster_id"], ascending=[False, True])

    # Export
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        view_df.to_excel(writer, index=False, sheet_name="cluster_answer_view")
        raw_df.to_excel(writer, index=False, sheet_name="cluster_answer_raw")


def load_config() -> dict:
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    import json
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_weaviate_url() -> str:
    """获取Weaviate URL配置"""
    config = load_config()
    return config.get("weaviate_url", "http://localhost:8080")


def get_embedding_model_name() -> str:
    """获取嵌入模型名称配置"""
    config = load_config()
    return config.get("embedding_model_name", "paraphrase-multilingual-MiniLM-L12-v2")


def get_clip_model_name() -> str:
    """获取CLIP模型名称配置"""
    config = load_config()
    return config.get("clip_model_name", "clip-ViT-B-32-multilingual-v1")
