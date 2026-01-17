# main.py
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    ensure_dir,
    load_qa_excel,
    extract_questions,
    build_text_embeddings,
    save_numpy_vectors,
    save_feature_preview_excel,
    quick_similarity_sanity_check,
    choose_best_k_by_silhouette,
    run_minibatch_kmeans,
    save_dataset_cluster_excel,
    save_cluster_summary_excel,
    save_cluster_answer_review_excel,   # NEW (Step3)
    get_embedding_model_name,
)
from src.agent import add_cluster_names_to_file

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
VECTORIZED_DATA_DIR = os.path.join(PROJECT_ROOT, "vectorized_data")

# 统计信息存储
stats_info = {
    'stage_times': {},
    'data_scales': {}
}


def stage1_feature_extraction_tianmao():
    """处理tianmao数据集的特征提取"""
    start_time = time.time()
    
    ensure_dir(OUTPUT_DIR)
    ensure_dir(VECTORIZED_DATA_DIR)

    dataset_path = os.path.join(DATA_DIR, "meaningful_answer_tianmao.xlsx")
    # tianmao数据集列名已经是标准的"问题"和"回答"
    df = load_qa_excel(dataset_path, column_mapping={"问题": "问题", "回答": "回答"})

    questions = extract_questions(df, question_col="问题")

    vectors, model_info = build_text_embeddings(
        texts=questions,
        model_name=get_embedding_model_name(),
        batch_size=64,
        normalize=True,
        device="cpu"
    )

    npy_path = os.path.join(VECTORIZED_DATA_DIR, "tianmao_question_vectors.npy")
    save_numpy_vectors(vectors, npy_path)

    preview_path = os.path.join(OUTPUT_DIR, "tianmao_feature_preview.xlsx")
    save_feature_preview_excel(
        df=df,
        questions=questions,
        vectors=vectors,
        out_path=preview_path,
        model_info=model_info,
        question_col="问题",
    )

    quick_similarity_sanity_check(questions, vectors, topk=5)

    # 记录统计信息
    stats_info['stage_times']['tianmao_feature_extraction'] = time.time() - start_time
    stats_info['data_scales']['tianmao_dataset'] = len(questions)

    print("\n[OK] Tianmao feature extraction done.")
    print(f" - Saved vectors: {npy_path} (in vectorized_data/)")
    print(f" - Saved preview: {preview_path}")

    return df, questions, vectors


def stage1_feature_extraction_overseas():
    """处理overseas数据集的特征提取"""
    start_time = time.time()
    
    ensure_dir(OUTPUT_DIR)
    ensure_dir(VECTORIZED_DATA_DIR)

    dataset_path = os.path.join(DATA_DIR, "meaningful_answer_overseas.xlsx")
    # overseas数据集需要列名映射
    df = load_qa_excel(dataset_path, column_mapping={"问题": "客户问题", "回答": "客服回复"})

    questions = extract_questions(df, question_col="问题")

    vectors, model_info = build_text_embeddings(
        texts=questions,
        model_name=get_embedding_model_name(),
        batch_size=64,
        normalize=True,
        device="cpu"
    )

    npy_path = os.path.join(VECTORIZED_DATA_DIR, "overseas_question_vectors.npy")
    save_numpy_vectors(vectors, npy_path)

    preview_path = os.path.join(OUTPUT_DIR, "overseas_feature_preview.xlsx")
    save_feature_preview_excel(
        df=df,
        questions=questions,
        vectors=vectors,
        out_path=preview_path,
        model_info=model_info,
        question_col="问题",
    )

    quick_similarity_sanity_check(questions, vectors, topk=5)

    # 记录统计信息
    stats_info['stage_times']['overseas_feature_extraction'] = time.time() - start_time
    stats_info['data_scales']['overseas_dataset'] = len(questions)

    print("\n[OK] Overseas feature extraction done.")
    print(f" - Saved vectors: {npy_path} (in vectorized_data/)")
    print(f" - Saved preview: {preview_path}")

    return df, questions, vectors


def stage2_clustering_dataset1(
    k_min: int = 5,
    k_max: int = 30,
    choose_k: bool = True,
    fixed_k: int = 12,
):
    """
    Task1-Step(2): clustering execution (MiniBatchKMeans)
    Outputs:
      output/dataset1_cluster.xlsx
      output/dataset1_cluster_summary.xlsx
    """
    ensure_dir(OUTPUT_DIR)

    dataset_path = os.path.join(DATA_DIR, "dataset1.xlsx")
    df = load_qa_excel(dataset_path)
    questions = extract_questions(df, question_col="问题")

    vectors, _ = build_text_embeddings(
        texts=questions,
        model_name=get_embedding_model_name(),
        batch_size=64,
        normalize=True,
        device="cpu"
    )

    # 1) choose best k
    if choose_k:
        best_k, eval_df = choose_best_k_by_silhouette(
            vectors=vectors,
            k_min=k_min,
            k_max=k_max,
            sample_size=4000,
            random_state=42
        )
        print("\n[K-Selection] silhouette scores:")
        print(eval_df.to_string(index=False))
        print(f"\n[K-Selection] best_k={best_k}")
        k = best_k
    else:
        k = fixed_k
        print(f"\n[K-Selection] use fixed_k={k}")

    # 2) run clustering
    labels, _model = run_minibatch_kmeans(
        vectors=vectors,
        n_clusters=k,
        random_state=42,
        batch_size=1024,
        max_iter=200
    )

    # 3) save per-row cluster result
    out_cluster_path = os.path.join(OUTPUT_DIR, "dataset1_cluster.xlsx")
    save_dataset_cluster_excel(
        df=df,
        questions=questions,
        labels=labels,
        out_path=out_cluster_path,
        question_col="问题",
        answer_col="回答"
    )

    # 4) save cluster summary for manual review (questions-focused)
    out_summary_path = os.path.join(OUTPUT_DIR, "dataset1_cluster_summary.xlsx")
    save_cluster_summary_excel(
        df=df,
        questions=questions,
        vectors=vectors,
        labels=labels,
        out_path=out_summary_path,
        top_examples=99999  # 显示所有属于该聚类的问题
    )

    print("\n[OK] Stage2 clustering done.")
    print(f" - Saved clustered dataset: {out_cluster_path}")
    print(f" - Saved cluster summary:   {out_summary_path}")

    return out_cluster_path


def stage1_merge_datasets():
    """合并所有数据集进行统一聚类分析"""
    start_time = time.time()
    
    ensure_dir(OUTPUT_DIR)
    ensure_dir(VECTORIZED_DATA_DIR)
    
    print("\n[Merge] 开始合并多数据集...")
    
    # 处理tianmao数据集
    print("[Merge] 处理tianmao数据集...")
    df_tianmao, questions_tianmao, vectors_tianmao = stage1_feature_extraction_tianmao()
    df_tianmao['source_dataset'] = 'tianmao'
    
    # 处理overseas数据集
    print("[Merge] 处理overseas数据集...")
    df_overseas, questions_overseas, vectors_overseas = stage1_feature_extraction_overseas()
    df_overseas['source_dataset'] = 'overseas'
    
    # 合并数据
    print("[Merge] 合并数据...")
    all_dfs = [df_tianmao, df_overseas]
    all_questions = questions_tianmao + questions_overseas
    all_vectors = np.vstack([vectors_tianmao, vectors_overseas])
    
    # 合并DataFrame，保持索引连续
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"[Merge] 合并完成:")
    print(f" - Tianmao: {len(questions_tianmao)} 个问题")
    print(f" - Overseas: {len(questions_overseas)} 个问题")
    print(f" - 总计: {len(all_questions)} 个问题")
    
    # 保存合并后的向量
    merged_vectors_path = os.path.join(VECTORIZED_DATA_DIR, "merged_question_vectors.npy")
    save_numpy_vectors(all_vectors, merged_vectors_path)
    
    # 保存合并后的特征预览
    merged_preview_path = os.path.join(OUTPUT_DIR, "merged_feature_preview.xlsx")
    save_feature_preview_excel(
        df=merged_df,
        questions=all_questions,
        vectors=all_vectors,
        out_path=merged_preview_path,
        model_info={'backend': 'sentence-transformers', 'model_name': get_embedding_model_name(), 'device': 'cpu', 'normalize': True, 'dim': 384, 'count': len(all_questions)},
        question_col="问题",
    )
    
    quick_similarity_sanity_check(all_questions, all_vectors, topk=5)
    
    # 记录统计信息
    stats_info['stage_times']['merge_datasets'] = time.time() - start_time
    stats_info['data_scales']['merged_dataset'] = len(all_questions)
    
    print(f"\n[OK] 数据集合并完成.")
    print(f" - 保存合并向量: {merged_vectors_path} (in vectorized_data/)")
    print(f" - 保存合并预览: {merged_preview_path}")
    
    return merged_df, all_questions, all_vectors


def stage2_clustering_merged(
    k_min: int = 5,
    k_max: int = 50,
    choose_k: bool = True,
    fixed_k: int = 30,
):
    """
    对合并数据集进行聚类分析
    """
    start_time = time.time()
    
    ensure_dir(OUTPUT_DIR)
    ensure_dir(VECTORIZED_DATA_DIR)

    # 加载合并后的数据
    merged_vectors_path = os.path.join(VECTORIZED_DATA_DIR, "merged_question_vectors.npy")
    if not os.path.exists(merged_vectors_path):
        print("[Error] 合并向量文件不存在，请先运行stage1_merge_datasets()")
        return None

    vectors = np.load(merged_vectors_path)
    
    # 重新构建合并的DataFrame和问题列表
    df_tianmao, questions_tianmao, _ = stage1_feature_extraction_tianmao()
    df_overseas, questions_overseas, _ = stage1_feature_extraction_overseas()
    
    df_tianmao['source_dataset'] = 'tianmao'
    df_overseas['source_dataset'] = 'overseas'
    
    merged_df = pd.concat([df_tianmao, df_overseas], ignore_index=True)
    all_questions = questions_tianmao + questions_overseas

    # 1) choose best k
    if choose_k:
        best_k, eval_df = choose_best_k_by_silhouette(
            vectors=vectors,
            k_min=k_min,
            k_max=k_max,
            sample_size=4000,
            random_state=42
        )
        print("\n[K-Selection] silhouette scores:")
        print(eval_df.to_string(index=False))
        print(f"\n[K-Selection] best_k={best_k}")
        k = best_k
    else:
        k = fixed_k
        print(f"\n[K-Selection] use fixed_k={k}")

    # 2) run clustering
    labels, _model = run_minibatch_kmeans(
        vectors=vectors,
        n_clusters=k,
        random_state=42,
        batch_size=1024,
        max_iter=200
    )

    # 3) save per-row cluster result
    out_cluster_path = os.path.join(OUTPUT_DIR, "merged_cluster.xlsx")
    save_dataset_cluster_excel(
        df=merged_df,
        questions=all_questions,
        labels=labels,
        out_path=out_cluster_path,
        question_col="问题",
        answer_col="回答"
    )

    # 4) save cluster summary for manual review
    out_summary_path = os.path.join(OUTPUT_DIR, "merged_cluster_summary.xlsx")
    save_cluster_summary_excel(
        df=merged_df,
        questions=all_questions,
        vectors=vectors,
        labels=labels,
        out_path=out_summary_path,
        top_examples=99999  # 显示所有属于该聚类的问题
    )

    # 记录统计信息
    stats_info['stage_times']['clustering_merged'] = time.time() - start_time
    stats_info['data_scales']['clustering_k'] = k

    print("\n[OK] 合并数据集聚类完成.")
    print(f" - 保存聚类数据集: {out_cluster_path}")
    print(f" - 保存聚类摘要: {out_summary_path}")

    return out_cluster_path


def stage3_answer_grouping_merged():
    """
    对合并数据集的聚类结果进行答案分组
    """
    start_time = time.time()
    
    ensure_dir(OUTPUT_DIR)

    clustered_path = os.path.join(OUTPUT_DIR, "merged_cluster.xlsx")
    if not os.path.exists(clustered_path):
        raise FileNotFoundError(
            "merged_cluster.xlsx not found. Please run stage2_clustering_merged() first."
        )

    df_clustered = pd.read_excel(clustered_path, engine="openpyxl")

    # Ensure we have cleaned question column
    if "问题_clean" not in df_clustered.columns:
        if "问题" in df_clustered.columns:
            df_clustered["问题_clean"] = df_clustered["问题"].astype(str)
        else:
            raise ValueError("No question column found in merged_cluster.xlsx")

    out_path = os.path.join(OUTPUT_DIR, "merged_cluster_answers.xlsx")
    save_cluster_answer_review_excel(
        df_clustered=df_clustered,
        out_path=out_path,
        cluster_col="cluster_id",
        question_col="问题_clean",
        answer_col="回答",
        top_questions=99999,
        top_answers=99999,
        answer_sep="\n---\n",
    )

    print("\n[OK] 合并数据集答案分组完成.")
    print(f" - 保存答案审核文件: {out_path}")
    
    # 添加聚类中文标签
    add_cluster_names_to_file(out_path)
    
    # 记录统计信息
    stats_info['stage_times']['answer_grouping_merged'] = time.time() - start_time

    return out_path


def stage3_answer_grouping_dataset1():
    """
    Task1-Step(3): Answer grouping & validation export (dataset1 only)
    Input:
      output/dataset1_cluster.xlsx
    Output:
      output/dataset1_cluster_answers.xlsx
    """
    ensure_dir(OUTPUT_DIR)

    clustered_path = os.path.join(OUTPUT_DIR, "dataset1_cluster.xlsx")
    if not os.path.exists(clustered_path):
        raise FileNotFoundError(
            "dataset1_cluster.xlsx not found. Please run stage2_clustering_dataset1() first."
        )

    df_clustered = pd.read_excel(clustered_path, engine="openpyxl")

    # Ensure we have cleaned question column
    if "问题_clean" not in df_clustered.columns:
        # Backward compatibility: if your file uses other name, try to recover
        if "问题_clean" not in df_clustered.columns and "问题_clean" not in df_clustered.columns:
            # If only 原始“问题” exists, treat it as clean for this export
            if "问题" in df_clustered.columns:
                df_clustered["问题_clean"] = df_clustered["问题"].astype(str)
            else:
                raise ValueError("No question column found in dataset1_cluster.xlsx")

    out_path = os.path.join(OUTPUT_DIR, "dataset1_cluster_answers.xlsx")
    save_cluster_answer_review_excel(
        df_clustered=df_clustered,
        out_path=out_path,
        cluster_col="cluster_id",
        question_col="问题_clean",
        answer_col="回答",
        top_questions=99999,
        top_answers=99999,
        answer_sep="\n---\n",
    )

    print("\n[OK] Stage3 answer grouping done.")
    print(f" - Saved answer review file: {out_path}")
    
    # 添加聚类中文标签
    add_cluster_names_to_file(out_path)


def print_final_statistics():
    """打印最终统计信息"""
    print("\n" + "=" * 80)
    print("📊 数据处理统计信息")
    print("=" * 80)
    
    print("\n📈 数据规模统计:")
    print(f"  • 天猫数据集: {stats_info['data_scales'].get('tianmao_dataset', 0):,} 个问题")
    print(f"  • 海外数据集: {stats_info['data_scales'].get('overseas_dataset', 0):,} 个问题")
    print(f"  • 合并数据集: {stats_info['data_scales'].get('merged_dataset', 0):,} 个问题")
    print(f"  • 聚类簇数量: {stats_info['data_scales'].get('clustering_k', 0)} 个")
    
    print("\n⏱️  各阶段耗时统计:")
    stage_names = {
        'tianmao_feature_extraction': '📊 天猫数据特征提取',
        'overseas_feature_extraction': '📊 海外数据特征提取',
        'merge_datasets': '📊 数据集合并',
        'clustering_merged': '🎯 合并数据集聚类',
        'answer_grouping_merged': '📝 答案分组和标签生成'
    }
    
    total_time = 0
    for stage_key, stage_name in stage_names.items():
        duration = stats_info['stage_times'].get(stage_key, 0)
        total_time += duration
        minutes, seconds = divmod(duration, 60)
        print(f"  • {stage_name}: {minutes:.0f}分{seconds:.1f}秒")
    
    total_minutes, total_seconds = divmod(total_time, 60)
    print(f"\n🚀 总计用时: {total_minutes:.0f}分{total_seconds:.1f}秒")
    
    print("=" * 80)


if __name__ == "__main__":
    print("=" * 80)
    print("QA聚类分析 - 多数据集处理流程")
    print("=" * 80)
    
    # 处理合并的新数据集 (tianmao + overseas)
    print("\n>>> 阶段1: 数据集合并和特征提取")
    stage1_merge_datasets()
    
    print("\n>>> 阶段2: 合并数据集聚类分析")
    stage2_clustering_merged(choose_k=False, fixed_k=30)
    
    print("\n>>> 阶段3: 合并数据集答案分组和标签生成")
    stage3_answer_grouping_merged()
    
    print("\n" + "=" * 80)
    print("所有处理完成！")
    print("输出文件:")
    print("  - merged_cluster_answers.xlsx: 最终合并聚类结果（包含中文标签）")
    print("  - merged_cluster.xlsx: 原始聚类分配")
    print("  - merged_cluster_summary.xlsx: 聚类摘要")
    print("=" * 80)
    
    # 打印统计信息
    print_final_statistics()
