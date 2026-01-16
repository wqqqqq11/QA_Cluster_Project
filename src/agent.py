# agent.py
import os
import pandas as pd
import requests
import json
from typing import List, Dict
from dotenv import load_dotenv

# 加载.env配置
load_dotenv()

def get_cluster_name_from_llm(questions: List[str]) -> str:
    """
    使用大模型API分析聚类问题，生成合适的中文标签
    
    Args:
        questions: 聚类中的问题列表
    
    Returns:
        str: 生成的中文标签
    """
    # 从.env读取配置
    api_key = os.getenv("QIANWEN_API_KEY")
    api_base = os.getenv("QIANWEN_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("QIANWEN_MODEL", "gpt-3.5-turbo")
    
    if not api_key:
        raise ValueError("请在.env文件中设置OPENAI_API_KEY")
    
    # 准备问题样本（最多取前20个问题作为分析样本）
    sample_questions = questions[:20] if len(questions) > 20 else questions
    questions_text = "\n".join([f"- {q}" for q in sample_questions])
    
    # 构建提示词
    prompt = f"""
你是一个客服数据分析专家。请分析以下客服问题聚类，为这个聚类生成一个准确、简洁的中文标签（2-6个字）。

聚类中的问题样本：
{questions_text}

要求：
1. 标签要准确反映这些问题的共同主题
2. 使用2-6个中文字符
3. 适合客服场景
4. 简洁明了，便于理解

请只返回标签文本，不要其他解释。
"""
    
    try:
        # 调用API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 50,
            "temperature": 0.3
        }
        
        response = requests.post(
            f"{api_base}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            cluster_name = result["choices"][0]["message"]["content"].strip()
            # 清理标签，移除可能的引号、标点等
            cluster_name = cluster_name.strip('"\'.,，。！？')
            return cluster_name
        else:
            print(f"API调用失败: {response.status_code}, {response.text}")
            return f"聚类问题"  # 默认标签
            
    except Exception as e:
        print(f"生成聚类标签时出错: {e}")
        return f"聚类问题"  # 默认标签


def add_cluster_names_to_file(excel_path: str) -> None:
    """
    为dataset1_cluster_answers.xlsx文件中的聚类添加中文标签
    
    Args:
        excel_path: Excel文件路径
    """
    try:
        print(f"\n[Agent] 正在为聚类生成中文标签...")
        
        # 读取Excel文件的聚类视图工作表
        df = pd.read_excel(excel_path, engine="openpyxl", sheet_name="cluster_answer_view")
        
        if 'cluster_id' not in df.columns or 'all_questions' not in df.columns:
            print("[Agent] 错误: Excel文件缺少必要的列 (cluster_id, all_questions)")
            return
        
        # 准备聚类数据
        cluster_data = {}
        
        for _, row in df.iterrows():
            cluster_id = row['cluster_id']
            all_questions_text = str(row['all_questions'])
            cluster_data[cluster_id] = all_questions_text.split('\n') if all_questions_text else []
        
        print(f"[Agent] 发现 {len(cluster_data)} 个聚类，正在生成标签...")
        
        # 为每个聚类生成标签
        cluster_names = {}
        for cluster_id, questions in cluster_data.items():
            print(f"[Agent] 处理聚类 {cluster_id} ({len(questions)} 个问题)...")
            
            # 去除空值和重复问题
            questions = list(set([q.strip() for q in questions if q and str(q).strip()]))
            
            if questions:
                cluster_name = get_cluster_name_from_llm(questions)
                cluster_names[cluster_id] = cluster_name
                print(f"[Agent] 聚类 {cluster_id}: {cluster_name}")
            else:
                cluster_names[cluster_id] = "未分类"
        
        # 更新cluster_name列
        df['cluster_name'] = df['cluster_id'].map(cluster_names)
        
        # 读取原始Excel文件，更新cluster_answer_view工作表
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode='a', if_sheet_exists='replace') as writer:
            # 读取原始的cluster_answer_raw工作表
            raw_df = pd.read_excel(excel_path, engine="openpyxl", sheet_name="cluster_answer_raw")
            
            # 保存更新后的视图工作表
            df.to_excel(writer, index=False, sheet_name="cluster_answer_view")
            # 保留原始数据工作表
            raw_df.to_excel(writer, index=False, sheet_name="cluster_answer_raw")
        
        print(f"\n[Agent] 聚类标签生成完成！已更新文件: {excel_path}")
        print("聚类标签映射:")
        for cluster_id, name in sorted(cluster_names.items()):
            count = len(cluster_data[cluster_id])
            print(f"  聚类 {cluster_id}: {name} ({count} 个问题)")
            
    except Exception as e:
        print(f"[Agent] 生成聚类标签时发生错误: {e}")


def preview_cluster_samples(excel_path: str, max_samples: int = 3) -> None:
    """
    预览每个聚类的样本问题（用于调试）
    
    Args:
        excel_path: Excel文件路径
        max_samples: 每个聚类显示的样本数量
    """
    try:
        df = pd.read_excel(excel_path, engine="openpyxl", sheet_name="cluster_answer_view")
        
        print(f"\n[Agent] 聚类样本预览:")
        for _, row in df.iterrows():
            cluster_id = row['cluster_id']
            all_questions = str(row['all_questions']).split('\n')
            questions = [q.strip() for q in all_questions if q.strip()]
            
            print(f"\n聚类 {cluster_id} ({len(questions)} 个问题):")
            samples = questions[:max_samples]
            for i, q in enumerate(samples, 1):
                print(f"  {i}. {q}")
            if len(questions) > max_samples:
                print(f"  ... 还有 {len(questions) - max_samples} 个问题")
                
    except Exception as e:
        print(f"[Agent] 预览聚类样本时发生错误: {e}")