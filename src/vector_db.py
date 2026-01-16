# vector_db.py
"""
向量数据库构建和检索系统
基于已有聚类结果，使用CLIP模型向量化并存储到Weaviate
"""

import os
import pandas as pd
import numpy as np
import weaviate
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import json
import time
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

class QAVectorDB:
    def __init__(self,
                 weaviate_url: str = "http://localhost:8080",
                 model_name: str = "clip-ViT-B-32-multilingual-v1",
                 device: str = "cpu"):
        """
        初始化QA向量数据库
        
        Args:
            weaviate_url: Weaviate服务地址
            model_name: CLIP模型名称
            device: 计算设备
        """
        self.weaviate_url = weaviate_url
        self.model_name = model_name
        self.device = device
        self.class_name = "QA_Embeddings"
        
        # 强制使用CPU
        print("[VectorDB] 使用CPU进行计算")
        print(f"[VectorDB] 加载CLIP模型: {model_name} 在设备: cpu")
        self.clip_model = SentenceTransformer(model_name, device="cpu")
        self.device = "cpu"
        
        # 初始化Weaviate客户端
        try:
            self.client = weaviate.Client(url=weaviate_url)
            print(f"[VectorDB] 连接Weaviate成功: {weaviate_url}")
        except Exception as e:
            print(f"[VectorDB] 连接Weaviate失败: {e}")
            print("请确保Weaviate服务正在运行")
            raise
    
    def create_schema(self) -> None:
        """创建Weaviate数据库Schema"""
        print(f"[VectorDB] 创建Schema: {self.class_name}")
        
        # 删除已存在的类（如果存在）
        try:
            self.client.schema.delete_class(self.class_name)
            print(f"[VectorDB] 已删除已存在的类: {self.class_name}")
        except:
            pass
        
        # 定义Schema
        schema = {
            "class": self.class_name,
            "description": "QA问答对向量数据库",
            "properties": [
                {
                    "name": "question",
                    "dataType": ["text"],
                    "description": "用户问题"
                },
                {
                    "name": "answer", 
                    "dataType": ["text"],
                    "description": "客服回答"
                },
                {
                    "name": "source_dataset",
                    "dataType": ["string"],
                    "description": "数据来源平台"
                },
                {
                    "name": "cluster_id",
                    "dataType": ["int"],
                    "description": "聚类ID"
                },
                {
                    "name": "cluster_name",
                    "dataType": ["string"],
                    "description": "聚类中文标签"
                }
            ],
            "vectorizer": "none",  # 使用外部向量
        }
        
        self.client.schema.create_class(schema)
        print(f"[VectorDB] Schema创建成功")
    
    def load_cluster_data(self, excel_path: str) -> pd.DataFrame:
        """
        加载已处理的聚类数据
        
        Args:
            excel_path: 聚类结果Excel文件路径
        
        Returns:
            DataFrame: 包含原始QA数据的DataFrame
        """
        print(f"[VectorDB] 加载聚类数据: {excel_path}")
        
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"文件不存在: {excel_path}")
        
        # 读取原始行级数据工作表
        df = pd.read_excel(excel_path, engine="openpyxl", sheet_name="cluster_answer_raw")
        
        # 确保必要的列存在
        required_cols = ['问题_clean', '回答', 'cluster_id', 'source_dataset']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")
        
        print(f"[VectorDB] 加载数据成功: {len(df)} 条记录")
        return df
    
    def vectorize_questions(self, questions: List[str], batch_size: int = 32) -> np.ndarray:
        """
        使用CLIP模型对问题进行向量化
        
        Args:
            questions: 问题列表
            batch_size: 批处理大小
        
        Returns:
            向量数组
        """
        print(f"[VectorDB] 向量化 {len(questions)} 个问题...")
        
        vectors = self.clip_model.encode(
            questions,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # 归一化向量
        )
        
        print(f"[VectorDB] 向量化完成，向量维度: {vectors.shape}")
        return vectors
    
    def import_data(self, df: pd.DataFrame, batch_size: int = 100) -> None:
        """
        批量导入数据到Weaviate
        
        Args:
            df: 包含QA数据的DataFrame
            batch_size: 批处理大小
        """
        print(f"[VectorDB] 开始导入 {len(df)} 条数据到Weaviate...")
        
        # 提取问题并向量化
        questions = df['问题_clean'].tolist()
        vectors = self.vectorize_questions(questions)
        
        # 读取聚类标签映射（从cluster_answer_view工作表）
        cluster_df = pd.read_excel(
            os.path.join(OUTPUT_DIR, "merged_cluster_answers.xlsx"),
            engine="openpyxl", 
            sheet_name="cluster_answer_view"
        )
        cluster_mapping = dict(zip(cluster_df['cluster_id'], cluster_df['cluster_name']))
        
        # 批量导入
        success_count = 0
        
        with self.client.batch as batch:
            batch.batch_size = batch_size
            
            for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="导入数据")):
                try:
                    # 准备数据对象
                    data_object = {
                        "question": str(row['问题_clean']),
                        "answer": str(row['回答']),
                        "source_dataset": str(row['source_dataset']),
                        "cluster_id": int(row['cluster_id']),
                        "cluster_name": cluster_mapping.get(row['cluster_id'], "未知聚类")
                    }
                    
                    # 添加到批次
                    batch.add_data_object(
                        data_object=data_object,
                        class_name=self.class_name,
                        vector=vectors[idx].tolist()
                    )
                    
                    success_count += 1
                    
                except Exception as e:
                    print(f"[VectorDB] 导入第 {idx} 条数据失败: {e}")
        
        print(f"[VectorDB] 数据导入完成: {success_count}/{len(df)} 条成功")
    
    def search(self, 
               query: str, 
               limit: int = 10,
               source_filter: Optional[str] = None,
               cluster_filter: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        向量相似性搜索
        
        Args:
            query: 查询问题
            limit: 返回结果数量
            source_filter: 数据源过滤 ("tianmao" 或 "overseas")
            cluster_filter: 聚类过滤
        
        Returns:
            搜索结果列表
        """
        print(f"[VectorDB] 搜索查询: '{query}'")
        
        # 向量化查询
        query_vector = self.clip_model.encode([query], normalize_embeddings=True)[0]
        
        # 构建查询
        query_builder = (self.client.query
                        .get(self.class_name, ["question", "answer", "source_dataset", "cluster_id", "cluster_name"])
                        .with_near_vector({"vector": query_vector.tolist()})
                        .with_limit(limit)
                        .with_additional(["distance"]))
        
        # 添加过滤条件
        where_conditions = []
        
        if source_filter:
            where_conditions.append({
                "path": ["source_dataset"],
                "operator": "Equal",
                "valueString": source_filter
            })
        
        if cluster_filter is not None:
            where_conditions.append({
                "path": ["cluster_id"],
                "operator": "Equal",
                "valueInt": cluster_filter
            })
        
        # 应用过滤条件
        if len(where_conditions) == 1:
            query_builder = query_builder.with_where(where_conditions[0])
        elif len(where_conditions) > 1:
            query_builder = query_builder.with_where({
                "operator": "And",
                "operands": where_conditions
            })
        
        # 执行查询
        try:
            response = query_builder.do()
            
            if 'errors' in response:
                print(f"[VectorDB] 查询错误: {response['errors']}")
                return []
            
            results = response['data']['Get'][self.class_name]
            print(f"[VectorDB] 找到 {len(results)} 个相似结果")
            
            return results
            
        except Exception as e:
            print(f"[VectorDB] 查询失败: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            # 总数据量
            total_response = (self.client.query
                            .aggregate(self.class_name)
                            .with_meta_count()
                            .do())
            
            total_count = total_response['data']['Aggregate'][self.class_name][0]['meta']['count']
            
            # 按数据源统计
            source_response = (self.client.query
                             .aggregate(self.class_name)
                             .with_group_by_filter(["source_dataset"])
                             .with_meta_count()
                             .do())
            
            stats = {
                "total_records": total_count,
                "by_source": {},
                "status": "connected"
            }
            
            for group in source_response['data']['Aggregate'][self.class_name]:
                if 'groupedBy' in group:
                    source = group['groupedBy']['value']
                    count = group['meta']['count']
                    stats["by_source"][source] = count
            
            return stats
            
        except Exception as e:
            return {"status": "error", "message": str(e)}


def main():
    """主程序入口"""
    print("=" * 80)
    print("QA向量数据库构建系统")
    print("基于CLIP-ViT-B-32-multilingual-v1 + Weaviate")
    print("=" * 80)
    
    # 配置参数
    excel_path = os.path.join(OUTPUT_DIR, "merged_cluster_answers.xlsx")
    weaviate_url = "http://localhost:8080"  # 根据实际情况修改
    
    try:
        # 初始化向量数据库
        vector_db = QAVectorDB(weaviate_url=weaviate_url, device="cuda")
        
        # 创建Schema
        vector_db.create_schema()
        
        # 加载聚类数据
        df = vector_db.load_cluster_data(excel_path)
        
        # 导入数据
        vector_db.import_data(df)
        
        # 显示统计信息
        stats = vector_db.get_database_stats()
        print(f"\n[VectorDB] 数据库统计:")
        print(f"  总记录数: {stats.get('total_records', 0)}")
        for source, count in stats.get('by_source', {}).items():
            print(f"  {source}: {count} 条")
        
        print("\n" + "=" * 80)
        print("向量数据库构建完成！")
        print("现在可以使用search功能进行相似性检索")
        print("=" * 80)
        
        # 演示搜索功能
        demo_search(vector_db)
        
    except Exception as e:
        print(f"[VectorDB] 程序执行失败: {e}")


def demo_search(vector_db: QAVectorDB):
    """演示搜索功能"""
    print("\n" + "=" * 50)
    print("搜索功能演示")
    print("=" * 50)
    
    # 示例查询
    demo_queries = [
        "手机配置怎么样",
        "退货政策",
        "发货时间"
    ]
    
    for query in demo_queries:
        print(f"\n>>> 查询: {query}")
        results = vector_db.search(query, limit=3)
        
        for i, result in enumerate(results, 1):
            distance = result['_additional']['distance']
            similarity = 1 - distance  # 转换为相似度
            print(f"  {i}. 相似度: {similarity:.3f}")
            print(f"     问题: {result['question'][:50]}...")
            print(f"     来源: {result['source_dataset']} | 聚类: {result['cluster_name']}")
            print()


if __name__ == "__main__":
    main()