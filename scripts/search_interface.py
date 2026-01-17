# search_interface.py
"""
QA向量数据库交互式搜索界面
"""

import sys
import json
import os
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_db import QAVectorDB
from src.utils import get_weaviate_url


class SearchInterface:
    def __init__(self, weaviate_url: str = None):
        """初始化搜索界面"""
        weaviate_url = weaviate_url or get_weaviate_url()
        try:
            self.vector_db = QAVectorDB(weaviate_url=weaviate_url, device="cpu")
            print("✅ 向量数据库连接成功!")
        except Exception as e:
            print(f"❌ 向量数据库连接失败: {e}")
            sys.exit(1)
    
    def show_welcome(self):
        """显示欢迎信息"""
        print("=" * 70)
        print("🔍 QA向量数据库 - 智能搜索系统")
        print("=" * 70)

        # 显示设备信息
        print(f"🖥️  计算设备: CPU")

        # 显示数据库统计
        stats = self.vector_db.get_database_stats()
        if stats.get("status") == "connected":
            print(f"📊 数据库状态: 已连接")
            print(f"📦 总记录数: {stats.get('total_records', 0):,}")
            for source, count in stats.get('by_source', {}).items():
                print(f"   └─ {source}: {count:,} 条")
        else:
            print(f"⚠️  数据库状态: {stats.get('message', '未知错误')}")
        
        print("\n💡 搜索提示:")
        print("  - 直接输入问题进行搜索")
        print("  - 使用 'tianmao:问题' 搜索天猫数据")
        print("  - 使用 'overseas:问题' 搜索海外数据")
        print("  - 输入 'help' 查看帮助")
        print("  - 输入 'quit' 退出程序")
        print("=" * 70)
    
    def show_help(self):
        """显示帮助信息"""
        print("\n📖 搜索帮助:")
        print("  基础搜索:")
        print("    退货政策                    # 在所有数据中搜索")
        print("    手机配置怎么样               # 普通问题搜索")
        print("")
        print("  数据源过滤:")
        print("    tianmao:退货政策            # 只在天猫数据中搜索")
        print("    overseas:shipping policy    # 只在海外数据中搜索")
        print("")
        print("  特殊命令:")
        print("    stats                      # 显示数据库统计信息")
        print("    clusters                   # 显示聚类信息")
        print("    help                       # 显示此帮助")
        print("    quit                       # 退出程序")
        print()
    
    def show_stats(self):
        """显示统计信息"""
        print("\n📊 数据库统计信息:")
        stats = self.vector_db.get_database_stats()
        
        if stats.get("status") == "connected":
            print(f"  总记录数: {stats.get('total_records', 0):,}")
            print("  数据源分布:")
            for source, count in stats.get('by_source', {}).items():
                percentage = (count / stats.get('total_records', 1)) * 100
                print(f"    {source:12}: {count:,} 条 ({percentage:.1f}%)")
        else:
            print(f"  错误: {stats.get('message', '无法获取统计信息')}")
        print()
    
    def show_clusters(self):
        """显示聚类信息"""
        print("\n🏷️  聚类标签信息:")
        try:
            # 获取所有聚类的统计信息（这里简化实现）
            results = self.vector_db.search("", limit=1000)  # 获取大量数据进行聚合
            
            cluster_stats = {}
            for result in results:
                cluster_name = result.get('cluster_name', '未知')
                cluster_id = result.get('cluster_id', -1)
                source = result.get('source_dataset', '未知')
                
                key = (cluster_id, cluster_name)
                if key not in cluster_stats:
                    cluster_stats[key] = {'tianmao': 0, 'overseas': 0}
                
                cluster_stats[key][source] = cluster_stats[key].get(source, 0) + 1
            
            # 显示聚类统计
            for (cluster_id, cluster_name), sources in sorted(cluster_stats.items()):
                total = sum(sources.values())
                print(f"  聚类 {cluster_id:2d}: {cluster_name:15} (总计: {total:3d})")
                
        except Exception as e:
            print(f"  错误: 无法获取聚类信息 - {e}")
        print()
    
    def parse_query(self, user_input: str) -> tuple:
        """
        解析用户输入
        
        Returns:
            (query, source_filter)
        """
        user_input = user_input.strip()
        
        # 检查是否有数据源前缀
        if user_input.startswith("tianmao:"):
            return user_input[8:].strip(), "tianmao"
        elif user_input.startswith("overseas:"):
            return user_input[9:].strip(), "overseas"
        else:
            return user_input, None
    
    def format_results(self, results: list, query: str) -> None:
        """格式化显示搜索结果"""
        if not results:
            print("😔 未找到相关结果，请尝试其他关键词")
            return
        
        print(f"\n🔍 搜索结果 (共找到 {len(results)} 个相关问题):")
        print("-" * 70)
        
        for i, result in enumerate(results, 1):
            # 计算相似度
            distance = result['_additional']['distance']
            similarity = (1 - distance) * 100
            
            # 格式化显示
            print(f"\n{i:2d}. 相似度: {similarity:.1f}% | {result['source_dataset']:8} | {result['cluster_name']:12}")
            
            # 显示问题（高亮关键词）
            question = result['question']
            print(f"    ❓ {question}")
            
            # 显示答案（截断显示）
            answer = result['answer']
            if len(answer) > 150:
                answer = answer[:150] + "..."
            print(f"    ✅ {answer}")
        
        print("-" * 70)

    def generate_search_json(self, query: str, source_filter: Optional[str], results: list) -> str:
        """
        生成搜索结果JSON文件

        Args:
            query: 搜索查询
            source_filter: 数据源过滤器
            results: 搜索结果列表

        Returns:
            生成的JSON文件路径
        """
        # 准备JSON数据
        json_data = {
            "search_info": {
                "query": query,
                "source_filter": source_filter or "all",
                "timestamp": datetime.now().isoformat(),
                "total_results": len(results)
            },
            "categories": {}
        }

        # 按类别组织结果
        for result in results:
            cluster_name = result.get('cluster_name', '未知类别')

            if cluster_name not in json_data["categories"]:
                json_data["categories"][cluster_name] = []

            # 计算相似度
            distance = result['_additional']['distance']
            similarity = (1 - distance) * 100

            # 添加结果
            json_data["categories"][cluster_name].append({
                "similarity_score": round(similarity, 2),
                "source_dataset": result.get('source_dataset', 'unknown'),
                "question": result.get('question', ''),
                "answer": result.get('answer', '')
            })

        # 限制每个类别最多5个结果
        for category in json_data["categories"]:
            json_data["categories"][category] = json_data["categories"][category][:5]

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_clean = query.replace(" ", "_").replace(":", "_")[:30]  # 清理查询字符串
        filename = f"search_result_{timestamp}_{query_clean}.json"

        # 创建output/search_results目录
        output_dir = os.path.join("output", "search_results")
        os.makedirs(output_dir, exist_ok=True)

        # 保存JSON文件
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        return filepath

    def run(self):
        """运行交互式搜索界面"""
        self.show_welcome()
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n🔍 请输入搜索内容 (或输入 help): ").strip()
                
                if not user_input:
                    continue
                
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 感谢使用QA搜索系统!")
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                elif user_input.lower() == 'clusters':
                    self.show_clusters()
                    continue
                
                # 解析查询
                query, source_filter = self.parse_query(user_input)
                
                if not query:
                    print("❌ 请输入有效的搜索内容")
                    continue
                
                # 执行搜索
                print(f"🔄 正在搜索: '{query}'", end="")
                if source_filter:
                    print(f" (限定数据源: {source_filter})")
                else:
                    print(" (全部数据源)")
                
                results = self.vector_db.search(
                    query=query,
                    limit=5,
                    source_filter=source_filter
                )
                
                # 显示结果
                self.format_results(results, query)

                # 生成JSON文件
                if results:
                    json_filepath = self.generate_search_json(query, source_filter, results)
                    print(f"📄 JSON结果已保存: {json_filepath}")
                
            except KeyboardInterrupt:
                print("\n👋 感谢使用QA搜索系统!")
                break
            except Exception as e:
                print(f"❌ 搜索过程中出现错误: {e}")


def main():
    """主程序入口"""
    search_interface = SearchInterface()
    search_interface.run()


if __name__ == "__main__":
    main()