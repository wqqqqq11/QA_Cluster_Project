# start_vectordb.py
"""
QA向量数据库系统一键启动脚本
"""

import os
import sys
import subprocess
import time
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_db import QAVectorDB
from src.utils import get_weaviate_url


def check_weaviate_running(url: str = None, timeout: int = 30) -> bool:
    """检查Weaviate服务是否运行"""
    url = url or get_weaviate_url()
    print(f"🔍 检查Weaviate服务状态 ({url})...")
    
    for i in range(timeout):
        try:
            response = requests.get(f"{url}/v1", timeout=3)
            if response.status_code == 200:
                print("✅ Weaviate服务运行正常")
                return True
        except:
            if i == 0:
                print(f"⏳ 等待Weaviate服务启动...", end="")
            print(".", end="", flush=True)
            time.sleep(1)
    
    print(f"\n❌ Weaviate服务未响应 (超时{timeout}秒)")
    return False


def start_weaviate():
    """启动Weaviate服务"""
    print("🚀 启动Weaviate向量数据库...")
    
    try:
        # 检查docker-compose文件
        if not os.path.exists("docker-compose.yml"):
            print("❌ 未找到docker-compose.yml文件")
            return False
        
        # 启动docker-compose
        result = subprocess.run(
            ["docker-compose", "up", "-d"],
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✅ Docker Compose 启动成功")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker Compose 启动失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ 未找到docker-compose命令，请确保Docker已安装")
        return False


def check_data_file() -> bool:
    """检查数据文件是否存在"""
    data_file = os.path.join("output", "merged_cluster_answers.xlsx")
    
    if os.path.exists(data_file):
        print(f"✅ 找到数据文件: {data_file}")
        return True
    else:
        print(f"❌ 未找到数据文件: {data_file}")
        print("请先运行 'python main.py' 生成聚类数据")
        return False


def check_database_data(vector_db: QAVectorDB) -> bool:
    """检查数据库中是否有数据"""
    try:
        stats = vector_db.get_database_stats()
        
        if stats.get("status") == "connected":
            total_records = stats.get("total_records", 0)
            if total_records > 0:
                print(f"✅ 数据库已有数据: {total_records:,} 条记录")
                return True
            else:
                print("📦 数据库为空，需要导入数据")
                return False
        else:
            print(f"❌ 数据库连接异常: {stats.get('message', '未知错误')}")
            return False
            
    except Exception as e:
        print(f"❌ 检查数据库状态失败: {e}")
        return False


def import_data_if_needed():
    """如果需要的话导入数据"""
    print("\n" + "="*60)
    print("📊 检查和导入数据")
    print("="*60)
    
    # 检查数据文件
    if not check_data_file():
        return False
    
    try:
        # 连接数据库
        vector_db = QAVectorDB()
        
        # 每次都重新构建数据库
        print("🔄 重新构建数据库...")
        
        # 导入数据
        print("📥 开始导入数据...")
        
        # 创建Schema
        vector_db.create_schema()
        
        # 加载和导入数据
        excel_path = os.path.join("output", "merged_cluster_answers.xlsx")
        df = vector_db.load_cluster_data(excel_path)
        vector_db.import_data(df)
        
        # 显示最终统计
        stats = vector_db.get_database_stats()
        if stats.get("status") == "connected":
            print(f"\n✅ 数据导入成功:")
            print(f"   总记录数: {stats.get('total_records', 0):,}")
            for source, count in stats.get('by_source', {}).items():
                print(f"   {source}: {count:,} 条")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据导入失败: {e}")
        return False

def main():
    """主程序"""
    print("="*70)
    print("🚀 QA向量数据库系统启动器")
    print("="*70)
    
    # 步骤1: 启动Weaviate服务
    print("\n步骤1: 启动Weaviate向量数据库")
    print("-" * 40)
    
    if not check_weaviate_running():
        if not start_weaviate():
            print("❌ 无法启动Weaviate服务，程序退出")
            sys.exit(1)
        
        # 等待服务完全启动
        if not check_weaviate_running(timeout=60):
            print("❌ Weaviate服务启动超时，程序退出")
            sys.exit(1)
    
    # 步骤2: 检查和导入数据
    print("\n步骤2: 检查和导入数据")
    print("-" * 40)
    
    if not import_data_if_needed():
        print("❌ 数据准备失败，程序退出")
        sys.exit(1)

if __name__ == "__main__":
    main()