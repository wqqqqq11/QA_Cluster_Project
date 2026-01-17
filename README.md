# QA聚类与向量搜索系统

一个基于机器学习的问答(QA)聚类分析和向量相似性搜索系统，使用CLIP和Sentence Transformers进行文本向量化，Weaviate作为向量数据库，提供RESTful API接口。

## 项目特性

- 🔍 **智能聚类**：使用嵌入模型对问答数据进行聚类分析
- 🗄️ **向量数据库**：基于Weaviate的向量存储和检索
- 🚀 **RESTful API**：提供高效的相似性搜索接口
- 🐳 **Docker支持**：容器化部署，快速启动
- ⚙️ **配置管理**：统一配置文件管理所有参数

## 技术栈

- **后端**: FastAPI + Uvicorn
- **向量数据库**: Weaviate
- **机器学习**: Sentence Transformers + CLIP
- **数据处理**: Pandas + NumPy + Scikit-learn
- **容器化**: Docker + Docker Compose

## 项目结构

```
QA_Cluster_Project/
├── api/                          # FastAPI后端接口
│   ├── api_main.py              # 主应用入口
│   └── router/
│       └── search_api.py        # 搜索API路由
├── src/                         # 核心代码
│   ├── vector_db.py             # 向量数据库操作类
│   ├── utils.py                 # 工具函数和配置管理
│   └── agent.py                 # AI代理相关功能
├── scripts/                     # 执行脚本
│   ├── main.py                  # 主数据处理脚本
│   ├── start_vectordb.py        # 向量数据库启动脚本
│   └── search_interface.py      # 交互式搜索界面
├── config/                      # 配置文件
│   └── config.json              # 系统配置
├── data/                        # 原始数据
│   ├── meaningful_answer_overseas.xlsx    # 海外客服数据
│   └── meaningful_answer_tianmao.xlsx     # 天猫客服数据
├── output/                      # 输出结果
│   ├── merged_cluster_answers.xlsx        # 聚类结果
│   ├── merged_cluster_summary.xlsx        # 聚类摘要
│   └── search_results/                    # 搜索结果缓存
├── vectorized_data/             # 向量化数据
│   ├── merged_question_vectors.npy        # 合并问题向量
│   ├── overseas_question_vectors.npy      # 海外问题向量
│   └── tianmao_question_vectors.npy       # 天猫问题向量
├── Dockerfile                   # Docker镜像构建文件
├── docker-compose.yml           # Docker Compose配置
└── requirements.txt             # Python依赖
```

## 主要文件说明

### 核心文件

- **`config/config.json`** - 系统配置文件
  ```json
  {
    "weaviate_url": "http://localhost:8080",
    "embedding_model_name": "paraphrase-multilingual-MiniLM-L12-v2",
    "clip_model_name": "clip-ViT-B-32-multilingual-v1"
  }
  ```

- **`src/utils.py`** - 工具函数和配置管理
  - 数据加载和处理函数
  - 向量化和聚类算法
  - 配置文件读取函数

- **`src/vector_db.py`** - 向量数据库操作
  - QAVectorDB类：封装Weaviate操作
  - 支持数据导入、Schema创建、相似性搜索

### API接口

- **`api/api_main.py`** - FastAPI主应用
  - 健康检查接口: `GET /health`
  - 搜索接口集成

- **`api/router/search_api.py`** - 搜索API
  - 相似性搜索: `POST /search`
  - 支持按数据源和聚类过滤

### 执行脚本

- **`scripts/main.py`** - 主数据处理流程
  - 数据加载和预处理
  - 特征提取和向量化
  - 聚类分析和结果保存

- **`scripts/start_vectordb.py`** - 向量数据库管理
  - 检查Weaviate服务状态
  - 数据导入和索引构建

- **`scripts/search_interface.py`** - 交互式搜索界面
  - 命令行搜索工具
  - 实时相似性查询

## 快速开始

### 方式一：Docker Compose（推荐）

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd QA_Cluster_Project
   ```

2. **启动服务**
   ```bash
   docker-compose up -d
   ```

3. **等待服务启动完成**
   - Weaviate: http://localhost:8080
   - API服务: http://localhost:8000

4. **验证服务状态**
   ```bash
   curl http://localhost:8000/health
   ```

### 方式二：本地开发

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **启动Weaviate**
   ```bash
   docker run -p 8080:8080 -v weaviate_data:/var/lib/weaviate semitechnologies/weaviate:1.22.4
   ```

3. **运行数据处理**
   ```bash
   python scripts/main.py
   ```

4. **启动向量数据库构建**
   ```bash
   python scripts/start_vectordb.py
   ```

5. **启动API服务**
   ```bash
   uvicorn api.api_main:app --host 0.0.0.0 --port 8000
   ```

## API使用说明

### 健康检查
```bash
GET /health
```

### 相似性搜索
```bash
POST /search
Content-Type: application/json

{
  "query": "手机防水吗",
  "top_k": 5,
  "source_filter": "tianmao"
}
```

响应示例：
```json
{
  "query": "手机防水吗",
  "total_results": 3,
  "results": [
    {
      "similarity_score": 0.95,
      "source_dataset": "tianmao",
      "question": "手机防水等级是多少",
      "answer": "支持IP68级防水",
      "image_url": ""
    }
  ]
}
```

## 数据流程

1. **数据预处理**: 加载Excel数据，提取问题文本
2. **特征提取**: 使用Sentence Transformers进行向量化
3. **聚类分析**: MiniBatch K-means聚类算法
4. **向量存储**: CLIP模型重新向量化，存储到Weaviate
5. **相似性搜索**: 向量相似性检索，返回相关QA对

## 配置说明

所有配置通过 `config/config.json` 管理：

- `weaviate_url`: Weaviate服务地址
- `embedding_model_name`: 嵌入模型（用于聚类）
- `clip_model_name`: CLIP模型（用于向量搜索）

## 开发说明

### 添加新模型
1. 在 `config.json` 中更新模型名称
2. 重启相关服务
3. 重新运行数据处理流程

### 自定义数据源
1. 准备Excel格式数据（问题-回答列）
2. 放置在 `data/` 目录
3. 修改 `scripts/main.py` 中的数据加载逻辑

## 故障排除

### Weaviate连接失败
- 确认Weaviate服务正在运行
- 检查 `config.json` 中的URL配置
- 查看端口占用情况

### 模型加载失败
- 确认网络连接正常
- 检查Hugging Face缓存目录
- 验证模型名称拼写

### API请求失败
- 检查API服务是否启动
- 验证请求格式和参数
- 查看应用日志

## 许可证

[请添加许可证信息]

## 贡献

欢迎提交Issue和Pull Request！