# 🏢 数字办公室 (AI-Office)

> **AI Agent 开发技术展示项目** — 基于 LangChain + LangGraph + MCP + RAG 的多 Agent 协作系统

![项目界面](ai-office/interface.jpg)

三个 AI NPC 在数字办公室里各司其职：**需求多**(产品经理)、**设计多**(设计师)、**技术多**(工程师)，玩家可以走进办公室和他们聊天、协作完成项目。后端采用 **FastAPI** 异步架构提供 LLM Agent 服务，前端集成 **Godot 4** 2D 游戏引擎实现沉浸式交互体验。

---

## 📋 技术栈

| 类别 | 技术 | 用途 |
|---|---|---|
| **LLM 框架** | LangChain | Prompt 管理、LLM 调用链、记忆系统 |
| **多 Agent 编排** | LangGraph (StateGraph) | 多 Agent 工作流状态机、Supervisor 路由 |
| **可观测性** | LangSmith | LLM 调用追踪、Token 统计、工作流可视化 |
| **工具调用** | MCP 协议 (Model Context Protocol) | 标准化工具定义与调用（代码生成、文件读写） |
| **向量检索** | FAISS + Embedding | NPC 长期对话记忆语义检索 |
| **知识库 RAG** | Qdrant in-memory + 关键词降级 | 设计规范知识库检索增强（PDF 支持） |
| **流式传输** | SSE (Server-Sent Events) | 实时对话输出 |
| **后端框架** | FastAPI + asyncio | 异步 API 服务 |
| **游戏前端** | Godot 4 + GDScript | 2D 角色扮演交互界面 |
| **评估框架** | RAGAS | RAG 检索与生成质量评估 |

---

## 🎯 项目亮点（Agent 开发技术展示）

### 1. 多 Agent 协作工作流 (LangGraph)

```
用户输入 → Supervisor(LLM路由决策) → 需求多(需求分析)
                                    → 设计多(设计方案+RAG检索)
                                    → 技术多(代码生成+MCP工具调用)
```

- 基于 **LangGraph StateGraph** 构建有状态的多 Agent 工作流，每个 Agent 节点独立维护上下文
- **Supervisor 节点** 通过 LLM 动态分析用户意图，自动路由到单 Agent 或全流程协作
- 技术支持 **多轮 ReAct 循环**，可反复调用工具直到任务完成，最多 3 轮自动纠错
- 完整的工作流状态管理：`AgentState` 定义状态结构，`router` 函数控制节点跳转

### 2. NPC 长期记忆系统 (RAG + FAISS)

- 每个 NPC 拥有独立的 **FAISS 向量库**，对话历史以 Document 形式存储并语义检索
- 对话时自动检索相关历史记忆，实现"回忆"能力
- **两级记忆结构**：FAISS 长期记忆库（持久化到磁盘）+ 工作记忆（短期缓存，最近 10 条）
- **自动记忆压缩**：超过阈值后用 LLM 总结旧记忆，降级时保留高重要性内容
- 支持完整的记忆 CRUD：查看、检索、清空、持久化
- 使用 `index_to_docstore_id` 遍历向量库，不依赖相似度搜索即可查看全部记忆

### 3. MCP 工具调用

- 基于 **Model Context Protocol** 封装标准化工具服务（FastMCP Server）
- 技术多 Agent 通过 ReAct 循环调用 5 个代码工具：

| 工具 | 功能 | 安全限制 |
|---|---|---|
| `read_file` | 读取文件内容 | 路径白名单，限制 backend/ 目录 |
| `write_file` | 写入/创建文件 | 路径白名单，自动创建目录 |
| `search_code` | 搜索代码关键词 | 纯 Python 实现（glob+正则），跨平台兼容 |
| `list_files` | 列出目录文件 | 路径白名单 |
| `run_command` | 运行命令 | 命令白名单（python/pytest/flake8/black 等） |

- 双向安全控制：路径白名单 + 命令白名单
- **自动校验**：生成代码后做语法检查（Python `compile()`、HTML 结构校验、JS 语法检查）

### 4. RAG 设计知识库

- 设计多 Agent 自动检索本地设计规范文档，输出符合规范的方案
- 支持 **向量检索 (Qdrant in-memory) + 关键词降级搜索** 双通道
- 兼容 DashScope（阿里云通义千问）Embedding 模型，自动分批处理避免 API 限制
- 支持 **PDF 和 Markdown** 两种文档格式，自动清理和切片
- 设计规范文档位于 `knowledge_base/design/`，支持热加载
- 内置 **RAGAS 评估脚本**，量化检索质量（faithfulness、answer_relevancy、context_precision、context_recall）

### 5. 流式对话 (SSE)

- 单 NPC 和多 Agent 协作均支持 **SSE 流式输出**
- 实时显示 Agent 思考过程和最终回复
- 前端逐字渲染，提升交互体验

### 6. 自动数据目录初始化

- 启动时自动创建 `generated/`（生成代码）、`memory_data/`（记忆持久化）、`logs/`（对话日志）目录，无需手动创建

### 7. 可观测性 (LangSmith)

- 集成 **LangSmith** 链路追踪，自动记录每次 LLM 调用、Token 消耗、延迟
- **LangGraph 工作流**自动可视化：每个 Agent 节点的输入/输出/工具调用分步展示
- 零代码侵入：设置环境变量即自动启用，无需修改任何调用代码

---

## 🚀 快速开始

### 前提条件
- Python 3.10+
- 一个兼容 OpenAI API 的 LLM 服务（OpenAI、DashScope、DeepSeek 等）

### 安装

```bash
# 1. 进入后端目录
cd backend

# 2. 创建虚拟环境
python -m venv .venv
# conda install -n proj-name python=3.10
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.template .env
# 编辑 .env 填入你的 API Key 和模型配置
```

### 配置 `.env`

```ini
# 必填: 你的 API Key 和模型
LLM_API_KEY=sk-your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_ID=gpt-4o-mini

# 可选: Embedding 模型（默认复用 LLM 配置）
# EMBEDDING_MODEL=text-embedding-v3
# EMBEDDING_API_KEY=sk-your-embedding-key
```

### 启动服务

```bash
# 启动 API 服务
python -m uvicorn main:app --reload

# 访问 API 文档
# http://localhost:8000/docs
```

### 启动 Godot 前端

用 Godot 4.5+ 打开 `ai-office/` 目录，运行场景即可。

---

## 📁 项目结构

```
📦 AI-Office
├── README.md                          # 项目文档
├── evaluation_results.json            # RAGAS 评估结果
├── evaluation_raw_results.json        # RAGAS 原始评估数据
│
├── backend/                           # 后端服务 (FastAPI)
│   ├── main.py                        # FastAPI 入口 + 路由
│   ├── agents.py                      # NPC Agent 系统 + 记忆管理
│   ├── models.py                      # Pydantic 数据模型
│   ├── config.py                      # 环境变量配置
│   ├── logger.py                      # 对话日志系统
│   ├── evaluate_rag.py                # RAGAS 质量评估脚本
│   ├── .env.template                  # 环境变量模板
│   ├── requirements.txt               # Python 依赖
│   │
│   ├── agent_workflow/                # LangGraph 多Agent工作流
│   │   ├── state.py                   # AgentState 状态定义
│   │   ├── supervisor.py              # LLM 路由决策
│   │   ├── nodes.py                   # 各 Agent 节点函数
│   │   └── workflow.py                # StateGraph 编译与条件边
│   │
│   ├── mcp_servers/                   # MCP 工具服务器
│   │   ├── code_mcp.py                # 5个代码工具 (FastMCP Server)
│   │   └── client.py                  # MCP 客户端封装
│   │
│   ├── knowledge_base/                # RAG 知识库
│   │   ├── rag.py                     # 设计规范知识库 (Qdrant + 关键词降级)
│   │   └── design/                    # 设计规范文档 (PDF)
│   │       ├── introduce.pdf
│   │       ├── philosophy.pdf
│   │       ├── style-guideline.pdf
│   │       └── values.pdf
│   │
│   ├── scripts/                       # 辅助工具
│   │   ├── test_chat.py               # CLI 测试工具 (单NPC/多Agent/流式)
│   │   ├── view_logs.py               # 对话日志查看 (tail -f 风格)
│   │   └── visualize_chunks.py        # 知识库分块可视化 HTML 报告
│   │
│   ├── generated/                     # 技术多生成的代码文件 (自动创建)
│   ├── memory_data/                   # NPC 记忆向量库 (自动创建)
│   └── logs/                          # 对话日志文件 (自动创建)
│
├── ai-office/                         # Godot 4 游戏前端
│   ├── scenes/                        # 场景文件
│   │   ├── main.tscn                  # 主场景
│   │   ├── player.tscn                # 玩家场景
│   │   ├── npc.tscn                   # NPC 场景
│   │   └── dialogue_ui.tscn           # 对话 UI 场景
│   ├── scripts/                       # GDScript 脚本
│   │   ├── main.gd                    # 主场景逻辑
│   │   ├── player.gd                  # 玩家控制 (WASD移动 + E交互)
│   │   ├── npc.gd                     # NPC 行为 (巡逻、交互、动画)
│   │   ├── dialogue_ui.gd             # 对话 UI (SSE 流式输出)
│   │   ├── api_client.gd              # HTTP API 通信
│   │   └── config.gd                  # 全局配置
│   ├── assets/                        # 资源文件
│   │   ├── interiors/Office.png       # 办公室场景背景
│   │   ├── characters/                # 角色精灵
│   │   ├── ui/                        # UI 素材
│   │   └── audio/                     # 音效 (BGM、交互音、脚步声)
│   ├── interface.jpg                  # 项目界面截图
│   └── project.godot                  # Godot 项目配置
```

---

## 🎮 核心功能

### 1. 单 NPC 对话

每个 NPC 有独立的性格、Prompt 和记忆系统，可以记住之前的对话内容。

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"npc_name":"需求多","message":"你好，在忙什么？"}'
```

### 2. 多 Agent 协作

用户输入 → Supervisor 路由 → 需求多(需求分析) → 设计多(设计方案+RAG检索) → 技术多(代码实现+MCP工具)

```bash
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"帮我设计一个登录页面"}'
```

### 3. CLI 测试工具

```bash
python scripts/test_chat.py --list                # 查看 NPC 列表
python scripts/test_chat.py --single "你好"       # 单 NPC 对话
python scripts/test_chat.py --agent "设计登录页"  # 多 Agent 协作
python scripts/test_chat.py --stream "你好"       # 流式响应（需先启动服务器）
```

### 4. RAG 评估

```bash
cd backend
python evaluate_rag.py
# 输出: 评估结果保存到 evaluation_results.json
```

---

## 📡 API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | API 信息 |
| GET | `/health` | 健康检查 |
| GET | `/npcs` | NPC 列表 |
| GET | `/npcs/{name}` | NPC 详情 |
| GET | `/npcs/{name}/memories` | NPC 记忆列表 |
| DELETE | `/npcs/{name}/memories` | 清空 NPC 记忆 |
| POST | `/chat` | 单 NPC 对话 |
| POST | `/chat/stream` | 单 NPC 流式对话 (SSE) |
| POST | `/agent/chat` | 多 Agent 协作 |
| POST | `/agent/chat/stream` | 多 Agent 流式协作 (SSE) |
| GET | `/agent/history/{user_id}` | Agent 协作历史 |

---

## 🗺️ 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Godot 4 前端                              │
│  玩家移动(WASD) → NPC 交互(E键) → 对话UI → HTTP API 请求        │
│                              │                                   │
│                     SSE 流式输出 (逐字渲染)                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI 后端                                 │
│                                                                 │
│  ┌──────────┐  ┌────────────────────────────────────────────┐   │
│  │ 单NPC对话  │  │         LangGraph 多 Agent 工作流          │   │
│  │          │  │                                              │   │
│  │ NPC记忆   │  │  Supervisor(LLM路由)                        │   │
│  │ 检索+保存  │  │      │                                      │   │
│  │          │  │      ├── "pm" ────→ 需求多(需求分析)          │   │
│  │ FAISS    │  │      ├── "designer" → 设计多(RAG检索设计规范)  │   │
│  │ 向量库    │  │      ├── "dev" ───→ 技术多(MCP工具调用)      │   │
│  └──────────┘  │      └── "multi" ──→ 全流程协作 (PM→Des→Dev)  │   │
│                └────────────────────────────────────────────┘   │
│                       │          │                              │
│                       ▼          ▼                              │
│                ┌──────────┐  ┌──────────┐                       │
│                │ RAG 知识库│  │MCP工具服务│                       │
│                │ (设计规范) │  │(代码生成) │                       │
│                │ Qdrant   │  │ 5个工具   │                       │
│                │ +关键词   │  │ +自动校验 │                       │
│                └──────────┘  └──────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 常见问题

### Embedding API 报 400 错误

如果使用 DashScope（阿里云通义千问）作为 LLM 服务，langchain-openai 的 tokenization 不兼容。项目已通过自定义 `DashScopeEmbeddings` 适配器解决，自动分批发送请求。

### 修改 NPC 名字

编辑 `agents.py` 中的 `NPC_NAMES` 字典即可：

```python
NPC_NAMES = {
    "pm": "需求多",
    "dev": "技术多",
    "designer": "设计多",
}
```

### 添加设计规范文档

在 `knowledge_base/design/` 目录下添加 `.md` 或 `.pdf` 文件，重启服务后自动加载。

### 查看对话日志

```bash
cd backend
python scripts/view_logs.py        # 实时查看 (tail -f)
python scripts/view_logs.py view   # 查看完整日志
python scripts/view_logs.py list   # 列出所有日志文件
```

### 查看知识库切片

```bash
cd backend
python scripts/visualize_chunks.py
# 生成 chunks_visual.html，浏览器打开查看
```

### 启用 LangSmith 追踪

```bash
# 1. 注册 https://smith.langchain.com 获取 API Key
# 2. 在 .env 中添加:
LANGSMITH_API_KEY=lsv2_pt_xxxxx
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=ai-office
# 3. 重启服务，所有 LLM 调用和 Agent 工作流自动追踪
```

---

## 📝 License

MIT