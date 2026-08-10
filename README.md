# 🏢 数字办公室 (AI-Office)

> **AI Agent 开发技术展示项目** — 基于 LangChain + LangGraph + MCP + RAG 的多 Agent 协作系统

三个 AI NPC 在数字办公室里各司其职：**需求多**(产品经理)、**设计多**(设计师)、**技术多**(工程师)，玩家可以和它们聊天、协作完成项目。后端采用 **FastAPI** 异步架构，前端集成 **Godot 4** 2D 游戏引擎。

---

## 📋 技术栈一览

| 类别 | 技术 | 用途 |
|---|---|---|
| **LLM 框架** | LangChain | Prompt 管理、LLM 调用链、记忆系统 |
| **多 Agent 编排** | LangGraph (StateGraph) | 多 Agent 工作流状态机、Supervisor 路由 |
| **工具调用** | MCP 协议 (Model Context Protocol) | 标准化工具定义与调用（代码生成、文件读写） |
| **RAG 检索增强** | FAISS + Embedding | NPC 长期对话记忆向量检索 |
| **知识库 RAG** | 关键词匹配 + 向量检索 | 设计规范知识库检索增强 |
| **流式传输** | SSE (Server-Sent Events) | 实时对话输出 |
| **后端框架** | FastAPI + asyncio | 异步 API 服务 |
| **游戏前端** | Godot 4 + GDScript | 2D 角色扮演交互界面 |

---

## 🎯 项目亮点（Agent 开发技术）

### 1. 多 Agent 协作工作流 (LangGraph)
```
用户输入 → Supervisor(LLM路由决策) → 需求多(需求分析)
                                    → 设计多(设计方案+RAG检索)
                                    → 技术多(代码生成+MCP工具调用)
```
- 基于 **StateGraph** 构建有状态的工作流，每个 Agent 节点独立维护上下文
- **Supervisor 节点** 通过 LLM 动态路由，根据用户意图选择单 Agent 或全流程协作
- 支持 **多轮 ReAct 循环**，技术多可反复调用工具直到任务完成

### 2. 长期记忆系统 (RAG + FAISS)
- 每个 NPC 拥有独立的 **FAISS 向量库**，存储对话历史作为长期记忆
- 对话时自动检索相关历史记忆，实现"回忆"能力
- 两级记忆结构：**FAISS 长期记忆库** + **工作记忆（短期缓存）**
- 支持完整的记忆 CRUD（查看、清空、持久化）

### 3. MCP 工具调用
- 基于 **MCP 协议** 定义标准化工具，技术多可通过 ReAct 循环调用
- 工具列表：`read_file` / `write_file` / `list_files` / `search_code` / `run_command`
- 纯 Python 实现，跨平台兼容（Windows/Linux/Mac）
- 最多 3 轮自动纠错，提高工具调用成功率

### 4. RAG 知识库
- 设计多自动检索本地设计规范文档，输出符合规范的设计方案
- 支持 **向量检索 (FAISS) + 关键词降级搜索** 双通道
- 兼容 DashScope (阿里云通义千问) Embedding 模型

### 5. 流式对话 (SSE)
- 单 NPC 和多 Agent 协作均支持 **SSE 流式输出**
- 实时显示 Agent 思考过程和最终回复

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

---

## 📁 项目结构

```
backend/                         # 后端服务
├── main.py                      # FastAPI 入口 + 路由
├── agents.py                    # NPC Agent 系统 + 记忆管理
├── models.py                    # Pydantic 数据模型
├── config.py                    # 环境变量配置
├── logger.py                    # 对话日志系统
├── test_chat.py                 # CLI 测试工具
├── .env.template                # 环境变量模板
├── requirements.txt             # Python 依赖
│
├── agent_workflow/              # LangGraph 多Agent工作流
│   ├── state.py                 # AgentState 状态定义
│   ├── supervisor.py            # LLM 路由决策
│   ├── nodes.py                 # 各 Agent 节点函数
│   └── workflow.py              # StateGraph 编译
│
├── mcp_servers/                 # MCP 工具服务器
│   ├── code_mcp.py              # 代码工具（read_file/write_file/list_files/search_code/run_command）
│   └── client.py                # MCP 客户端封装
│
├── rag/                         # RAG 检索增强系统
│   └── knowledge_base.py        # 设计规范知识库 (FAISS + 关键词降级)
│
├── knowledge_base/design/       # 设计规范文档
│   ├── 设计原则.md
│   ├── 组件规范.md
│   └── UI模式.md
│
├── generated/                   # 技术多生成的代码文件
└── logs/                        # 对话日志文件

ai-office/                       # Godot 2D 游戏前端
├── scenes/                      # 场景文件
│   └── main.tscn
├── scripts/                     # GDScript 脚本
│   ├── main.gd                  # 主场景逻辑
│   ├── player.gd                # 玩家控制
│   ├── npc.gd                   # NPC 行为
│   ├── dialogue_ui.gd           # 对话 UI
│   ├── api_client.gd            # API 通信
│   └── config.gd                # 全局配置
└── assets/                      # 资源文件
```

---

## 🎮 核心功能

### 1. 单 NPC 对话

每个 NPC 有独立的性格、记忆系统，可以记住之前的对话内容。

```bash
# 通过 API
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"npc_name":"需求多","message":"你好，在忙什么？"}'

# 通过 CLI 测试工具
python test_chat.py --single "你好"
```

### 2. 多 Agent 协作 (LangGraph)

用户输入 → Supervisor 路由 → 需求多(需求分析) → 设计多(设计方案) → 技术多(代码实现)

```bash
# 通过 API
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"帮我设计一个登录页面"}'

# 通过 CLI
python test_chat.py --agent "帮我设计一个登录页面"
```

### 3. MCP 工具调用

技术多通过 ReAct 循环调用真实工具：
- `write_file` — 生成并保存代码文件
- `read_file` — 读取文件验证
- `list_files` — 列出目录文件
- `search_code` — 搜索代码
- `run_command` — 运行命令行

### 4. 流式响应 (SSE)

```bash
# 单 NPC 流式
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"npc_name":"需求多","message":"你好"}'

# 多 Agent 流式
curl -N -X POST http://localhost:8000/agent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"帮我设计一个登录页面"}'
```

### 5. CLI 测试工具

```bash
# 查看 NPC 列表
python test_chat.py --list

# 单 NPC 对话
python test_chat.py --single "你好"

# 多 Agent 协作（默认）
python test_chat.py --agent "帮我设计一个登录页面"

# 流式响应（需要先启动服务器）
python test_chat.py --stream "你好"
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
| POST | `/chat/stream` | 单 NPC 流式对话 |
| POST | `/agent/chat` | 多 Agent 协作 |
| POST | `/agent/chat/stream` | 多 Agent 流式协作 |
| GET | `/agent/history/{user_id}` | Agent 协作历史 |

---

## 🗺️ 架构图

```
用户输入
    │
    ▼
┌─────────────┐
│  Supervisor  │  ←  LLM 分析意图，决定路由
│   (路由决策)  │
└─────┬───────┘
      │
      ├─── "chat" ────→ 随机 NPC 闲聊
      │
      ├─── "pm" ──────→ 需求多（需求分析）
      │
      ├─── "designer" ─→ 设计多（RAG 检索设计规范 → 设计方案）
      │
      ├─── "dev" ──────→ 技术多（MCP 工具调用 → 生成代码）
      │
      └─── "multi" ────→ 全流程：需求多 → 设计多 → 技术多
```

---

## 🔧 常见问题

### Embedding API 报 400 错误

如果使用 DashScope (阿里云通义千问) 作为 LLM 服务，langchain-openai 的 tokenization 不兼容。项目已通过自定义 `DashScopeEmbeddings` 适配器解决。

### 修改 NPC 名字

编辑 `agents.py` 中的 `NPC_NAMES` 字典即可：

```python
NPC_NAMES = {
    "pm": "需求多",       # 改成你想要的
    "dev": "技术多",
    "designer": "设计多",
}
```

### 添加设计规范文档

在 `knowledge_base/design/` 目录下添加 `.md` 文件，重启服务后自动加载。

---

## 📝 License

MIT