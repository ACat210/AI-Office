"""数字办公室 FastAPI 后端主程序 - LangChain版本"""

import sys
import os
import json

# 修复 Windows 终端 emoji 显示问题
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# 修复系统代理导致 LLM API 连接失败的问题
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import uvicorn

from config import settings
from models import (
    ChatRequest, ChatResponse,
    NPCListResponse, NPCInfo,
    AgentChatRequest, AgentChatResponse
)
from agents import get_npc_manager, NPC_ROLES
from agent_workflow.workflow import get_workflow
from logger import (
    log_error, log_info, log_npc_response,
    log_memory_saved, log_dialogue_end
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    print("\n" + "=" * 60)
    print("🎮 数字办公室后端服务启动中...")
    print("=" * 60)

    # 初始化可观测性 (LangSmith)
    _init_observability()

    # 验证配置
    settings.validate()

    # 确保数据目录存在
    _ensure_directories()

    # 初始化NPC管理器
    npc_manager = get_npc_manager()

    print("\n✅ 所有服务已启动!")
    print(f"📡 API地址: http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"📚 API文档: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print("=" * 60 + "\n")

    yield

    # 关闭时
    print("\n🛑 正在关闭服务...")
    print("✅ 服务已关闭\n")


def _ensure_directories():
    """确保必要的数据目录存在"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dirs = [
        "generated",      # 技术多生成的代码文件
        "memory_data",    # NPC 记忆向量库持久化
        "logs",           # 对话日志
    ]
    for name in dirs:
        path = os.path.join(base_dir, name)
        os.makedirs(path, exist_ok=True)
        print(f"  📁 已确认目录: {name}")


def _init_observability():
    """初始化 LangSmith 可观测性

    LangSmith 通过环境变量自动检测，设置后所有 LangChain/LangGraph
    调用会自动追踪，无需修改任何代码调用。
    代理配置已在 config.py 中自动处理。
    """
    if not settings.LANGSMITH_API_KEY:
        return

    os.environ.setdefault("LANGSMITH_TRACING", "true" if settings.LANGSMITH_TRACING else "false")
    os.environ.setdefault("LANGSMITH_API_KEY", settings.LANGSMITH_API_KEY)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.LANGSMITH_PROJECT)
    os.environ.setdefault("LANGSMITH_ENDPOINT", settings.LANGSMITH_ENDPOINT)

    if settings.LANGSMITH_TRACING:
        status = "已启用"
        if settings.LANGSMITH_PROXY:
            status += f" (代理: {settings.LANGSMITH_PROXY})"
        else:
            status += " (未配置代理，国内网络可能需要代理才能连接)"
        print(f"  📊 LangSmith 可观测性: {status} (项目: {settings.LANGSMITH_PROJECT})")

def get_npc_mgr():
    """获取NPC管理器实例"""
    global npc_manager
    if npc_manager is None:
        npc_manager = get_npc_manager()
    return npc_manager


# ==================== API路由 ====================

@app.get("/")
async def root():
    """根路径 - API信息"""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "features": ["AI对话", "NPC记忆系统", "多Agent协作(LangGraph)"],
        "framework": "LangChain + LangGraph",
        "endpoints": {
            "docs": "/docs",
            "chat": "/chat",
            "npcs": "/npcs",
            "npc_memories": "/npcs/{npc_name}/memories",
            "agent_chat": "/agent/chat",
            "agent_history": "/agent/history/{user_id}"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/chat", response_model=ChatResponse)
async def chat_with_npc(request: ChatRequest):
    """与NPC对话接口"""
    npc_mgr = get_npc_mgr()

    npc_info = npc_mgr.get_npc_info(request.npc_name)
    if not npc_info:
        raise HTTPException(
            status_code=404,
            detail=f"NPC '{request.npc_name}' 不存在"
        )

    try:
        response_text = npc_mgr.chat(request.npc_name, request.message)

        return ChatResponse(
            npc_name=request.npc_name,
            npc_title=npc_info["title"],
            message=response_text,
            success=True
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"对话处理失败: {str(e)}"
        )


@app.get("/npcs", response_model=NPCListResponse)
async def list_npcs():
    """获取所有NPC列表"""
    npc_mgr = get_npc_mgr()

    npcs_data = npc_mgr.get_all_npcs()
    npcs = [NPCInfo(**npc) for npc in npcs_data]

    return NPCListResponse(
        npcs=npcs,
        total=len(npcs)
    )


@app.get("/npcs/{npc_name}")
async def get_npc_info(npc_name: str):
    """获取指定NPC的详细信息"""
    npc_mgr = get_npc_mgr()

    npc_info = npc_mgr.get_npc_info(npc_name)
    if not npc_info:
        raise HTTPException(
            status_code=404,
            detail=f"NPC '{npc_name}' 不存在"
        )

    return npc_info


@app.get("/npcs/{npc_name}/memories")
async def get_npc_memories(npc_name: str, limit: int = 10):
    """获取NPC的记忆列表"""
    npc_mgr = get_npc_mgr()

    npc_info = npc_mgr.get_npc_info(npc_name)
    if not npc_info:
        raise HTTPException(
            status_code=404,
            detail=f"NPC '{npc_name}' 不存在"
        )

    try:
        memories = npc_mgr.get_npc_memories(npc_name, limit=limit)

        return {
            "npc_name": npc_name,
            "memories": memories,
            "total": len(memories)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取记忆失败: {str(e)}"
        )


@app.delete("/npcs/{npc_name}/memories")
async def clear_npc_memories(npc_name: str, memory_type: str = None):
    """清空NPC的记忆 (用于测试)"""
    npc_mgr = get_npc_mgr()

    npc_info = npc_mgr.get_npc_info(npc_name)
    if not npc_info:
        raise HTTPException(
            status_code=404,
            detail=f"NPC '{npc_name}' 不存在"
        )

    try:
        npc_mgr.clear_npc_memory(npc_name, memory_type)

        return {
            "message": f"已清空{npc_name}的记忆",
            "npc_name": npc_name,
            "memory_type": memory_type or "all"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"清空记忆失败: {str(e)}"
        )


# ==================== LangGraph 多Agent协作路由 ====================

@app.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest):
    """多Agent协作对话 - 智能路由到PM/Dev/Designer"""
    try:
        workflow = get_workflow()

        initial_state = {
            "messages": [],
            "user_input": request.message,
            "active_agent": "",
            "task_analysis": None,
            "shared_context": {},
            "tool_results": [],
            "rag_context": None,
            "final_response": None,
            "agent_history": [],
        }

        config = {"configurable": {"thread_id": f"agent_{request.user_id}"}}
        result = workflow.invoke(initial_state, config=config)

        final_response = result.get("final_response") or ""
        agent_history = result.get("agent_history", [])
        active_agent = result.get("active_agent", "")

        # 保存到历史记录
        _agent_histories.setdefault(request.user_id, []).append({
            "user_input": request.message,
            "response": final_response,
            "active_agent": active_agent,
            "agent_history": agent_history,
        })
        # 只保留最近 50 条
        _agent_histories[request.user_id] = _agent_histories[request.user_id][-50:]

        # 记录日志到终端
        log_info(f"\n{'='*60}")
        log_info(f"  🤝 多Agent协作完成")
        log_info(f"  💬 用户: {request.message}")
        log_info(f"{'='*60}")
        for h in agent_history:
            agent = h.get("agent", "?")
            npc = h.get("npc_name", "?")
            output = h.get("output", "")
            if agent == "dev_tool":
                log_info(f"  🔧 [{npc}] 工具调用: {output[:200]}")
            else:
                log_info(f"  [{agent}] {npc}:")
                for line in output.split("\n")[:5]:
                    log_info(f"    {line}")
                if len(output.split("\n")) > 5:
                    line_count = len(output.split("\n"))
                    log_info(f"    ... (共{line_count}行)")
        log_info(f"\n  📝 最终回复: {final_response[:200]}")
        log_info("=" * 60)

        if not final_response:
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, dict):
                    final_response = last_msg.get("content", "")

        return AgentChatResponse(
            response=final_response,
            active_agent=active_agent,
            agent_history=agent_history,
            success=True
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Agent协作失败: {str(e)}"
        )


# ==================== 流式响应 ====================

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式对话 - SSE 逐字返回NPC回复"""
    npc_mgr = get_npc_mgr()

    npc_info = npc_mgr.get_npc_info(request.npc_name)
    if not npc_info:
        raise HTTPException(status_code=404, detail=f"NPC '{request.npc_name}' 不存在")

    role = NPC_ROLES.get(request.npc_name)
    if not role:
        raise HTTPException(status_code=404, detail=f"NPC '{request.npc_name}' 不存在")

    system_prompt = f"""你是{role['title']}{request.npc_name}。
性格: {role['personality']}
专长: {role['expertise']}
说话风格: {role['style']}
回复要自然友好，像真实同事一样对话。"""

    llm = ChatOpenAI(
        model=settings.LLM_MODEL_ID,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
        streaming=True,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.message},
    ]

    async def generate():
        yield f"data: {{\"npc\": \"{request.npc_name}\", \"type\": \"start\"}}\n\n"
        full_text = ""
        try:
            for chunk in llm.stream(messages):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if content:
                    full_text += content
                    # SSE 格式: data: {json}\n\n
                    yield f"data: {json.dumps({'type': 'chunk', 'text': content})}\n\n"
        finally:
            # 流结束后保存记忆和日志（不管是否正常结束）
            if full_text:
                try:
                    # 保存到对话记忆
                    memory_system = npc_mgr.memories.get(request.npc_name)
                    if memory_system and hasattr(memory_system, 'add_interaction'):
                        memory_system.add_interaction(
                            request.message, full_text,
                            {"source": "stream", "npc_name": request.npc_name}
                        )
                except Exception as e:
                    log_error(f"流式对话保存记忆失败: {e}")

                # 记录日志到终端和文件
                log_npc_response(request.npc_name, full_text)
                log_memory_saved(request.npc_name)
                log_dialogue_end()

        yield f"data: {json.dumps({'type': 'done', 'text': full_text})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/agent/chat/stream")
async def agent_chat_stream(request: AgentChatRequest):
    """流式多Agent协作 - SSE 逐字返回"""
    workflow = get_workflow()

    initial_state = {
        "messages": [],
        "user_input": request.message,
        "active_agent": "",
        "task_analysis": None,
        "shared_context": {},
        "tool_results": [],
        "rag_context": None,
        "final_response": None,
        "agent_history": [],
    }

    config = {"configurable": {"thread_id": f"agent_{request.user_id}"}}

    async def generate():
        try:
            # 使用 workflow.stream() 逐节点输出
            for step in workflow.stream(initial_state, config=config):
                node_name = list(step.keys())[0] if isinstance(step, dict) else "?"
                node_data = step.get(node_name, {}) if isinstance(step, dict) else step

                # 检查是否是最终节点（有消息输出）
                if isinstance(node_data, dict):
                    agent = node_data.get("active_agent", node_name)
                    yield f"data: {json.dumps({'type': 'node', 'agent': agent})}\n\n"

                    # 流式输出消息内容
                    messages = node_data.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if isinstance(last_msg, dict):
                            content = last_msg.get("content", "")
                            if content:
                                # 逐行发送
                                for line in content.split("\n"):
                                    if line:
                                        yield f"data: {json.dumps({'type': 'line', 'text': line})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/agent/history/{user_id}")
async def get_agent_history(user_id: str = "player"):
    """获取Agent协作历史"""
    history = _agent_histories.get(user_id, [])
    return {
        "user_id": user_id,
        "total": len(history),
        "history": [
            {
                "user_input": h["user_input"],
                "response": h["response"][:200] + "..." if len(h["response"]) > 200 else h["response"],
                "active_agent": h["active_agent"],
                "agent_count": len(h["agent_history"]),
            }
            for h in reversed(history[-20:])  # 最近20条，最新的在前
        ]
    }


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    print("\n🚀 启动数字办公室后端服务...")
    print(f"📍 监听地址: {settings.API_HOST}:{settings.API_PORT}")
    print(f"📖 访问文档: http://localhost:{settings.API_PORT}/docs\n")

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )