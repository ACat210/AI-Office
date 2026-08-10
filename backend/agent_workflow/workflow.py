"""LangGraph 工作流 - 多Agent协作

定义 StateGraph 的节点、边和编译逻辑。
"""

from typing import Dict, Any, Literal, Optional
from langgraph.graph.state import StateGraph
from langgraph.constants import END
from langgraph.checkpoint.memory import InMemorySaver

from langchain_openai import ChatOpenAI

from config import settings
from logger import log_info
from agent_workflow.state import AgentState
from agent_workflow.supervisor import SupervisorAgent
from agent_workflow.nodes import (
    pm_node, dev_node, designer_node,
    chat_node, multi_agent_node
)
from rag.knowledge_base import get_design_knowledge_base


def create_workflow() -> StateGraph:
    """创建并配置多Agent协作工作流

    Returns:
        编译后的 StateGraph 应用
    """
    log_info("🕸️  正在初始化 LangGraph 多Agent工作流...")

    # 初始化LLM
    llm = ChatOpenAI(
        model=settings.LLM_MODEL_ID,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        temperature=settings.LLM_TEMPERATURE
    )
    log_info(f"  ✅ LLM 初始化成功 (模型: {settings.LLM_MODEL_ID})")

    # 初始化Supervisor
    supervisor = SupervisorAgent(llm)
    log_info("  ✅ Supervisor Agent 初始化成功")

    # 初始化设计规范知识库 (RAG for 设计多)
    design_kb = get_design_knowledge_base()
    design_kb.initialize()
    log_info("  ✅ 设计规范知识库初始化成功")

    # ==================== 构建StateGraph ====================

    builder = StateGraph(AgentState)

    # 1. 注册所有节点
    builder.add_node("supervisor", lambda state: _supervisor_node(state, supervisor))
    builder.add_node("pm", lambda state: pm_node(state, llm))
    builder.add_node("dev", lambda state: dev_node(state, llm))
    builder.add_node("designer", lambda state: _designer_with_rag(state, llm, design_kb))
    builder.add_node("chat", lambda state: chat_node(state, llm))
    builder.add_node("multi", lambda state: _multi_agent_with_rag(state, llm, design_kb))

    # 2. 设置入口点
    builder.set_entry_point("supervisor")

    # 3. 定义条件路由
    def router(state: AgentState) -> Literal["pm", "dev", "designer", "chat", "multi", "__end__"]:
        """根据supervisor的决策路由到对应Agent"""
        next_agent = state.get("active_agent", "chat")
        log_info(f"  🔀 路由到: {next_agent}")
        if next_agent == "end":
            return END
        return next_agent

    # 4. 添加条件边 (supervisor → 各Agent)
    builder.add_conditional_edges(
        "supervisor",
        router,
        {
            "pm": "pm",
            "dev": "dev",
            "designer": "designer",
            "chat": "chat",
            "multi": "multi",
            END: END
        }
    )

    # 5. 各Agent执行完后直接结束
    builder.add_edge("pm", END)
    builder.add_edge("dev", END)
    builder.add_edge("designer", END)
    builder.add_edge("chat", END)
    builder.add_edge("multi", END)

    # 6. 编译
    workflow = builder.compile(checkpointer=InMemorySaver())

    log_info("  ✅ LangGraph 工作流编译完成")
    return workflow


def _supervisor_node(state: AgentState, supervisor: SupervisorAgent) -> AgentState:
    """Supervisor节点 - 分析用户输入，决定路由"""
    user_input = state.get("user_input", "")

    # Supervisor 决策
    decision = supervisor.decide(user_input)

    return {
        "active_agent": decision.get("next_agent", "chat"),
        "task_analysis": {
            "reason": decision.get("reason", ""),
            "task_summary": decision.get("task_summary", user_input[:50])
        },
        "messages": [{"role": "system", "content": f"Supervisor 决策: 路由到 {decision['next_agent']}"}]
    }


def _designer_with_rag(state: AgentState, llm: ChatOpenAI, design_kb) -> AgentState:
    """设计多节点 - 自动注入RAG上下文"""
    user_input = state.get("user_input", "")
    task = state.get("task_analysis", {}).get("task_summary", user_input)

    # 查询设计规范知识库
    rag_context = design_kb.format_context(task, top_k=3)
    if rag_context:
        log_info("  📚 RAG 检索到设计规范上下文")

    # 注入RAG上下文到状态
    state["rag_context"] = rag_context

    # 调用设计多节点
    return designer_node(state, llm)


def _multi_agent_with_rag(state: AgentState, llm: ChatOpenAI, design_kb) -> AgentState:
    """多Agent协作节点 - PM→Designer→Dev，带RAG"""
    user_input = state.get("user_input", "")
    task = state.get("task_analysis", {}).get("task_summary", user_input)

    # 提前查询设计规范知识库，供设计师使用
    rag_context = design_kb.format_context(task, top_k=3)
    if rag_context:
        log_info("  📚 RAG 检索到设计规范上下文")
    state["rag_context"] = rag_context

    return multi_agent_node(state, llm)


# 全局单例
_workflow = None


def get_workflow() -> StateGraph:
    """获取工作流实例 (单例)"""
    global _workflow
    if _workflow is None:
        _workflow = create_workflow()
    return _workflow