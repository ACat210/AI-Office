"""Agent协作状态定义"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """多Agent协作的全局状态

    每个Agent节点可以读取和修改这个状态，
    通过 LangGraph 的 StateGraph 机制传递。
    """
    # 对话历史 (LangGraph 内置的 Message 管理)
    messages: Annotated[List[Dict[str, str]], add_messages]

    # 用户原始输入
    user_input: str

    # 当前活跃的Agent名称 (supervisor 决定路由)
    active_agent: str

    # 任务分解结果
    task_analysis: Optional[Dict[str, Any]]

    # 共享上下文 (Agent间传递的数据)
    shared_context: Dict[str, Any]

    # MCP工具调用结果
    tool_results: List[Dict[str, Any]]

    # RAG检索结果
    rag_context: Optional[str]

    # 最终回复
    final_response: Optional[str]

    # 协作历史 (记录每个Agent做了什么)
    agent_history: List[Dict[str, str]]