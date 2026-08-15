"""LangGraph Agent 节点函数

每个节点是一个独立的工作步骤，接收状态并返回更新。
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI

from config import settings
from logger import log_info, log_error
from agents import NPC_ROLES, NPC_NAMES

# ==================== 系统提示词 ====================

PM_SYSTEM_PROMPT = """你是{name}，一名资深产品经理。

【角色设定】
- 性格: {personality}
- 专长: {expertise}
- 说话风格: {style}

【当前任务】
{task_summary}

【输出要求】
请用结构化格式输出，包含以下内容：
1. 需求描述：用户想要什么
2. 功能列表：需要实现哪些功能，每个功能一句话说明
3. 优先级：哪些先做，哪些后做
4. 验收标准：怎么才算完成

【注意】
- 用第一人称"我"回答
- 只做需求分析，不需要写代码或做设计
- 输出格式要清晰，方便下游角色阅读
"""

DESIGNER_SYSTEM_PROMPT = """你是{name}，一名UI/UX设计师。

【角色设定】
- 性格: {personality}
- 专长: {expertise}
- 说话风格: {style}

【当前任务】
{task_summary}

【上游需求分析】
{upstream_context}

【设计规范参考】
{rag_context}

【输出要求】
请参考上面的设计规范，输出设计方案：
1. 页面布局：整体结构、区域划分
2. 色彩方案：主色、辅助色、文字色
3. 组件选择：用什么组件，怎么用
4. 交互说明：用户操作流程、反馈状态

【注意】
- 用第一人称"我"回答
- 只输出设计方案，不需要写代码
- 要引用设计规范中的具体内容（如色值、字号、间距）
- 如果没有相关规范，就按自己的专业判断
"""

DEV_SYSTEM_PROMPT = """你是{name}，一名全栈工程师。

【角色设定】
- 性格: {personality}
- 专长: {expertise}
- 说话风格: {style}

【当前任务】
{task_summary}

【上游需求分析】
{upstream_pm}

【上游设计方案】
{upstream_designer}

【核心要求】
你只需要做一件事：**生成一个完整的 HTML 文件并保存到本地**。

要求：
- 生成一个独立的 HTML 文件（所有样式和脚本都在一个文件里）
- 用 write_file 工具保存到 generated/ 目录下
- 用 read_file 工具验证文件是否保存成功
- 文件名用有意义的英文名，比如 login.html

【工具】
1. write_file(filepath, content) - 写入/创建文件（这是你主要用的工具）
2. read_file(filepath) - 读取文件内容（用来验证）
3. list_files(directory, pattern) - 列出文件

【工作方式】
1. 第一步：生成 HTML 代码，用 write_file 保存
2. 第二步：用 read_file 验证文件已保存
3. 第三步：输出【最终回复】告诉用户文件路径

【工具调用格式】
需要调用工具时，只输出以下 JSON：
{
    "action": "call_tool",
    "tool": "write_file",
    "args": {
        "filepath": "generated/login.html",
        "content": "<完整的HTML代码>"
    }
}

【最终回复格式】
完成所有工作后，以"【最终回复】"开头输出摘要。
"""

# ReAct 工具调用提示词（用于第二轮，带上工具结果）
REACT_CONTINUE_PROMPT = """你刚才调用了工具，以下是工具返回的结果：

【工具结果】
{tool_results}

请根据工具结果继续你的工作。如果还需要调用工具，按 JSON 格式输出。
如果已经足够，输出以"【最终回复】"开头的最终回复。
"""


def _build_messages(system_prompt: str, user_input: str) -> list:
    """构造消息列表，避免 ChatPromptTemplate 的模板变量冲突"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]


def _format_system_prompt(template: str, **kwargs) -> str:
    """安全地格式化系统提示词，避免 { 和 } 冲突"""
    # 先替换已知的模板变量
    for key, value in kwargs.items():
        placeholder = "{" + key + "}"
        template = template.replace(placeholder, str(value))
    return template


def _parse_tool_call(response: str) -> dict:
    """解析 LLM 输出中的工具调用 JSON

    如果 LLM 决定调用工具，输出格式：
    {"action": "call_tool", "tool": "read_file", "args": {"filepath": "..."}}

    如果 LLM 输出最终回复，返回 None
    """
    import json
    import re

    # 找到第一个 { 和最后一个 }，提取整个 JSON 块
    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    json_str = response[start:end+1]

    try:
        data = json.loads(json_str)
        if data.get("action") == "call_tool" and data.get("tool"):
            return {
                "tool": data["tool"],
                "args": data.get("args", {}),
            }
        # 也支持直接 {"tool": "...", "args": {...}} 格式
        if data.get("tool") and data.get("args") is not None:
            return {
                "tool": data["tool"],
                "args": data["args"],
            }
    except (json.JSONDecodeError, KeyError):
        pass

    return None


def _truncate_args(args: dict, max_len: int = 80) -> dict:
    """截断参数字典中的大段文本，用于日志显示"""
    truncated = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > max_len:
            truncated[k] = v[:max_len] + "..."
        else:
            truncated[k] = v
    return truncated


def pm_node(state: Dict[str, Any], llm: ChatOpenAI) -> Dict[str, Any]:
    """需求多(PM) - 需求分析节点"""
    role = NPC_ROLES[NPC_NAMES["pm"]]
    task = state.get("task_analysis", {}).get("task_summary", state.get("user_input", ""))

    system_prompt = _format_system_prompt(
        PM_SYSTEM_PROMPT,
        name=NPC_NAMES["pm"],
        personality=role["personality"],
        expertise=role["expertise"],
        style=role["style"],
        task_summary=task
    )

    log_info(f"  🤖 [{NPC_NAMES['pm']}] 正在分析需求...")
    messages = _build_messages(system_prompt, state.get("user_input", ""))
    response_obj = llm.invoke(messages)
    response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
    log_info(f"  💬 [{NPC_NAMES['pm']}] 需求分析完成")

    agent_history = state.get("agent_history", [])
    agent_history.append({
        "agent": "pm",
        "npc_name": NPC_NAMES["pm"],
        "output": response[:500] + "..." if len(response) > 500 else response
    })

    return {
        "active_agent": "pm",
        "shared_context": {
            **state.get("shared_context", {}),
            "pm_analysis": response,
            "last_agent": "pm"
        },
        "agent_history": agent_history,
        "messages": [{"role": "assistant", "content": f"【{NPC_NAMES['pm']}-产品经理】\n{response}"}]
    }


def designer_node(state: Dict[str, Any], llm: ChatOpenAI) -> Dict[str, Any]:
    """设计多(Designer) - 设计输出节点 (使用RAG)"""
    role = NPC_ROLES[NPC_NAMES["designer"]]
    task = state.get("task_analysis", {}).get("task_summary", state.get("user_input", ""))
    upstream = state.get("shared_context", {}).get("pm_analysis", "")
    rag_context = state.get("rag_context", "")

    system_prompt = _format_system_prompt(
        DESIGNER_SYSTEM_PROMPT,
        name=NPC_NAMES["designer"],
        personality=role["personality"],
        expertise=role["expertise"],
        style=role["style"],
        task_summary=task,
        upstream_context=upstream,
        rag_context=rag_context or "（暂无设计规范参考）"
    )

    log_info(f"  🎨 [{NPC_NAMES['designer']}] 正在设计方案...")
    messages = _build_messages(system_prompt, state.get("user_input", ""))
    response_obj = llm.invoke(messages)
    response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
    log_info(f"  💬 [{NPC_NAMES['designer']}] 设计方案完成")

    agent_history = state.get("agent_history", [])
    agent_history.append({
        "agent": "designer",
        "npc_name": NPC_NAMES["designer"],
        "output": response[:500] + "..." if len(response) > 500 else response
    })

    return {
        "active_agent": "designer",
        "shared_context": {
            **state.get("shared_context", {}),
            "design_output": response,
            "last_agent": "designer"
        },
        "agent_history": agent_history,
        "messages": [{"role": "assistant", "content": f"【{NPC_NAMES['designer']}-UI/UX设计师】\n{response}"}]
    }


def dev_node(state: Dict[str, Any], llm: ChatOpenAI) -> Dict[str, Any]:
    """技术多(Dev) - 技术实现节点 (使用真实MCP工具调用)

    实现 ReAct 模式：
    1. LLM 决定是否调用工具 → 输出 JSON
    2. 解析 JSON → 调用 CodeMCPClient
    3. 工具结果喂回 LLM
    4. LLM 输出最终回复
    """
    from mcp_servers.client import CodeMCPClient

    role = NPC_ROLES[NPC_NAMES["dev"]]
    task = state.get("task_analysis", {}).get("task_summary", state.get("user_input", ""))
    upstream_pm = state.get("shared_context", {}).get("pm_analysis", "")
    upstream_designer = state.get("shared_context", {}).get("design_output", "")

    system_prompt = _format_system_prompt(
        DEV_SYSTEM_PROMPT,
        name=NPC_NAMES["dev"],
        personality=role["personality"],
        expertise=role["expertise"],
        style=role["style"],
        task_summary=task,
        upstream_pm=upstream_pm or "（暂无需求分析输入）",
        upstream_designer=upstream_designer or "（暂无设计方案输入）"
    )

    # ==================== ReAct 循环 ====================
    log_info(f"  🔧 [{NPC_NAMES['dev']}] 正在实现代码...")
    messages = _build_messages(system_prompt, state.get("user_input", ""))
    max_rounds = 3  # 最多调用 3 次工具
    tool_call_log = []

    for round_idx in range(max_rounds):
        response_obj = llm.invoke(messages)
        response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)

        # 检查是否是工具调用
        tool_call = _parse_tool_call(response)
        if tool_call is None:
            # 没有工具调用，这就是最终回复
            final_response = response
            break

        # 执行工具调用
        tool_name = tool_call["tool"]
        tool_args = tool_call["args"]
        log_info(f"  🔧 技术多调用工具: {tool_name}({_truncate_args(tool_args)})")

        try:
            tool_result = CodeMCPClient.call_tool(tool_name, **tool_args)
            log_info(f"  ✅ 工具返回: {tool_result[:100]}...")
        except Exception as e:
            tool_result = f"工具调用失败: {e}"
            log_error(f"  ❌ 工具调用失败: {e}")

        tool_call_log.append({
            "tool": tool_name,
            "args": tool_args,
            "result": tool_result[:200],
        })

        # 把工具结果喂回 LLM，继续下一轮
        continue_prompt = _format_system_prompt(
            REACT_CONTINUE_PROMPT,
            tool_results=tool_result
        )
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": continue_prompt})
    else:
        # 超出最大轮次，直接取最后一次回复
        final_response = response

    # 去掉"【最终回复】"前缀
    if final_response.startswith("【最终回复】"):
        final_response = final_response[6:].strip()

    # ==================== 工具调用结果校验 ====================
    verify_messages = []
    for call in tool_call_log:
        if call["tool"] == "write_file":
            filepath = call["args"].get("filepath", call["args"].get("file_path", ""))
            if not filepath:
                continue

            try:
                content_result = CodeMCPClient.call_tool("read_file", filepath=filepath)
                if isinstance(content_result, str) and (content_result.startswith("错误") or content_result.startswith("失败")):
                    verify_messages.append(f"⚠️ {filepath}: 文件读取失败 — {content_result}")
                    continue

                content = content_result if isinstance(content_result, str) else str(content_result)

                if filepath.endswith(".py"):
                    try:
                        compile(content, filepath, "exec")
                        verify_messages.append(f"✅ {filepath}: Python 语法检查通过")
                    except SyntaxError as e:
                        verify_messages.append(f"❌ {filepath}: Python 语法错误 — {e}")

                elif filepath.endswith(".html"):
                    if "<html" in content.lower() or "<!DOCTYPE" in content:
                        verify_messages.append(f"✅ {filepath}: HTML 基本结构检查通过")
                    else:
                        verify_messages.append(f"⚠️ {filepath}: 文件内容不是有效的 HTML（缺少 <html> 或 <!DOCTYPE>）")

                elif filepath.endswith(".css"):
                    verify_messages.append(f"✅ {filepath}: 文件已保存（{len(content)} 字节）")

                elif filepath.endswith(".js"):
                    try:
                        compile(content, filepath, "exec")
                        verify_messages.append(f"✅ {filepath}: JavaScript 语法检查通过")
                    except SyntaxError as e:
                        verify_messages.append(f"❌ {filepath}: JavaScript 语法错误 — {e}")

                else:
                    verify_messages.append(f"✅ {filepath}: 文件已保存（{len(content)} 字节）")

            except Exception as e:
                verify_messages.append(f"⚠️ {filepath}: 校验失败 — {e}")

    if verify_messages:
        verify_text = "\n".join(verify_messages)
        final_response = f"{final_response}\n\n【代码校验】\n{verify_text}"
        log_info(f"  🔍 代码校验结果:\n{verify_text}")

    log_info(f"  💬 [{NPC_NAMES['dev']}] 代码实现完成")

    # 记录工具调用日志
    mcp_tool_calls = state.get("tool_results", [])
    mcp_tool_calls.extend(tool_call_log)

    agent_history = state.get("agent_history", [])
    agent_history.append({
        "agent": "dev",
        "npc_name": NPC_NAMES["dev"],
        "output": final_response[:100] + "..." if len(final_response) > 100 else final_response
    })

    # 记录工具调用到协作历史
    for call in tool_call_log:
        # 截断 args 中的大段内容，只保留前100字符
        truncated_args = {}
        for k, v in call["args"].items():
            if isinstance(v, str) and len(v) > 100:
                truncated_args[k] = v[:100] + "..."
            else:
                truncated_args[k] = v
        agent_history.append({
            "agent": "dev_tool",
            "npc_name": NPC_NAMES["dev"],
            "output": f"调用了 {call['tool']}({truncated_args}) → {call['result'][:80]}"
        })

    return {
        "active_agent": "dev",
        "shared_context": {
            **state.get("shared_context", {}),
            "dev_plan": final_response,
            "last_agent": "dev",
            "mcp_tool_calls": mcp_tool_calls,
        },
        "agent_history": agent_history,
        "messages": [{"role": "assistant", "content": f"【{NPC_NAMES['dev']}-全栈工程师】\n{final_response}"}]
    }


def chat_node(state: Dict[str, Any], llm: ChatOpenAI) -> Dict[str, Any]:
    """普通聊天节点 - 随机选一个NPC回复"""
    import random
    npc_name = random.choice([NPC_NAMES["pm"], NPC_NAMES["dev"], NPC_NAMES["designer"]])
    role = NPC_ROLES[npc_name]

    system_prompt = f"你是{role['title']}{npc_name}。\n性格:{role['personality']}\n说话风格:{role['style']}\n\n回复要自然友好，像真实同事一样闲聊。"
    messages = _build_messages(system_prompt, state.get("user_input", ""))
    response_obj = llm.invoke(messages)
    response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)

    return {
        "active_agent": "chat",
        "final_response": response,
        "messages": [{"role": "assistant", "content": f"【{npc_name}-{role['title']}】\n{response}"}]
    }


def multi_agent_node(state: Dict[str, Any], llm: ChatOpenAI) -> Dict[str, Any]:
    """多Agent协作节点 - PM → Designer → Dev 流水线

    流程:
    1. 需求多(PM) → 输出需求文档 (Prompt约束)
    2. 设计多(Designer) → 查RAG规范 → 输出设计方案
    3. 技术多(Dev) → 参考需求+设计 → MCP工具 → 输出代码
    """
    # Step 1: PM 需求分析
    pm_result = pm_node(state, llm)
    state.update(pm_result)

    # Step 2: Designer 设计方案 (带RAG)
    designer_result = designer_node(state, llm)
    state.update(designer_result)

    # Step 3: Dev 技术实现 (带MCP)
    dev_result = dev_node(state, llm)
    state.update(dev_result)

    # 合成最终回复
    history = state.get("agent_history", [])
    summary_parts = []
    for h in history:
        npc = h.get("npc_name", "")
        role = h.get("agent", "")
        output = h.get("output", "")
        summary_parts.append(f"【{npc} ({role})】\n{output}")

    summary = "\n\n".join(summary_parts)
    state["final_response"] = f"团队协作完成！以下是各角色的输出：\n\n{summary}"

    return state