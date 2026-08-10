"""Supervisor Agent - 任务路由决策

分析用户输入，决定由哪个NPC处理，或是否需要多Agent协作。
"""

import json
import re
from langchain_openai import ChatOpenAI

from config import settings
from logger import log_info, log_error
from agents import NPC_NAMES

# 可路由的Agent列表
AGENTS = ["pm", "dev", "designer", "multi", "chat"]


def _build_supervisor_prompt() -> str:
    """构建包含当前NPC名字的提示词"""
    prompt = """你是一个智能任务分配器，负责分析用户请求并分配给最合适的专家。

【团队角色】
- {pm_name} (pm): 产品经理 - 擅长需求分析、产品规划、项目管理、拆解任务
- {dev_name} (dev): 全栈工程师 - 擅长技术实现、代码编写、系统设计、架构方案
- {designer_name} (designer): UI/UX设计师 - 擅长界面设计、交互设计、视觉呈现

【路由规则】
根据用户请求的内容，判断应该分配给谁：

1. "pm" - 以下情况分配给产品经理：
   - 需求分析、功能规划、产品方向
   - 需要拆解复杂任务
   - 需要写文档/PRD
   - 涉及业务流程、用户需求

2. "dev" - 以下情况分配给工程师：
   - 技术问题、代码实现
   - 架构设计、技术方案
   - Bug修复、代码审查
   - 需要写代码/技术文档

3. "designer" - 以下情况分配给设计师：
   - UI设计、界面美化
   - 交互设计、用户体验
   - 视觉呈现、设计规范
   - 色彩、布局、图标

4. "multi" - 以下情况需要多Agent协作：
   - 需要多个角色配合的复杂任务
   - 需求涉及产品+技术+设计的全流程
   - 用户明确要求"团队协作"

5. "chat" - 普通聊天/问候/闲聊：
   - 打招呼、问候
   - 不涉及具体任务的闲聊
   - 简单问题不需要专业分工

【输出格式】(只输出JSON，不要其他内容)
{
    "next_agent": "pm|dev|designer|multi|chat",
    "reason": "选择原因(10字以内)",
    "task_summary": "任务简要描述"
}
"""
    prompt = prompt.replace("{pm_name}", NPC_NAMES["pm"])
    prompt = prompt.replace("{dev_name}", NPC_NAMES["dev"])
    prompt = prompt.replace("{designer_name}", NPC_NAMES["designer"])
    return prompt


SUPERVISOR_PROMPT = _build_supervisor_prompt()


class SupervisorAgent:
    """Supervisor Agent - 路由决策"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def decide(self, user_input: str) -> dict:
        """分析用户输入，决定路由目标"""
        try:
            messages = [
                {"role": "system", "content": SUPERVISOR_PROMPT},
                {"role": "user", "content": f"用户请求: {user_input}"}
            ]
            response_obj = self.llm.invoke(messages)
            response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)

            result = self._parse_response(response)

            # 验证路由目标是否合法
            if result.get("next_agent") not in AGENTS:
                result["next_agent"] = "chat"

            log_info(f"  Supervisor 路由: {result['next_agent']} (原因: {result.get('reason', '')})")
            return result

        except Exception as e:
            log_error(f"  Supervisor 决策失败: {e}")
            return {
                "next_agent": "chat",
                "reason": "决策失败，默认聊天",
                "task_summary": user_input[:50]
            }

    def _parse_response(self, response: str) -> dict:
        """解析JSON响应"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {
                "next_agent": "chat",
                "reason": "解析失败",
                "task_summary": ""
            }