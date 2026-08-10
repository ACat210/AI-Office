"""数据模型定义"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class ChatRequest(BaseModel):
    """单个NPC对话请求"""
    npc_name: str = Field(..., description="NPC名称")
    message: str = Field(..., description="玩家消息")
    
    class Config:
        json_schema_extra = {
            "example": {
                "npc_name": "需求多",
                "message": "你好,你在做什么?"
            }
        }

class ChatResponse(BaseModel):
    """单个NPC对话响应"""
    npc_name: str = Field(..., description="NPC名称")
    npc_title: str = Field(..., description="NPC职位")
    message: str = Field(..., description="NPC回复")
    success: bool = Field(default=True, description="是否成功")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="时间戳")
    
    class Config:
        json_schema_extra = {
            "example": {
                "npc_name": "张三",
                "npc_title": "Python工程师",
                "message": "你好!我正在写代码,调试一个多智能体系统的bug。",
                "success": True
            }
        }

class NPCInfo(BaseModel):
    """NPC信息"""
    name: str = Field(..., description="NPC名称")
    title: str = Field(..., description="NPC职位")
    location: str = Field(..., description="NPC位置")
    activity: str = Field(..., description="当前活动")
    available: bool = Field(default=True, description="是否可对话")

class NPCListResponse(BaseModel):
    """NPC列表响应"""
    npcs: List[NPCInfo] = Field(..., description="NPC列表")
    total: int = Field(..., description="NPC总数")


# ==================== LangGraph 多Agent模型 ====================

class AgentChatRequest(BaseModel):
    """多Agent协作对话请求"""
    message: str = Field(..., description="用户消息")
    user_id: str = Field(default="player", description="用户ID")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "帮我设计一个用户登录功能",
                "user_id": "player"
            }
        }


class AgentChatResponse(BaseModel):
    """多Agent协作对话响应"""
    response: str = Field(..., description="最终回复内容")
    active_agent: str = Field(default="", description="最后处理的Agent")
    agent_history: List[Dict[str, str]] = Field(default=[], description="Agent协作历史")
    success: bool = Field(default=True, description="是否成功")

