"""配置文件 - LangChain版本"""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# 修复系统代理导致 LLM API 连接失败的问题
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")


class Settings:
    """应用配置"""

    # API配置
    API_TITLE = "数字办公室 API"
    API_VERSION = "1.0.0"
    API_HOST = os.getenv("API_HOST")
    API_PORT = int(os.getenv("API_PORT"))

    # NPC配置
    NPC_UPDATE_INTERVAL = int(os.getenv("NPC_UPDATE_INTERVAL", "30"))  # NPC状态更新间隔(秒)

    # LLM配置 (兼容 OpenAI / ModelScope 等 OpenAI 兼容 API)
    LLM_TYPE: str = os.getenv("LLM_TYPE", "openai")  # openai / modelscope
    LLM_MODEL_ID: str = os.getenv("LLM_MODEL_ID", "gpt-4o-mini")
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")
    LLM_BASE_URL: Optional[str] = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    # Embedding 配置 (用于向量记忆检索)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    EMBEDDING_BASE_URL: Optional[str] = os.getenv("EMBEDDING_BASE_URL")
    EMBEDDING_API_KEY: Optional[str] = os.getenv("EMBEDDING_API_KEY")

    # CORS配置
    CORS_ORIGINS = ["*"]

    @classmethod
    def validate(cls):
        """验证配置"""
        if not cls.LLM_API_KEY:
            print("⚠️  警告: 未设置LLM_API_KEY环境变量")
            print("   请在.env文件中配置LLM_API_KEY")
            return False

        print(f"✅ LLM配置:")
        print(f"   类型: {cls.LLM_TYPE}")
        print(f"   模型: {cls.LLM_MODEL_ID}")
        print(f"   服务地址: {cls.LLM_BASE_URL}")
        return True


settings = Settings()