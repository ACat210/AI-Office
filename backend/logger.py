"""对话日志系统"""

import logging
import os
from datetime import datetime
from pathlib import Path

# 创建logs目录
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# 创建日志文件名 (按日期)
today = datetime.now().strftime("%Y-%m-%d")
LOG_FILE = LOGS_DIR / f"dialogue_{today}.log"

# 配置日志格式
LOG_FORMAT = "%(asctime)s - %(message)s"
DATE_FORMAT = "%H:%M:%S"

# 创建logger
dialogue_logger = logging.getLogger("dialogue")
dialogue_logger.setLevel(logging.INFO)

# 移除已有的handlers (避免重复)
dialogue_logger.handlers.clear()

# 创建文件handler
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

# 创建控制台handler
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

# 添加handlers
dialogue_logger.addHandler(file_handler)
dialogue_logger.addHandler(console_handler)

# 防止日志传播到root logger
dialogue_logger.propagate = False

# 抑制第三方库的HTTP请求日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

def log_dialogue_start(npc_name: str, player_message: str):
    """记录对话开始"""
    dialogue_logger.info("=" * 60)
    dialogue_logger.info(f"💬 对话开始: {npc_name} <-> 玩家")
    dialogue_logger.info("=" * 60)
    dialogue_logger.info(f"📝 玩家消息: {player_message}")

def log_memory_retrieval(npc_name: str, count: int, memories: list = None):
    """记录记忆检索"""
    dialogue_logger.info(f"🧠 检索到{count}条相关记忆")
    if memories:
        dialogue_logger.info("  📚 相关记忆:")
        for i, mem in enumerate(memories[:3], 1):
            content = mem.page_content[:50] + "..." if len(mem.page_content) > 50 else mem.page_content
            dialogue_logger.info(f"    {i}. {content}")

def log_generating_response():
    """记录正在生成回复"""
    dialogue_logger.info("🤖 正在生成回复...")

def log_npc_response(npc_name: str, response: str):
    """记录NPC回复"""
    dialogue_logger.info(f"💬 {npc_name}回复: {response}")

def log_memory_saved(npc_name: str):
    """记录记忆保存"""
    dialogue_logger.info(f"  💾 对话已保存到{npc_name}的记忆中")

def log_dialogue_end():
    """记录对话结束"""
    dialogue_logger.info("=" * 60)
    dialogue_logger.info("✅ 对话完成\n")

def log_info(message: str):
    """记录普通信息"""
    dialogue_logger.info(message)

def log_error(message: str):
    """记录错误信息"""
    dialogue_logger.error(message)

# 启动时记录日志文件位置
try:
    print(f"\n📝 对话日志文件: {LOG_FILE}")
    print(f"📂 日志目录: {LOGS_DIR}\n")
except UnicodeEncodeError:
    print(f"\n[LOG] 对话日志文件: {LOG_FILE}")
    print(f"[LOG] 日志目录: {LOGS_DIR}\n")

