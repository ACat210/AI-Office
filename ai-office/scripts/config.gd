# 数字办公室 - 全局配置
extends Node

# ==================== API配置 ====================
const API_BASE_URL = "http://localhost:8000"
const API_HOST = "localhost"
const API_PORT = 8000
const API_USE_SSL = false
const API_CHAT = API_BASE_URL + "/chat"
const API_CHAT_STREAM = API_BASE_URL + "/chat/stream"
const API_AGENT_CHAT = API_BASE_URL + "/agent/chat"
const API_AGENT_CHAT_STREAM = API_BASE_URL + "/agent/chat/stream"
const API_NPCS = API_BASE_URL + "/npcs"
# const API_NPC_STATUS = API_BASE_URL + "/npcs/status"  # REMOVED - endpoint doesn't exist

# ==================== NPC配置 ====================
# NPC名字在这里统一配置，方便修改
const NPC_NAMES = ["需求多", "技术多", "设计多"]
const NPC_TITLES = {
	"需求多": "产品经理",
	"技术多": "全栈工程师",
	"设计多": "UI/UX设计师"
}

# ==================== 游戏配置 ====================
const PLAYER_SPEED = 200.0  # 玩家移动速度
const INTERACTION_DISTANCE = 80.0  # 交互距离
const NPC_STATUS_UPDATE_INTERVAL = 30.0  # NPC状态更新间隔(秒)

# ==================== UI配置 ====================
const DIALOGUE_FADE_TIME = 0.3  # 对话框淡入淡出时间
const NPC_LABEL_OFFSET = Vector2(0, -60)  # NPC名字标签偏移

# ==================== 调试配置 ====================
const DEBUG_MODE = true  # 调试模式
const SHOW_INTERACTION_RANGE = true  # 显示交互范围

# ==================== 工具函数 ====================
func log_info(message: String) -> void:
	if DEBUG_MODE:
		print("[INFO] ", message)

func log_error(message: String) -> void:
	print("[ERROR] ", message)

func log_api(endpoint: String, data: Dictionary) -> void:
	if DEBUG_MODE:
		print("[API] ", endpoint, " -> ", JSON.stringify(data))
