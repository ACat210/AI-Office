# API客户端 - 与FastAPI后端通信
extends Node

# ==================== 信号 ====================
# 普通对话
signal chat_response_received(npc_name: String, message: String)
signal chat_error(error_message: String)

# 流式对话 (SSE)
signal chat_stream_chunk(npc_name: String, text: String)
signal chat_stream_finished(npc_name: String, full_text: String)
signal chat_stream_error(npc_name: String, error_msg: String)

# 多Agent协作
signal agent_chat_response_received(message: String)
signal agent_chat_error(error_message: String)

# 状态查询
signal npc_status_received(dialogues: Dictionary)
signal npc_list_received(npcs: Array)

# ==================== HTTP请求节点 ====================
var http_chat: HTTPRequest
var http_status: HTTPRequest
var http_npcs: HTTPRequest
var http_agent: HTTPRequest

# ==================== 流式客户端状态 ====================
var _stream_client: HTTPClient = null
var _stream_buffer: String = ""
var _stream_npc_name: String = ""
var _stream_message: String = ""
var _stream_full_text: String = ""
var _stream_active: bool = false
var _stream_request_sent: bool = false
var _stream_body: PackedByteArray = PackedByteArray()

func _ready():
	# 创建HTTP请求节点
	http_chat = HTTPRequest.new()
	http_status = HTTPRequest.new()
	http_npcs = HTTPRequest.new()
	http_agent = HTTPRequest.new()

	add_child(http_chat)
	add_child(http_status)
	add_child(http_npcs)
	add_child(http_agent)

	# 连接信号
	http_chat.request_completed.connect(_on_chat_request_completed)
	http_status.request_completed.connect(_on_status_request_completed)
	http_npcs.request_completed.connect(_on_npcs_request_completed)
	http_agent.request_completed.connect(_on_agent_request_completed)

	print("[INFO] API客户端初始化完成")

# ==================== 普通对话API ====================
func send_chat(npc_name: String, message: String) -> void:
	"""发送对话请求（非流式，备用）"""
	var data = {
		"npc_name": npc_name,
		"message": message
	}
	var json_string = JSON.stringify(data)
	var headers = ["Content-Type: application/json"]

	print("[API] POST /chat -> ", npc_name)
	var error = http_chat.request(
		Config.API_CHAT,
		headers,
		HTTPClient.METHOD_POST,
		json_string
	)
	if error != OK:
		print("[ERROR] 发送对话请求失败: ", error)
		chat_error.emit("网络请求失败")

func _on_chat_request_completed(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	"""处理对话响应"""
	if response_code != 200:
		print("[ERROR] 对话请求失败: HTTP ", response_code)
		chat_error.emit("服务器错误: " + str(response_code))
		return

	var json = JSON.new()
	var parse_result = json.parse(body.get_string_from_utf8())
	if parse_result != OK:
		print("[ERROR] 解析响应失败")
		chat_error.emit("响应解析失败")
		return

	var response = json.data
	if response.has("success") and response["success"]:
		var npc_name = response["npc_name"]
		var msg = response["message"]
		print("[INFO] 收到NPC回复: ", npc_name)
		chat_response_received.emit(npc_name, msg)
	else:
		chat_error.emit("对话失败")

# ==================== 流式对话API (SSE) ====================
func send_chat_stream(npc_name: String, message: String) -> void:
	"""发送流式对话请求（SSE，逐字返回）"""
	if _stream_active:
		print("[WARN] 已有流式请求进行中，跳过")
		return

	_stream_npc_name = npc_name
	_stream_message = message
	_stream_full_text = ""
	_stream_buffer = ""
	_stream_body = PackedByteArray()
	_stream_active = true

	# 创建HTTP客户端
	_stream_client = HTTPClient.new()
	var err = _stream_client.connect_to_host(Config.API_HOST, Config.API_PORT)
	if err != OK:
		chat_stream_error.emit(npc_name, "连接服务器失败")
		_stream_active = false
		_stream_client = null
		return

	print("[API] SSE /chat/stream -> ", npc_name)
	set_process(true)

func _stop_stream() -> void:
	"""停止流式连接"""
	_stream_active = false
	_stream_request_sent = false
	set_process(false)
	if _stream_client:
		_stream_client.close()
		_stream_client = null

func _parse_sse_chunk(chunk: String) -> void:
	"""解析SSE数据块"""
	_stream_buffer += chunk
	# SSE事件以 \n\n 分隔
	while "\n\n" in _stream_buffer:
		var idx = _stream_buffer.find("\n\n")
		var event_str = _stream_buffer.substr(0, idx)
		_stream_buffer = _stream_buffer.substr(idx + 2)

		# 解析事件行
		for line in event_str.split("\n"):
			line = line.strip_edges()
			if line.begins_with("data: "):
				var json_str = line.substr(6).strip_edges()
				var json = JSON.parse_string(json_str)
				if json is Dictionary:
					var type_str = json.get("type", "")
					match type_str:
						"start":
							pass  # 流开始，无需处理
						"chunk":
							var text = json.get("text", "")
							_stream_full_text += text
							chat_stream_chunk.emit(_stream_npc_name, text)
						"done":
							chat_stream_finished.emit(_stream_npc_name, _stream_full_text)
							_stop_stream()
						"error":
							chat_stream_error.emit(_stream_npc_name, json.get("text", "未知错误"))
							_stop_stream()

func _process(_delta: float) -> void:
	"""轮询HTTPClient，读取流式数据"""
	if not _stream_active or _stream_client == null:
		set_process(false)
		return

	_stream_client.poll()
	var status = _stream_client.get_status()

	match status:
		HTTPClient.STATUS_CONNECTING, HTTPClient.STATUS_RESOLVING:
			# 还在连接中，等待
			pass

		HTTPClient.STATUS_CONNECTED:
			# 连接成功，发送请求（只发送一次）
			if not _stream_request_sent:
				_stream_request_sent = true
				var headers = [
					"Content-Type: application/json",
					"Accept: text/event-stream",
					"Cache-Control: no-cache"
				]
				var body = JSON.stringify({
					"npc_name": _stream_npc_name,
					"message": _stream_message
				})
				_stream_client.request(
					HTTPClient.METHOD_POST,
					"/chat/stream",
					headers,
					body
				)

		HTTPClient.STATUS_REQUESTING:
			# 请求已发送，等待响应
			pass

		HTTPClient.STATUS_BODY:
			# 正在接收响应体
			var chunk = _stream_client.read_response_body_chunk()
			if chunk.size() > 0:
				_stream_body.append_array(chunk)
				var text = chunk.get_string_from_utf8()
				if text.length() > 0:
					_parse_sse_chunk(text)

		HTTPClient.STATUS_DISCONNECTED:
			# 连接断开
			if _stream_active:
				# 如果还有未完成的流，输出已完成的部分
				if _stream_full_text.length() > 0:
					chat_stream_finished.emit(_stream_npc_name, _stream_full_text)
				else:
					chat_stream_error.emit(_stream_npc_name, "连接意外断开")
			_stop_stream()

		_:
			# 其他状态（错误等）
			if _stream_active:
				chat_stream_error.emit(_stream_npc_name, "连接状态异常: " + str(status))
			_stop_stream()

# ==================== 多Agent协作API ====================
func send_agent_chat(message: String) -> void:
	"""发送多Agent协作请求"""
	var data = {
		"message": message,
		"user_id": "player"
	}
	var json_string = JSON.stringify(data)
	var headers = ["Content-Type: application/json"]

	print("[API] POST /agent/chat")
	var error = http_agent.request(
		Config.API_AGENT_CHAT,
		headers,
		HTTPClient.METHOD_POST,
		json_string
	)
	if error != OK:
		print("[ERROR] 发送协作请求失败: ", error)
		agent_chat_error.emit("网络请求失败")

func _on_agent_request_completed(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	"""处理多Agent协作响应"""
	if response_code != 200:
		print("[ERROR] 协作请求失败: HTTP ", response_code)
		agent_chat_error.emit("服务器错误: " + str(response_code))
		return

	var json = JSON.new()
	var parse_result = json.parse(body.get_string_from_utf8())
	if parse_result != OK:
		print("[ERROR] 解析协作响应失败")
		agent_chat_error.emit("响应解析失败")
		return

	var response = json.data
	if response.has("success") and response["success"]:
		var msg = response.get("message", "")
		print("[INFO] 收到协作回复: ", msg.length(), "字符")
		agent_chat_response_received.emit(msg)
	else:
		agent_chat_error.emit("协作处理失败")

# ==================== NPC状态API ====================
func get_npc_status() -> void:
	"""获取NPC状态"""
	if http_status.get_http_client_status() != HTTPClient.STATUS_DISCONNECTED:
		print("[WARN] NPC状态请求正在处理中,跳过本次请求")
		return

	print("[API] GET /npcs/status")
	var error = http_status.request(Config.API_NPC_STATUS)
	if error != OK:
		print("[ERROR] 获取NPC状态失败: ", error)

func _on_status_request_completed(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	"""处理NPC状态响应"""
	if response_code != 200:
		print("[ERROR] NPC状态请求失败: HTTP ", response_code)
		return

	var json = JSON.new()
	var parse_result = json.parse(body.get_string_from_utf8())
	if parse_result != OK:
		print("[ERROR] 解析NPC状态失败")
		return

	var response = json.data
	if response.has("dialogues"):
		var dialogues = response["dialogues"]
		print("[INFO] 收到NPC状态更新: ", dialogues.size(), "个NPC")
		npc_status_received.emit(dialogues)

# ==================== NPC列表API ====================
func get_npc_list() -> void:
	"""获取NPC列表"""
	print("[API] GET /npcs")
	var error = http_npcs.request(Config.API_NPCS)
	if error != OK:
		print("[ERROR] 获取NPC列表失败: ", error)

func _on_npcs_request_completed(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	"""处理NPC列表响应"""
	if response_code != 200:
		print("[ERROR] NPC列表请求失败: HTTP ", response_code)
		return

	var json = JSON.new()
	var parse_result = json.parse(body.get_string_from_utf8())
	if parse_result != OK:
		print("[ERROR] 解析NPC列表失败")
		return

	var response = json.data
	if response.has("npcs"):
		var npcs = response["npcs"]
		print("[INFO] 收到NPC列表: ", npcs.size(), "个NPC")
		npc_list_received.emit(npcs)