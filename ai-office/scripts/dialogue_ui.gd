# 对话UI脚本 - 支持SSE流式输出和多Agent协作
extends CanvasLayer

# 节点引用
@onready var panel: Panel = $Panel
@onready var npc_name_label: Label = $Panel/NPCName
@onready var npc_title_label: Label = $Panel/NPCTitle
@onready var dialogue_text: RichTextLabel = $Panel/DialogueText
@onready var player_input: LineEdit = $Panel/PlayerInput
@onready var send_button: Button = $Panel/SendButton
@onready var close_button: Button = $Panel/CloseButton

# 当前对话的NPC
var current_npc_name: String = ""

# API客户端引用
var api_client: Node = null

# 流式输出状态
var _is_streaming: bool = false
var _stream_buffer: String = ""

# 加载动画
var _loading_timer: Timer = null
var _loading_label: Label = null
var _loading_dots: int = 0

# 协作模式
var _collab_mode: bool = false
var _collab_button: Button = null

func _ready():
	# 添加到对话系统组
	add_to_group("dialogue_system")

	# 初始隐藏
	visible = false

	# 连接按钮信号
	send_button.pressed.connect(_on_send_button_pressed)
	close_button.pressed.connect(_on_close_button_pressed)
	player_input.text_submitted.connect(_on_text_submitted)

	# 创建加载动画标签
	_loading_label = Label.new()
	_loading_label.name = "LoadingLabel"
	_loading_label.text = ""
	_loading_label.add_theme_color_override("font_color", Color(0.5, 0.5, 0.5))
	_loading_label.add_theme_font_size_override("font_size", 14)
	_loading_label.visible = false
	$Panel.add_child(_loading_label)
	_loading_label.position = Vector2(20, 170)

	# 创建加载动画计时器
	_loading_timer = Timer.new()
	_loading_timer.name = "LoadingTimer"
	_loading_timer.wait_time = 0.5
	_loading_timer.one_shot = false
	_loading_timer.timeout.connect(_on_loading_timer_timeout)
	add_child(_loading_timer)

	# 创建协作模式切换按钮
	_collab_button = Button.new()
	_collab_button.name = "CollabButton"
	_collab_button.text = "🤝 协作"
	_collab_button.toggle_mode = true
	_collab_button.toggled.connect(_on_collab_toggled)
	_collab_button.visible = false
	$Panel.add_child(_collab_button)
	_collab_button.position = Vector2(1030, 140)

	# 获取API客户端
	api_client = get_node_or_null("/root/APIClient")
	if api_client:
		# 普通对话
		api_client.chat_response_received.connect(_on_chat_response_received)
		api_client.chat_error.connect(_on_chat_error)
		# 流式对话
		api_client.chat_stream_chunk.connect(_on_chat_stream_chunk)
		api_client.chat_stream_finished.connect(_on_chat_stream_finished)
		api_client.chat_stream_error.connect(_on_chat_stream_error)
		# 多Agent协作
		api_client.agent_chat_response_received.connect(_on_agent_chat_response_received)

	print("[INFO] 对话UI初始化完成")

# ⭐ 处理对话框快捷键
func _input(event: InputEvent):
	if not visible:
		return

	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_ESCAPE:
			hide_dialogue()
			get_viewport().set_input_as_handled()
			return

		if event.keycode == KEY_ENTER or event.keycode == KEY_KP_ENTER:
			if player_input.has_focus():
				return
			send_message()
			get_viewport().set_input_as_handled()
			return

		# 屏蔽移动键和交互键
		if event.keycode in [KEY_E, KEY_SPACE, KEY_W, KEY_A, KEY_S, KEY_D]:
			get_viewport().set_input_as_handled()

func start_dialogue(npc_name: String):
	"""开始与NPC对话"""
	current_npc_name = npc_name

	# 通知NPC进入交互状态
	var npc = get_npc_by_name(npc_name)
	if npc and npc.has_method("set_interacting"):
		npc.set_interacting(true)

	# 设置NPC信息
	npc_name_label.text = npc_name
	npc_title_label.text = Config.NPC_TITLES.get(npc_name, "")

	# 清空对话内容
	dialogue_text.clear()
	dialogue_text.append_text("[color=gray]与 " + npc_name + " 的对话开始...[/color]\n")

	# 清空输入框
	player_input.text = ""

	# 显示协作按钮
	_collab_button.visible = true
	_collab_mode = false
	_collab_button.button_pressed = false

	# 显示对话框
	show_dialogue()

	# 聚焦输入框
	player_input.grab_focus()

	print("[INFO] 开始对话: ", npc_name)

func show_dialogue():
	"""显示对话框"""
	visible = true
	var player = get_tree().get_first_node_in_group("player")
	if player and player.has_method("set_interacting"):
		player.set_interacting(true)

func hide_dialogue():
	"""隐藏对话框"""
	_cleanup_streaming()
	visible = false
	_collab_button.visible = false

	if current_npc_name != "":
		var npc = get_npc_by_name(current_npc_name)
		if npc and npc.has_method("set_interacting"):
			npc.set_interacting(false)

	current_npc_name = ""

	var player = get_tree().get_first_node_in_group("player")
	if player and player.has_method("set_interacting"):
		player.set_interacting(false)

# ==================== 加载动画 ====================
func _start_loading():
	"""启动加载动画"""
	_loading_dots = 0
	_loading_label.text = "🤔 思考中"
	_loading_label.visible = true
	_loading_timer.start()

func _stop_loading():
	"""停止加载动画"""
	_loading_timer.stop()
	_loading_label.visible = false

func _on_loading_timer_timeout():
	"""加载动画计时器更新"""
	_loading_dots = (_loading_dots + 1) % 4
	var dots = ""
	for _i in range(_loading_dots):
		dots += "."
	_loading_label.text = "🤔 思考中" + dots

# ==================== 协作模式 ====================
func _on_collab_toggled(toggled: bool):
	"""协作模式切换"""
	_collab_mode = toggled
	if toggled:
		_collab_button.text = "🤝 协作模式 ON"
		npc_name_label.text = "团队协作"
		npc_title_label.text = "需求多 · 设计多 · 技术多"
		dialogue_text.clear()
		dialogue_text.append_text("[color=gold]🤝 进入协作模式，输入需求即可启动团队协作[/color]\n")
	else:
		_collab_button.text = "🤝 协作模式"
		npc_name_label.text = current_npc_name
		npc_title_label.text = Config.NPC_TITLES.get(current_npc_name, "")
		dialogue_text.clear()
		dialogue_text.append_text("[color=gray]与 " + current_npc_name + " 的对话开始...[/color]\n")

# ==================== 发送消息 ====================
func _on_send_button_pressed():
	send_message()

func _on_text_submitted(_text: String):
	send_message()

func send_message():
	"""发送消息（根据模式选择不同API）"""
	if _is_streaming:
		print("[WARN] 正在接收回复中，请等待")
		return

	var message = player_input.text.strip_edges()
	if message.is_empty():
		return

	if current_npc_name.is_empty() and not _collab_mode:
		print("[ERROR] 没有选择NPC")
		return

	# 显示玩家消息
	dialogue_text.append_text("\n[color=cyan]玩家:[/color] " + message + "\n")

	# 清空输入框
	player_input.text = ""

	# 启动加载动画
	_start_loading()

	if _collab_mode:
		# 协作模式：调用多Agent API
		_stream_buffer = ""
		if api_client:
			api_client.send_agent_chat(message)
		else:
			_stop_loading()
			dialogue_text.append_text("[color=red]错误: API客户端未找到[/color]\n")
	else:
		# 普通模式：调用流式对话API
		_is_streaming = true
		_stream_buffer = ""
		if api_client:
			api_client.send_chat_stream(current_npc_name, message)
		else:
			_is_streaming = false
			_stop_loading()
			dialogue_text.append_text("[color=red]错误: API客户端未找到[/color]\n")

# ==================== 流式响应处理 ====================
func _on_chat_stream_chunk(_npc_name: String, text: String):
	"""收到流式文本块"""
	if _loading_label.visible:
		_stop_loading()

	_stream_buffer += text
	# 实时更新对话框
	dialogue_text.append_text(text)
	dialogue_text.scroll_to_line(dialogue_text.get_line_count() - 1)

func _on_chat_stream_finished(npc_name: String, full_text: String):
	"""流式对话完成"""
	_is_streaming = false
	_stop_loading()

	# 显示完整的NPC回复
	if not _stream_buffer.is_empty():
		# 已经逐字追加了，不需要重复显示
		pass
	elif not full_text.is_empty():
		dialogue_text.append_text("[color=yellow]" + npc_name + ":[/color] " + full_text + "\n")

	dialogue_text.scroll_to_line(dialogue_text.get_line_count() - 1)

	# 更新NPC头顶对话气泡
	var npc = get_npc_by_name(npc_name)
	if npc and npc.has_method("update_dialogue"):
		npc.update_dialogue(full_text)

	print("[INFO] 流式对话完成: ", npc_name)

func _on_chat_stream_error(npc_name: String, error_msg: String):
	"""流式对话错误"""
	_is_streaming = false
	_stop_loading()
	dialogue_text.append_text("[color=red]连接错误: " + error_msg + "[/color]\n")
	print("[WARN] 流式失败: ", error_msg)

# ==================== 普通响应处理（备用） ====================
func _on_chat_response_received(npc_name: String, message: String):
	"""收到普通NPC回复（流式失败时的降级）"""
	_stop_loading()
	if npc_name != current_npc_name:
		return

	dialogue_text.append_text("[color=yellow]" + npc_name + ":[/color] " + message + "\n")
	dialogue_text.scroll_to_line(dialogue_text.get_line_count() - 1)

	var npc = get_npc_by_name(npc_name)
	if npc and npc.has_method("update_dialogue"):
		npc.update_dialogue(message)

# ==================== 多Agent协作响应处理 ====================
func _on_agent_chat_response_received(message: String):
	"""收到多Agent协作响应"""
	_stop_loading()

	# 按角色分段显示
	var sections = message.split("\n\n")
	for section in sections:
		section = section.strip_edges()
		if section.is_empty():
			continue
		# 检测角色标题
		if section.begins_with("【"):
			# 角色标题用金色显示
			dialogue_text.append_text("\n" + section + "\n")
		else:
			dialogue_text.append_text(section + "\n")

	dialogue_text.scroll_to_line(dialogue_text.get_line_count() - 1)
	print("[INFO] 协作回复完成")

# ==================== 错误处理 ====================
func _on_chat_error(error_message: String):
	"""对话错误"""
	_is_streaming = false
	_stop_loading()
	dialogue_text.append_text("[color=red]错误: " + error_message + "[/color]\n")

# ==================== 清理 ====================
func _cleanup_streaming():
	"""清理流式状态"""
	_is_streaming = false
	_stop_loading()

func _on_close_button_pressed():
	hide_dialogue()

# ==================== 工具函数 ====================
func get_npc_by_name(npc_name: String) -> Node:
	"""根据名字获取NPC节点"""
	var npcs = get_tree().get_nodes_in_group("npcs")
	for npc in npcs:
		if npc.npc_name == npc_name:
			return npc
	return null