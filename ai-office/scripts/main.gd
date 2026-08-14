# 主场景脚本
extends Node2D

# NPC节点引用
@onready var npc_UI: Node2D = $NPCs/NPC_UI
@onready var npc_Dev: Node2D = $NPCs/NPC_Dev
@onready var npc_PM: Node2D = $NPCs/NPC_PM

@onready var npc_dict: Dictionary = {
	Config.NPC_NAMES[0]: npc_PM,
	Config.NPC_NAMES[1]: npc_Dev,
	Config.NPC_NAMES[2]: npc_UI,
}


func _ready():
	print("[INFO] 主场景初始化")

func update_npc_dialogue(npc_name: String, dialogue: String):
	"""更新指定NPC的对话 (由dialogue_ui调用)"""
	var npc_node = get_npc_node(npc_name)
	if npc_node and npc_node.has_method("update_dialogue"):
		npc_node.update_dialogue(dialogue)

#func get_npc_node(npc_name: String) -> Node2D:
	#"""根据名字获取NPC节点"""
	#match npc_name:
		#Config.NPC_NAMES[2]:  # 设计多
			#return npc_UI
		#Config.NPC_NAMES[1]:  # 技术多
			#return npc_Dev
		#Config.NPC_NAMES[0]:  # 需求多
			#return npc_PM
		#_:
			#return null
func get_npc_node(npc_name: String) -> Node2D:
	return npc_dict.get(npc_name)
