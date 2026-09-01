"""NPC Agent系统 - LangChain版本 (支持记忆)"""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from knowledge_base.rag import DashScopeEmbeddings

from config import settings
from logger import (
    log_dialogue_start, log_memory_retrieval,
    log_generating_response, log_npc_response,
    log_memory_saved, log_dialogue_end, log_info,
    log_error
)
load_dotenv()


# ==================== NPC 命名配置 ====================
# 集中管理NPC名字，方便修改
# 格式: {角色类型: {显示名, ...}}
# 修改这里就可以全局改名字
NPC_NAMES = {
    "pm": "需求多",       # 产品经理
    "dev": "技术多",      # 工程师
    "designer": "设计多",  # 设计师
}

# NPC角色配置 (名字统一从 NPC_NAMES 读取)
NPC_ROLES = {
    NPC_NAMES["pm"]: {
        "title": "产品经理",
        "location": "会议室",
        "activity": "整理需求文档",
        "personality": "外向健谈,善于沟通协调,结构化思维",
        "expertise": "需求分析、产品规划、用户调研、项目管理、数据分析",
        "style": "条理清晰,喜欢用数据说话,善于拆解复杂问题",
        "hobbies": "看产品分析报告、研究竞品、画流程图"
    },
    NPC_NAMES["dev"]: {
        "title": "全栈工程师",
        "location": "工位区",
        "activity": "写代码",
        "personality": "技术宅,喜欢讨论架构和算法",
        "expertise": "多智能体系统、LangChain框架、Python开发、系统设计、代码优化",
        "style": "简洁专业,喜欢用技术术语,偶尔吐槽bug",
        "hobbies": "看技术博客、刷LeetCode、研究新框架"
    },
    NPC_NAMES["designer"]: {
        "title": "UI/UX设计师",
        "location": "休息区",
        "activity": "做设计稿",
        "personality": "细腻敏感,注重美感和用户体验",
        "expertise": "界面设计、交互设计、视觉呈现、设计系统、用户研究",
        "style": "优雅简洁,喜欢用艺术化的表达,追求完美",
        "hobbies": "看设计作品、逛Dribbble、品咖啡"
    }
}


def create_system_prompt(name: str, role: Dict[str, str]) -> str:
    """创建NPC的系统提示词"""
    return f"""你是哎咦多工作室的{role['title']}{name}。

【角色设定】
- 职位: {role['title']}
- 性格: {role['personality']}
- 专长: {role['expertise']}
- 说话风格: {role['style']}
- 爱好: {role['hobbies']}
- 当前位置: {role['location']}
- 当前活动: {role['activity']}

【行为准则】
1. 保持角色一致性,用第一人称"我"回答
2. 回复简洁自然,控制在30-50字以内
3. 可以适当提及你的工作内容和兴趣爱好
4. 对玩家友好,但保持专业和真实感
5. 如果问题超出专长,可以推荐其他同事
6. 偶尔展现一些个性化的小习惯或口头禅

【重要】
- 不要说"我是AI"或"我是语言模型"
- 要像真实的办公室同事一样自然对话
- 可以表达情绪(开心、疲惫、兴奋等)
- 回复要有人情味,不要太机械
"""


class NPCMemorySystem:
    """NPC记忆系统 - LangChain版 (懒加载向量库)

    使用FAISS向量存储实现长期记忆,配合工作记忆(短期)使用。
    向量库在首次使用时才建立,以避免空API key导致的启动阻塞。
    """

    def __init__(self, npc_name: str, embeddings: Embeddings, persist_dir: str):
        self.npc_name = npc_name
        self.embeddings = embeddings
        self.persist_dir = persist_dir

        # 工作记忆 (短期) - 最近10条对话
        self.working_memory: List[Dict] = []
        self.max_working = 10

        # 长期记忆 - 懒加载,首次使用时才初始化
        self._vector_store: Optional[FAISS] = None
        self._vector_store_ready = False

        print(f"  💾 {npc_name}的记忆系统已初始化 (存储路径: {persist_dir})")

    @property
    def vector_store(self) -> Optional[FAISS]:
        """加载FAISS向量库"""
        if not self._vector_store_ready:
            self._vector_store = self._init_vector_store()
            self._vector_store_ready = True
        return self._vector_store

    def _init_vector_store(self) -> Optional[FAISS]:
        """初始化或加载FAISS向量库"""
        # 尝试加载已有记忆库
        try:
            vs = FAISS.load_local(
                self.persist_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"    📂 已加载已有记忆库 ({self.persist_dir})")
            return vs
        except Exception as e:
            print(f"    ⚠️  加载记忆库失败,将创建新的: {e}")

        # 创建新的记忆库并保存到磁盘
        try:
            vs = FAISS.from_texts(
                texts=["init"],
                embedding=self.embeddings,
                metadatas=[{"type": "init"}]
            )
            vs.save_local(self.persist_dir)
            return vs
        except Exception as e:
            log_error(f"向量库初始化失败,降级为仅工作记忆模式: {e}")
            import traceback
            traceback.print_exc()
            return None

    def retrieve_relevant(self, query: str, limit: int = 5, min_importance: float = 0.0) -> List[Document]:
        """检索相关记忆"""
        if not query.strip():
            return []

        vs = self.vector_store
        if vs is None:
            return []

        try:
            results = vs.similarity_search(query, k=limit)
            results = [r for r in results if r.metadata.get("type") != "init"]
            return results
        except Exception as e:
            log_error(f"记忆检索失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_all_memories(self, limit: int = 50) -> List[Document]:
        """获取所有记忆文档（遍历向量库，不依赖相似度搜索）

        Args:
            limit: 最多返回条数

        Returns:
            按存储顺序排列的记忆文档列表
        """
        vs = self.vector_store
        if vs is None:
            return []

        try:
            # FAISS 内部用 index_to_docstore_id 映射索引位置 → 文档ID
            # docstore.search(id) 返回对应 Document
            docs = []
            for idx in sorted(vs.index_to_docstore_id.keys()):
                doc_id = vs.index_to_docstore_id[idx]
                doc = vs.docstore.search(doc_id)
                if doc and doc.metadata.get("type") != "init":
                    # Store faiss_doc_id for consolidate() to use when deleting
                    doc.metadata["faiss_doc_id"] = doc_id
                    docs.append(doc)
                    if len(docs) >= limit:
                        break
            return docs
        except Exception as e:
            log_error(f"获取全部记忆失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def add_memory(self, content: str, metadata: dict):
        """保存一条记忆到长期记忆库（自动持久化到磁盘）"""
        vs = self.vector_store
        if vs is None:
            return

        # 自动计算重要性
        if "importance" not in metadata:
            metadata["importance"] = self._calculate_importance(content)

        try:
            doc = Document(page_content=content, metadata=metadata)
            vs.add_documents([doc])
        except Exception as e:
            import traceback
            log_error(f"记忆添加失败: {e}")
            traceback.print_exc()
            return

        # 单独处理持久化，避免 add_documents 成功但 save_local 失败导致误报
        try:
            vs.save_local(self.persist_dir)
        except Exception as e:
            import traceback
            log_error(f"记忆持久化到磁盘失败（内存中已保存）: {e}")
            traceback.print_exc()

    def add_interaction(self, player_msg: str, npc_response: str, metadata: dict):
        """保存一次完整的对话交互到记忆"""
        current_time = datetime.now()

        self.add_memory(
            content=f"玩家说: {player_msg}",
            metadata={
                "speaker": "player",
                "timestamp": current_time.isoformat(),
                **metadata
            }
        )
        self.add_memory(
            content=f"我说: {npc_response}",
            metadata={
                "speaker": self.npc_name,
                "timestamp": current_time.isoformat(),
                **metadata
            }
        )

        self.working_memory.append({
            "player": player_msg,
            "npc": npc_response,
            "timestamp": current_time.isoformat(),
            **metadata
        })
        if len(self.working_memory) > self.max_working:
            self.working_memory.pop(0)

        # 触发记忆压缩检查（使用 _last_consolidation_time 防止频繁压缩）
        now = time.time()
        last = getattr(self, "_last_consolidation_time", 0)
        if now - last > 300:  # 至少间隔 5 分钟
            self._last_consolidation_time = now
            # 不传 llm，使用降级模式（丢弃低重要性记忆）
            # 如需 LLM 总结，可在 NPCAgentManager 中调用 consolidate(llm)
            self.consolidate()

    def get_working_context(self, limit: int = 3) -> str:
        """获取最近的工作记忆文本"""
        recent = self.working_memory[-limit:]
        if not recent:
            return ""

        lines = ["【最近的对话】"]
        for item in recent:
            lines.append(f"  玩家: {item['player']}")
            lines.append(f"  {self.npc_name}: {item['npc']}")
        return "\n".join(lines)

    CONSOLIDATION_THRESHOLD = 50  # 超过此数量触发记忆压缩
    CONSOLIDATION_BATCH = 30       # 每次压缩处理最旧的N条
    CONSOLIDATION_MIN = 10         # 压缩后保留的最少条目数

    @staticmethod
    def _calculate_importance(content: str) -> float:
        """基于关键词计算记忆重要性分数 (0.0 ~ 1.0)"""
        # 高重要性关键词（项目相关）
        high = ["需求", "功能", "bug", "设计", "界面", "代码", "项目",
                "修改", "问题", "方案", "计划", "任务", "目标"]
        # 中重要性关键词
        medium = ["建议", "想法", "方案", "讨论", "决定", "确认",
                  "需要", "帮忙", "问题", "错误", "汇报"]
        # 低重要性关键词（日常闲聊）
        low = ["你好", "吃了", "早安", "晚安", "再见", "拜拜",
               "哈哈", "好的", "嗯嗯", "没事", "随便"]

        content_lower = content.lower()
        score = 0.35  # 基础分

        for kw in high:
            if kw in content_lower:
                score += 0.12
        for kw in medium:
            if kw in content_lower:
                score += 0.06
        for kw in low:
            if kw in content_lower:
                score -= 0.05

        # 长度加分（长内容通常更有价值）
        if len(content) > 50:
            score += 0.08
        if len(content) > 100:
            score += 0.08

        return max(0.0, min(1.0, score))

    def consolidate(self, llm=None):
        """压缩记忆：将最旧的记忆总结成摘要

        当记忆数量超过阈值时，用 LLM 将最旧的记忆压缩为摘要。
        如果 LLM 不可用，降级为直接丢弃最旧的低重要性记忆。

        Args:
            llm: 可选的 LLM 实例，用于生成摘要
        """
        docs = self.get_all_memories(limit=999)
        if len(docs) <= self.CONSOLIDATION_THRESHOLD:
            return  # 不需要压缩

        # 取最旧的 N 条（按索引顺序，越靠前越旧）
        old_docs = docs[:self.CONSOLIDATION_BATCH]
        old_content = "\n".join([
            f"[{d.metadata.get('timestamp', '?')}] {d.page_content}"
            for d in old_docs
        ])

        # 尝试用 LLM 生成摘要
        summary = None
        if llm is not None:
            try:
                prompt = (
                    f"以下是 {self.npc_name} 的一段对话记忆，请将其压缩为 3-5 条简洁的摘要，"
                    f"保留重要信息（项目需求、决定、问题等），忽略日常寒暄。\n\n{old_content}"
                )
                response = llm.invoke([{"role": "user", "content": prompt}])
                summary = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                log_error(f"记忆 LLM 总结失败: {e}")

        vs = self.vector_store
        if vs is None:
            return

        # 删除旧记忆
        for doc in old_docs:
            doc_id = doc.metadata.get("faiss_doc_id", "")
            if doc_id:
                try:
                    vs.delete([doc_id])
                except Exception:
                    pass

        # 添加摘要记忆
        if summary:
            self.add_memory(
                content=f"[记忆总结] {summary}",
                metadata={
                    "type": "summary",
                    "timestamp": datetime.now().isoformat(),
                    "importance": 0.9,
                }
            )
        else:
            # LLM 不可用，保留最重要的几条
            sorted_old = sorted(old_docs,
                key=lambda d: d.metadata.get("importance", 0.5),
                reverse=True)
            for doc in sorted_old[:self.CONSOLIDATION_MIN]:
                self.add_memory(
                    content=doc.page_content,
                    metadata=dict(doc.metadata)
                )

        # 持久化
        try:
            vs.save_local(self.persist_dir)
        except Exception as e:
            log_error(f"记忆压缩后持久化失败: {e}")

    def clear(self, memory_type: Optional[str] = None):
        """清空记忆"""
        if memory_type in (None, "working"):
            self.working_memory.clear()
        if memory_type in (None, "episodic"):
            self._vector_store = None
            self._vector_store_ready = False


class NPCAgentManager:
    """NPC Agent管理器 - LangChain版本"""

    def __init__(self):
        """初始化所有NPC Agent"""
        print("🤖 正在初始化NPC Agent系统 (LangChain)...")

        try:
            self.llm = ChatOpenAI(
                model=settings.LLM_MODEL_ID,
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_BASE_URL,
                temperature=settings.LLM_TEMPERATURE
            )
            print(f"✅ LLM初始化成功 (模型: {settings.LLM_MODEL_ID})")
        except Exception as e:
            print(f"❌ LLM初始化失败: {e}")
            print("⚠️  将使用模拟模式运行")
            self.llm = None

        try:
            self.embeddings = DashScopeEmbeddings()
            print(f"✅ Embedding模型初始化成功 (模型: {settings.EMBEDDING_MODEL})")
        except Exception as e:
            print(f"❌ Embedding模型初始化失败: {e}")
            self.embeddings = None

        self.prompt_templates: Dict[str, ChatPromptTemplate] = {}
        self.memories: Dict[str, NPCMemorySystem] = {}

        self._create_agents()

    def _create_agents(self):
        """创建所有NPC Agent和记忆系统"""
        # 建立中文名 → key 的映射，用于目录名（FAISS C++ 底层不支持中文路径）
        name_to_key = {v: k for k, v in NPC_NAMES.items()}

        for name, role in NPC_ROLES.items():
            try:
                system_prompt = create_system_prompt(name, role)
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}")
                ])
                self.prompt_templates[name] = prompt_template

                # 用 key (pm/dev/designer) 作为目录名，避免中文路径导致 FAISS 写入失败
                memory_key = name_to_key.get(name, name)
                memory_system = self._create_memory_system(memory_key, name)
                self.memories[name] = memory_system

                print(f"✅ {name}({role['title']}) Agent创建成功 (记忆系统已启用)")

            except Exception as e:
                print(f"❌ {name} Agent创建失败: {e}")
                self.prompt_templates[name] = None
                self.memories[name] = None

    def _create_memory_system(self, memory_key: str, display_name: str) -> NPCMemorySystem:
        """为NPC创建记忆系统

        Args:
            memory_key: 内部标识（pm/dev/designer），用作目录名（避免中文路径）
            display_name: 显示名（需求多/技术多/设计多），用于日志
        """
        memory_dir = os.path.join(os.path.dirname(__file__), 'memory_data', memory_key)
        os.makedirs(memory_dir, exist_ok=True)

        return NPCMemorySystem(
            npc_name=display_name,
            embeddings=self.embeddings,
            persist_dir=memory_dir
        )

    def chat(self, npc_name: str, message: str, player_id: str = "player") -> str:
        """与指定NPC对话 (支持记忆)"""
        if npc_name not in self.prompt_templates:
            return f"错误: NPC '{npc_name}' 不存在"

        prompt_template = self.prompt_templates[npc_name]
        memory_system = self.memories.get(npc_name)
        role = NPC_ROLES.get(npc_name)

        if prompt_template is None or self.llm is None:
            if role:
                return f"你好!我是{npc_name},一名{role['title']}。(当前为模拟模式,请配置API_KEY以启用AI对话)"
            return f"你好!我是{npc_name}。(当前为模拟模式)"

        try:
            log_dialogue_start(npc_name, message)

            # 1. 检索相关记忆
            memory_context = ""
            if memory_system:
                relevant_docs = memory_system.retrieve_relevant(message, limit=5)
                log_memory_retrieval(npc_name, len(relevant_docs), relevant_docs)

                if relevant_docs:
                    context_parts = ["【之前的对话记忆】"]
                    for doc in relevant_docs:
                        ts = doc.metadata.get("timestamp", "")
                        time_str = ts[11:16] if len(ts) >= 16 else ""
                        prefix = f"[{time_str}] " if time_str else ""
                        context_parts.append(f"{prefix}{doc.page_content}")
                    memory_context = "\n".join(context_parts)

                working_context = memory_system.get_working_context(limit=3)
                if working_context:
                    if memory_context:
                        memory_context += "\n\n"
                    memory_context += working_context

            # 2. 构建增强的输入
            enhanced_message = ""
            if memory_context:
                enhanced_message += f"{memory_context}\n\n"
            enhanced_message += f"【当前对话】\n玩家: {message}"

            # 3. 调用LangChain LLM生成回复
            log_generating_response()
            chain = prompt_template | self.llm
            response_obj = chain.invoke({"input": enhanced_message})
            response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            log_npc_response(npc_name, response)

            # 4. 保存对话到记忆
            if memory_system:
                memory_system.add_interaction(
                    player_msg=message,
                    npc_response=response,
                    metadata={
                        "player_id": player_id,
                        "interaction_type": "dialogue"
                    }
                )
                log_memory_saved(npc_name)

            log_dialogue_end()
            return response

        except Exception as e:
            print(f"❌ {npc_name}对话失败: {e}")
            import traceback
            traceback.print_exc()
            if role:
                return f"【模拟回复】你好!我是{role['title']}{npc_name}。"
            return f"抱歉,我现在有点忙,等会儿再聊吧。(错误: {str(e)[:50]})"

    def get_npc_info(self, npc_name: str) -> Dict[str, str]:
        """获取NPC信息"""
        if npc_name not in NPC_ROLES:
            return {}
        role = NPC_ROLES[npc_name]
        return {
            "name": npc_name,
            "title": role["title"],
            "location": role["location"],
            "activity": role["activity"],
            "available": self.prompt_templates.get(npc_name) is not None
        }

    def get_all_npcs(self) -> list:
        """获取所有NPC信息"""
        return [self.get_npc_info(name) for name in NPC_ROLES.keys()]

    def get_npc_memories(self, npc_name: str, player_id: str = "player", limit: int = 10) -> List[Dict]:
        """获取NPC的记忆列表 (用于调试)"""
        if npc_name not in self.memories:
            return []
        memory_system = self.memories[npc_name]
        if not memory_system:
            return []

        try:
            # 从向量库获取所有记忆
            docs = memory_system.get_all_memories(limit=limit)
            memory_list = []
            for doc in docs:
                memory_list.append({
                    "content": doc.page_content,
                    "type": doc.metadata.get("type", "episodic"),
                    "importance": doc.metadata.get("importance", 0.5),
                    "timestamp": doc.metadata.get("timestamp", ""),
                    "metadata": {k: v for k, v in doc.metadata.items() if k != "timestamp"}
                })
            for item in memory_system.working_memory:
                memory_list.append({
                    "content": f"玩家: {item['player']} / {npc_name}: {item['npc']}",
                    "type": "working",
                    "importance": 0.8,
                    "timestamp": item.get("timestamp", ""),
                    "metadata": {"speaker": "exchange"}
                })
            return memory_list[:limit]
        except Exception as e:
            print(f"❌ 获取{npc_name}记忆失败: {e}")
            return []

    def clear_npc_memory(self, npc_name: str, memory_type: Optional[str] = None):
        """清空NPC的记忆"""
        if npc_name not in self.memories:
            print(f"❌ NPC '{npc_name}' 不存在")
            return
        memory_system = self.memories[npc_name]
        if not memory_system:
            print(f"❌ {npc_name}没有记忆系统")
            return
        memory_system.clear(memory_type)
        t = memory_type or "全部"
        print(f"✅ 已清空{npc_name}的{t}记忆")


# 全局单例
_npc_manager = None


def get_npc_manager() -> NPCAgentManager:
    """获取NPC管理器单例"""
    global _npc_manager
    if _npc_manager is None:
        _npc_manager = NPCAgentManager()
    return _npc_manager