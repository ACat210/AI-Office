"""知识库管理 - 加载文档、切片、建立索引"""

import os
import glob
from typing import List, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from config import settings
from logger import log_info, log_error

# 知识库根目录
KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_base"


class DashScopeEmbeddings(Embeddings):
    """DashScope Embedding 适配器

    实现 LangChain Embeddings 接口，直接调用 DashScope 的 embedding API。
    用于替换 langchain-openai 的 OpenAIEmbeddings（它在 DashScope 上会 tokenization 失败）。

    用法:
        embeddings = DashScopeEmbeddings()
        vectors = embeddings.embed_documents(["文本1", "文本2"])
        vector = embeddings.embed_query("查询文本")
    """

    def __init__(self):
        self.model = settings.EMBEDDING_MODEL
        self.api_key = settings.EMBEDDING_API_KEY or settings.LLM_API_KEY
        self.base_url = settings.LLM_BASE_URL

    def embed_documents(self, texts: list) -> list:
        """批量 embedding"""
        return _dashscope_embed(texts)

    def embed_query(self, text: str) -> list:
        """单条查询 embedding"""
        return _dashscope_embed([text])[0]


def load_markdown_files(directory: str) -> List[Document]:
    """加载指定目录下的所有 markdown 文件

    Args:
        directory: 知识库目录路径

    Returns:
        Document 列表
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        log_info(f"  ⚠️  知识库目录不存在: {directory}")
        return []

    md_files = glob.glob(str(dir_path / "**/*.md"), recursive=True)
    documents = []

    for filepath in md_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # 文件名作为文档标题
            filename = Path(filepath).stem
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(filepath),
                    "title": filename,
                    "type": "design_spec",
                }
            )
            documents.append(doc)
            log_info(f"    📄 加载文档: {filename}")
        except Exception as e:
            log_error(f"    ⚠️  加载文档失败 {filepath}: {e}")

    return documents


def chunk_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """将文档切成小块

    Args:
        documents: 原始文档列表
        chunk_size: 每块字符数
        chunk_overlap: 块之间重叠字符数

    Returns:
        切块后的 Document 列表
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", " "],
    )

    chunks = splitter.split_documents(documents)
    log_info(f"    ✂️  文档切片: {len(documents)} 篇 → {len(chunks)} 块")
    return chunks


def _dashscope_embed(texts: list) -> list:
    """直接调用 DashScope 的 embedding API

    绕过 langchain-openai 的 tokenization 问题，
    直接发送原始文本给 DashScope。

    Args:
        texts: 文本列表

    Returns:
        向量列表
    """
    import requests as req
    import json

    url = f"{settings.LLM_BASE_URL.rstrip('/')}/embeddings"
    api_key = settings.EMBEDDING_API_KEY or settings.LLM_API_KEY
    model = settings.EMBEDDING_MODEL

    if not api_key:
        raise ValueError("未设置 API Key")

    payload = {
        "model": model,
        "input": texts,
        "encoding_format": "float",
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        resp = req.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # 按输入顺序提取向量
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]
    except Exception as e:
        log_error(f"  ⚠️  DashScope embedding 调用失败: {e}")
        raise


class DesignKnowledgeBase:
    """设计规范知识库

    管理 设计多 使用的 RAG 知识库。
    使用 Qdrant in-memory 模式，无需额外部署。
    """

    def __init__(self):
        self.client = QdrantClient(":memory:")
        self.collection_name = "design_specs"
        self.ready = False
        self._chunks: List[Document] = []  # 纯文本副本，用于降级搜索

    def initialize(self, force_rebuild: bool = False):
        """初始化知识库：加载文档 → 切片 → 向量化 → 存入 Qdrant

        Args:
            force_rebuild: 是否强制重建
        """
        if self.ready and not force_rebuild:
            return

        log_info("  📚 初始化设计规范知识库...")

        # 1. 加载文档
        design_dir = KNOWLEDGE_BASE_DIR / "design"
        docs = load_markdown_files(str(design_dir))
        if not docs:
            log_info("  ⚠️  没有找到设计文档，知识库为空")
            return

        # 2. 切片
        chunks = chunk_documents(docs)
        self._chunks = chunks  # 保存一份纯文本副本，作为降级方案

        # 3. 向量化并存入 Qdrant
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        try:
            vectors = _dashscope_embed(texts)
            vector_size = len(vectors[0]) if vectors else 1024

            # 创建 collection
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

            # 存入向量
            points = [
                PointStruct(
                    id=i,
                    vector=vectors[i],
                    payload={
                        "text": texts[i],
                        **metadatas[i],
                    }
                )
                for i in range(len(texts))
            ]
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            self.ready = True
            log_info(f"  ✅ 知识库初始化完成: {len(chunks)} 个向量已存入")
        except Exception as e:
            log_error(f"  ⚠️  知识库向量化失败: {e}")
            log_info("  ↪ 降级为文本关键词搜索模式")
            self.ready = True  # 仍然标记为可用，使用降级搜索

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """搜索最相关的设计规范

        Args:
            query: 查询文本
            top_k: 返回前几条结果

        Returns:
            相关文档列表
        """
        if not self.ready:
            return self._fallback_search(query, top_k)

        try:
            query_vector = _dashscope_embed([query])[0]
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
            ).points

            hits = []
            for r in results:
                payload = r.payload or {}
                hits.append({
                    "text": payload.get("text", ""),
                    "title": payload.get("title", ""),
                    "source": payload.get("source", ""),
                    "score": r.score,
                })
            return hits

        except Exception as e:
            log_error(f"  ⚠️  知识库向量检索失败，降级为关键词搜索: {e}")
            return self._fallback_search(query, top_k)

    def _fallback_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """降级搜索：关键词匹配（当向量库不可用时）

        使用关键词匹配，支持中文分词，按命中次数排序。
        """
        if not self._chunks:
            return []

        # 提取关键词
        import re
        # 中文：提取所有2字符以上的连续汉字组合
        chinese_phrases = re.findall(r'[一-鿿]{2,}', query)
        # 英文：按单词拆分
        english_words = re.findall(r'[a-zA-Z_]{2,}', query.lower())
        # 中文还提取单个有意义的2-gram
        all_chars = re.findall(r'[一-鿿]', query)
        bigrams = set()
        for i in range(len(all_chars) - 1):
            bigrams.add(all_chars[i] + all_chars[i+1])
        keywords = set(chinese_phrases + english_words) | bigrams

        if not keywords:
            # 如果没提取到有效关键词，返回前几条
            return [
                {
                    "text": chunk.page_content,
                    "title": chunk.metadata.get("title", ""),
                    "source": chunk.metadata.get("source", ""),
                    "score": 0.1,
                }
                for chunk in self._chunks[:top_k]
            ]

        scored = []
        for chunk in self._chunks:
            text = chunk.page_content
            title = chunk.metadata.get("title", "")
            source = chunk.metadata.get("source", "")

            # 计算关键词命中次数
            score = 0
            for kw in keywords:
                score += text.count(kw) * 2 if kw in text else 0

            # 标题命中大幅加分
            for kw in keywords:
                if kw in title:
                    score += 10

            if score > 0:
                scored.append({
                    "text": chunk.page_content,
                    "title": title,
                    "source": source,
                    "score": score,
                })

        # 按分数排序
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def format_context(self, query: str, top_k: int = 3) -> str:
        """搜索并格式化为上下文文本

        Args:
            query: 查询文本
            top_k: 返回前几条结果

        Returns:
            格式化的上下文字符串，直接注入到 Agent 提示词中
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return ""

        lines = ["【设计规范参考】"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "未知")
            score = r.get("score", 0)
            text = r.get("text", "")[:300]  # 截断太长
            lines.append(f"\n--- 参考 {i}: {title} (相关性: {score:.2f}) ---")
            lines.append(text)

        return "\n".join(lines)


# 全局单例
_design_kb = None


def get_design_knowledge_base() -> DesignKnowledgeBase:
    """获取设计知识库单例"""
    global _design_kb
    if _design_kb is None:
        _design_kb = DesignKnowledgeBase()
    return _design_kb