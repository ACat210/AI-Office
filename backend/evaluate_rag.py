"""RAG 系统评估脚本 (使用 RAGAS + 传统检索指标)

评估维度：
- faithfulness: 生成答案是否忠于检索到的上下文（不胡编）
- answer_relevancy: 答案是否与问题相关
- context_precision: 检索到的上下文中有多少是真正相关的
- context_recall: 检索到的上下文是否覆盖了 ground truth 中的关键信息
- precision@k: 检索结果中相关文档的比例
- recall@k: 检索结果覆盖了多少相关文档
- MRR: 第一个相关文档出现在第几位

用法：
    cd backend
    python evaluate_rag.py
"""

import os
import sys
import json
from typing import List, Dict, Any

# 评估脚本不需要 LangSmith 追踪
os.environ["LANGSMITH_TRACING"] = "false"

# 修复 Windows GBK 编码问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# /backend 目录运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_base.rag import get_design_knowledge_base, _dashscope_embed
from config import settings
from logger import log_info, log_error

# ============================================================
# 1. 测试数据集
# ============================================================

TEST_QUESTIONS = [
    # === 设计价值观 ===
    {
        "question": "Arco Design 的设计价值观是什么？",
        "reference": "Arco Design 的设计价值观是「清晰」、「一致」、「韵律」和「开放」。",
        "category": "价值观",
    },
    {
        "question": "Arco Design 的「清晰」价值观具体指什么？",
        "reference": "「清晰」指效率的提升。清晰的设计体系使得产品操作直观，流程一步到位；信息传达清晰且表意明确，使用户在极短时间内快速理解进而作出判断。",
        "category": "价值观",
    },
    {
        "question": "Arco Design 的「一致」价值观有什么作用？",
        "reference": "「一致」的设计能产生品牌信赖感。一致的设计使品牌感呈现出系统性的传达，高标准的一致设计体系给用户带来品牌信赖感，同时还能通过一致的重复降低用户反复学习成本。",
        "category": "价值观",
    },
    {
        "question": "Arco Design 的「韵律」价值观是什么意思？",
        "reference": "「韵律」指推敲设计的韵律，对元素之间重复与对比等规律的潜在追求与把握，构成UI设计中的韵律之美。悦耳的韵律使得用户能够根据习惯轻松的完成任务。",
        "category": "价值观",
    },
    {
        "question": "Arco Design 的「开放」价值观是什么意思？",
        "reference": "「开放」指包容开放的体系能够有效解决多样化的问题。开放包容的思路让设计系统能够适应不同场景和需求。",
        "category": "价值观",
    },
    # === 设计原则 ===
    {
        "question": "Arco Design 设计原则中的「及时反馈」是什么？",
        "reference": "系统应该让用户知道目前的状态，并及时给予相对应的反馈。",
        "category": "原则",
    },
    {
        "question": "Arco Design 设计原则中的「贴近现实」是什么意思？",
        "reference": "系统应该用用户的语言，用词，短语和用户熟悉的概念，而不是系统术语。遵循现实世界的惯例，让信息符合自然思考逻辑。",
        "category": "原则",
    },
    {
        "question": "Arco Design 如何防止用户操作错误？",
        "reference": "比出现错误信息提示更好的是更用心的设计防止这类问题发生。在用户选择动作发生之前，就要防止用户容易混淆或者错误的选择。",
        "category": "原则",
    },
    {
        "question": "Arco Design 的「突出重点」原则是什么？",
        "reference": "用户的浏览动作不是读，不是看，而是扫。设计中应该突出重点，弱化和剔除无关信息。重要对话中不应该包含无关紧要的信息。",
        "category": "原则",
    },
    {
        "question": "Arco Design 的「遵从习惯」原则是什么？",
        "reference": "尽量减少用户对操作目标的记忆负荷，动作和选项都应该是可见的。比如填完表单，下一步应该生成表单，而不是下一步就是完成。",
        "category": "原则",
    },
    {
        "question": "Arco Design 的「人性化帮助」原则是什么？",
        "reference": "如果系统不需要使用帮助文档是最好的，但有必要时提供帮助文档也是必须的。帮助文档应该易于搜索，并提供具体的操作步骤。",
        "category": "原则",
    },
    # === 样式指南 - 色彩 ===
    {
        "question": "Arco Design 将色彩分为哪几类？",
        "reference": "Arco 将色彩分为主色、中性色、功能色和遮罩色。",
        "category": "色彩",
    },
    {
        "question": "Arco Design 的主色是什么？",
        "reference": "主色是一个产品的代表颜色，一般与品牌色相关联。Arco 基于主色通过动态梯度色彩算法衍生出13套基础色板。",
        "category": "色彩",
    },
    {
        "question": "Arco Design 的功能色有什么作用？",
        "reference": "功能色的主要作用是向用户明确的传达成功、警告、错误、链接等信息和状态。Arco 基于用户对色彩的通用认知，提供了适合不同状态的功能色及其配套色板。",
        "category": "色彩",
    },
    # === 样式指南 - 文字 ===
    {
        "question": "Arco Design 对界面文字的最小尺寸有什么要求？",
        "reference": "最小可识别文字尺寸为12px。为保障文本的易读性，界面文字需满足最小可识别文字尺寸的要求。",
        "category": "文字",
    },
    {
        "question": "Arco Design 的主字号是多少？",
        "reference": "Arco 将主字号定义为14px，并提供了不同层级的字号以适配不同信息层级的展示需求。",
        "category": "文字",
    },
    {
        "question": "Arco Design 推荐的默认行高是多少？",
        "reference": "Arco 默认的行高为1.4倍字体大小。",
        "category": "文字",
    },
    # === 样式指南 - 阴影 ===
    {
        "question": "Arco Design 中如何使用阴影？",
        "reference": "在界面中常用阴影来模拟元素的高度和层次关系。交互操作可以使用二级阴影，需要进行突出展示以及表示在空间上最上层的元素（如下拉菜单、模态框等）可以使用三级和四级阴影。",
        "category": "阴影",
    },
    # === 文案样式 ===
    {
        "question": "Arco Design 对日期与时间的格式有什么建议？",
        "reference": "Arco 建议使用24小时制，最大程度避免因格式不统一而带来的困惑与误解。",
        "category": "文案",
    },
    {
        "question": "Arco Design 的中英文排版规则是什么？",
        "reference": "中英文之间需要加空格。中文与链接之间增加空格。全角标点与英文或数字之间不加空格。遇到完整的英文句子使用半角标点。",
        "category": "文案",
    },
    {
        "question": "Arco Design 的用语原则有哪些？",
        "reference": "界面中的用语应遵循5个主要原则：词汇统一、语法正确、文案精炼、通俗易懂、语言友好。",
        "category": "文案",
    },
    # === 综合 ===
    {
        "question": "ArcoDesign 是什么？",
        "reference": "ArcoDesign 是一套设计系统的简称，基于 Byte Design 升级而来，是能力全面的企业级产品设计系统，主要服务于字节跳动旗下中后台产品的体验设计和技术实现。",
        "category": "综合",
    },
    {
        "question": "Arco Design 的「务实的浪漫主义」是什么意思？",
        "reference": "务实的浪漫主义是对设计语言的形容，代表着ArcoDesign试图建立的工作模式。务实=同理心，浪漫=想象力。浪漫与务实并非矛盾对立，而是相辅相成。",
        "category": "综合",
    },
]


# ============================================================
# 2. RAG 系统封装
# ============================================================

class RAGEvaluator:
    """RAG 评估器 - 封装检索和生成"""

    def __init__(self):
        # 初始化知识库
        self.kb = get_design_knowledge_base()
        self.kb.initialize()
        log_info(f"  ✅ 知识库已初始化")

        # 初始化 LLM
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL_ID,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=0.3,
        )
        log_info(f"  ✅ LLM 已初始化: {settings.LLM_MODEL_ID}")

    def retrieve(self, question: str, top_k: int = 3) -> List[str]:
        """检索相关文档"""
        results = self.kb.search(question, top_k=top_k)
        return [r["text"] for r in results]

    def generate(self, question: str, contexts: List[str]) -> str:
        """基于检索结果生成答案"""
        context_str = "\n\n".join([
            f"--- 参考 {i+1} ---\n{ctx[:500]}"
            for i, ctx in enumerate(contexts)
        ])

        prompt = f"""你是一个设计规范专家。请根据以下提供的设计规范文档，回答用户的问题。

【设计规范参考】
{context_str}

【问题】
{question}

【要求】
- 只基于上述设计规范文档中的信息回答
- 如果文档中没有相关信息，请明确说明
- 回答要简洁准确，直接回答问题
"""

        response = self.llm.invoke(prompt)
        return response.content

    def evaluate_question(self, question: str, reference: str) -> Dict[str, Any]:
        """评估单条问答"""
        # 检索
        contexts = self.retrieve(question)
        # 生成
        answer = self.generate(question, contexts)
        return {
            "question": question,
            "reference": reference,
            "answer": answer,
            "contexts": contexts,
        }


# ============================================================
# 3. RAGAS 评估
# ============================================================

def run_ragas_evaluation(results: List[Dict[str, Any]]):
    """使用 RAGAS 评估检索和生成质量"""
    # ragas 0.3.x 仍从 langchain_community.chat_models.vertexai 导入
    # 但新版 langchain-community 已移除该模块，这里做兼容
    import sys as _sys
    if "langchain_community.chat_models.vertexai" not in _sys.modules:
        try:
            from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI
            import types as _types
            _vertexai_module = _types.ModuleType("langchain_community.chat_models.vertexai")
            _vertexai_module.ChatVertexAI = _ChatVertexAI
            _sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_module
        except ImportError:
            pass

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        LLMContextPrecisionWithoutReference,
        context_recall,
    )

    # 准备数据集
    data = {
        "user_input": [r["question"] for r in results],
        "response": [r["answer"] for r in results],
        "retrieved_contexts": [r["contexts"] for r in results],
        "reference": [r["reference"] for r in results],
    }
    dataset = Dataset.from_dict(data)

    log_info(f"  📊 数据集大小: {len(dataset)} 条")

    # 配置 LLM 评估器
    from langchain_openai import ChatOpenAI
    eval_llm = ChatOpenAI(
        model=settings.LLM_MODEL_ID,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        temperature=0.0,
    )

    from ragas.llms import LangchainLLMWrapper
    langchain_llm = LangchainLLMWrapper(eval_llm)

    # 配置 Embedding 评估器
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_core.embeddings import Embeddings as LangchainEmbeddings

    class DashScopeRagasEmbeddings(LangchainEmbeddings):
        """适配 RAGAS 的 Embedding 接口"""
        def embed_documents(self, texts):
            return _dashscope_embed(texts)
        def embed_query(self, text):
            return _dashscope_embed([text])[0]

    ragas_embeddings = LangchainEmbeddingsWrapper(DashScopeRagasEmbeddings())

    # 运行评估
    log_info("  🏃 正在运行 RAGAS 评估...")
    log_info("    指标: faithfulness, answer_relevancy, context_precision, context_recall")

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            LLMContextPrecisionWithoutReference(),
            context_recall,
        ],
        llm=langchain_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
        show_progress=True,
    )

    return result


# ============================================================
# 4. 传统检索指标 (Precision@k, Recall@k, MRR)
# ============================================================

def _extract_keywords(text: str) -> set:
    """提取文本中的关键词（中文词组 + 英文单词）"""
    import re
    # 2字以上的中文连续词组
    chinese = re.findall(r'[一-鿿]{2,}', text)
    # 英文单词
    english = re.findall(r'[a-zA-Z_]{2,}', text.lower())
    # 中文 bigram（单字的重要组合）
    chars = re.findall(r'[一-鿿]', text)
    bigrams = set()
    for i in range(len(chars) - 1):
        bigrams.add(chars[i] + chars[i+1])
    return set(chinese + english) | bigrams


def _compute_chunk_relevance(chunk_text: str, reference: str) -> float:
    """计算一个 chunk 与参考答案的相关性分数

    基于关键词重叠 + 参考文本子串匹配，返回 0.0~1.0 的分数。
    """
    if not reference or not chunk_text:
        return 0.0

    # 1. 参考文本直接在 chunk 中出现（全匹配）
    if reference in chunk_text:
        return 1.0

    # 2. 参考文本中 30% 以上的连续子串出现在 chunk 中
    ref_kws = _extract_keywords(reference)
    chunk_kws = _extract_keywords(chunk_text)

    if not ref_kws:
        return 0.0

    # 关键词重叠率
    overlap = ref_kws & chunk_kws
    jaccard = len(overlap) / len(ref_kws) if ref_kws else 0

    return jaccard


def compute_retrieval_metrics(
    evaluator: RAGEvaluator,
    questions: List[Dict[str, Any]],
    top_k: int = 3,
) -> Dict[str, Any]:
    """计算传统检索指标：Precision@k, Recall@k, MRR

    对每个问题，先标注所有 chunk 中哪些与参考答案相关，
    然后对比检索结果，计算指标。
    """
    import re

    # 获取所有 chunk
    all_chunks = evaluator.kb._chunks
    log_info(f"  📚 总共有 {len(all_chunks)} 个 chunk")

    precisions = []
    recalls = []
    mrrs = []

    for item in questions:
        question = item["question"]
        reference = item["reference"]

        # 1. 标注所有 chunk 的相关性
        relevant_indices = set()
        for i, chunk in enumerate(all_chunks):
            score = _compute_chunk_relevance(chunk.page_content, reference)
            if score >= 0.3:  # 阈值：关键词重叠 30% 以上算相关
                relevant_indices.add(i)

        total_relevant = len(relevant_indices)
        if total_relevant == 0:
            # 没有找到相关 chunk，跳过（不统计）
            continue

        # 2. 检索 top_k
        retrieved = evaluator.kb.search(question, top_k=top_k)

        # 3. 判断每个检索结果是否相关
        relevant_retrieved = 0
        first_relevant_rank = None
        for rank, r in enumerate(retrieved, 1):
            chunk_text = r["text"]
            score = _compute_chunk_relevance(chunk_text, reference)
            if score >= 0.3:
                relevant_retrieved += 1
                if first_relevant_rank is None:
                    first_relevant_rank = rank

        # 4. 计算指标
        precisions.append(relevant_retrieved / top_k)
        recalls.append(relevant_retrieved / total_relevant)
        mrrs.append(1.0 / first_relevant_rank if first_relevant_rank else 0.0)

    # 汇总
    metrics = {
        "precision_at_k": {
            "value": sum(precisions) / len(precisions) if precisions else 0.0,
            "description": f"Precision@{top_k}: 检索结果中相关文档的比例",
        },
        "recall_at_k": {
            "value": sum(recalls) / len(recalls) if recalls else 0.0,
            "description": f"Recall@{top_k}: 检索结果覆盖了多少相关文档",
        },
        "mrr": {
            "value": sum(mrrs) / len(mrrs) if mrrs else 0.0,
            "description": "MRR: 第一个相关文档排在第几位（倒数）",
        },
        "_details": {
            "total_questions_with_ground_truth": len(precisions),
            "per_question": [
                {
                    "question": questions[i]["question"],
                    "category": questions[i]["category"],
                    "precision": precisions[i],
                    "recall": recalls[i],
                    "mrr": mrrs[i],
                }
                for i in range(len(precisions))
            ],
        },
    }

    log_info(f"  📊 Precision@{top_k}: {metrics['precision_at_k']['value']:.4f}")
    log_info(f"  📊 Recall@{top_k}: {metrics['recall_at_k']['value']:.4f}")
    log_info(f"  📊 MRR: {metrics['mrr']['value']:.4f}")

    return metrics


# ============================================================
# 5. 主流程
# ============================================================

def main():
    """主流程"""
    print("=" * 60)
    print("  RAG 系统评估 (RAGAS)")
    print("=" * 60)

    # 初始化评估器
    print("\n[1/4] 初始化...")
    evaluator = RAGEvaluator()

    # 按类别统计
    categories = {}
    for item in TEST_QUESTIONS:
        cat = item["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item)

    print(f"\n[2/4] 测试集: {len(TEST_QUESTIONS)} 条问题, {len(categories)} 个类别")
    for cat, items in categories.items():
        print(f"   - {cat}: {len(items)} 条")

    # 逐条检索+生成
    print(f"\n[3/4] 正在检索并生成答案...")
    results = []
    for i, item in enumerate(TEST_QUESTIONS):
        print(f"  [{i+1}/{len(TEST_QUESTIONS)}] {item['category']}: {item['question'][:40]}...")
        r = evaluator.evaluate_question(item["question"], item["reference"])
        results.append(r)

    # RAGAS评估
    print(f"\n[4/4] 运行 RAGAS 评估 + 传统检索指标...")
    try:
        ragas_result = run_ragas_evaluation(results)

        # 输出结果 - 使用 to_pandas()
        df = ragas_result.to_pandas()

        # 只取数值类型的列（评分列）
        score_cols = [col for col in df.columns if df[col].dtype in ("float64", "int64")]

        print("\n" + "=" * 60)
        print("  📊 RAGAS 评估结果")
        print("=" * 60)

        scores = {}
        for col in score_cols:
            avg = df[col].mean()
            scores[col] = avg
            print(f"  {col:40s}: {avg:.4f}")

        # 传统检索指标
        print("\n" + "=" * 60)
        print("  📊 传统检索指标 (Precision@k, Recall@k, MRR)")
        print("=" * 60)
        retrieval_metrics = compute_retrieval_metrics(evaluator, TEST_QUESTIONS, top_k=3)
        for key in ["precision_at_k", "recall_at_k", "mrr"]:
            m = retrieval_metrics[key]
            print(f"  {key:40s}: {m['value']:.4f}")

        # 按类别分析
        print("\n  --- 按类别分析 (RAGAS) ---")
        cat_map = {item["question"]: item["category"] for item in TEST_QUESTIONS}
        df_with_cat = df.copy()
        df_with_cat["category"] = [cat_map.get(q, "未知") for q in df["user_input"]]
        for cat in sorted(df_with_cat["category"].unique()):
            cat_df = df_with_cat[df_with_cat["category"] == cat]
            print(f"  [{cat}]")
            for col in score_cols:
                val = cat_df[col].mean()
                print(f"    {col}: {val:.4f}")

        print("\n  --- 按类别分析 (传统检索指标) ---")
        for cat in sorted(set(cat_map.values())):
            cat_questions = [q for q in cat_map if cat_map[q] == cat]
            cat_scores = {"precision": [], "recall": [], "mrr": []}
            for detail in retrieval_metrics["_details"]["per_question"]:
                if detail["category"] == cat:
                    cat_scores["precision"].append(detail["precision"])
                    cat_scores["recall"].append(detail["recall"])
                    cat_scores["mrr"].append(detail["mrr"])
            if cat_scores["precision"]:
                print(f"  [{cat}]")
                print(f"    precision@3: {sum(cat_scores['precision']) / len(cat_scores['precision']):.4f}")
                print(f"    recall@3:    {sum(cat_scores['recall']) / len(cat_scores['recall']):.4f}")
                print(f"    mrr:         {sum(cat_scores['mrr']) / len(cat_scores['mrr']):.4f}")

        # 保存结果
        output = {
            "scores": {k: float(v) for k, v in scores.items()},
            "retrieval_metrics": {
                "precision_at_3": retrieval_metrics["precision_at_k"]["value"],
                "recall_at_3": retrieval_metrics["recall_at_k"]["value"],
                "mrr": retrieval_metrics["mrr"]["value"],
            },
            "per_category": {},
            "retrieval_per_category": {},
            "details": [
                {
                    "question": r["question"],
                    "category": cat_map.get(r["question"], "未知"),
                    "answer": r["answer"],
                    "reference": r["reference"],
                    "contexts_count": len(r["contexts"]),
                }
                for r in results
            ],
        }

        # 按类别统计 (RAGAS)
        for cat in sorted(cat_map.values()):
            cat_questions = [q for q in cat_map if cat_map[q] == cat]
            cat_scores = {}
            for col in score_cols:
                mask = df["user_input"].isin(cat_questions)
                cat_scores[col] = float(df[mask][col].mean())
            output["per_category"][cat] = cat_scores

        # 按类别统计 (传统检索指标)
        for cat in sorted(set(cat_map.values())):
            cat_questions = [q for q in cat_map if cat_map[q] == cat]
            cat_scores = {"precision": [], "recall": [], "mrr": []}
            for detail in retrieval_metrics["_details"]["per_question"]:
                if detail["category"] == cat:
                    cat_scores["precision"].append(detail["precision"])
                    cat_scores["recall"].append(detail["recall"])
                    cat_scores["mrr"].append(detail["mrr"])
            if cat_scores["precision"]:
                output["retrieval_per_category"][cat] = {
                    "precision_at_3": sum(cat_scores["precision"]) / len(cat_scores["precision"]),
                    "recall_at_3": sum(cat_scores["recall"]) / len(cat_scores["recall"]),
                    "mrr": sum(cat_scores["mrr"]) / len(cat_scores["mrr"]),
                }

        output_path = os.path.join(
            os.path.dirname(__file__),
            "..", "evaluation_results.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n  💾 详细结果已保存: {output_path}")

    except Exception as e:
        log_error(f"  ❌ RAGAS 评估失败: {e}")
        import traceback
        traceback.print_exc()

        # 降级：保存原始结果供人工分析
        fallback_path = os.path.join(
            os.path.dirname(__file__),
            "..", "evaluation_raw_results.json"
        )
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump({
                "total": len(results),
                "results": [
                    {
                        "question": r["question"],
                        "answer": r["answer"][:500],
                        "reference": r["reference"],
                        "contexts_count": len(r["contexts"]),
                        "contexts": [c[:200] for c in r["contexts"]],
                    }
                    for r in results
                ],
            }, f, ensure_ascii=False, indent=2)
        print(f"  💾 原始结果已保存: {fallback_path}")


if __name__ == "__main__":
    main()