"""知识库分块可视化 - 生成 HTML 报告

用法:
    python visualize_chunks.py

输出:
    chunks_visual.html - 浏览器打开即可查看
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
os.environ['LLM_API_KEY'] = 'test'
os.environ['LLM_BASE_URL'] = 'http://localhost'
os.environ['EMBEDDING_MODEL'] = 'text-embedding-v1'
os.environ['API_PORT'] = '8000'
os.environ['API_HOST'] = '0.0.0.0'

from knowledge_base.rag import load_design_files, chunk_documents
from pathlib import Path

# 配置（使用 chunk_documents 的默认值）
from knowledge_base.rag import chunk_documents as _chunk_docs
import inspect
_sig = inspect.signature(_chunk_docs)
CHUNK_SIZE = _sig.parameters['chunk_size'].default   # 300
CHUNK_OVERLAP = _sig.parameters['chunk_overlap'].default  # 60

# 颜色映射 - 每个文档一个颜色
DOC_COLORS = {
    "introduce": "#E8F5E9",
    "philosophy": "#FFF3E0",
    "style-guideline": "#E3F2FD",
    "values": "#F3E5F5",
}
DOC_BORDERS = {
    "introduce": "#4CAF50",
    "philosophy": "#FF9800",
    "style-guideline": "#2196F3",
    "values": "#9C27B0",
}

# 加载文档
print("📚 加载文档...")
docs = load_design_files("backend/knowledge_base/design")

# 分块
print(f"✂️  分块 (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
chunks = chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# 统计
doc_stats = {}
for d in docs:
    title = d.metadata["title"]
    doc_stats[title] = {
        "total_chars": len(d.page_content),
        "chunk_count": 0,
        "chunks": [],
    }

for i, chunk in enumerate(chunks):
    title = chunk.metadata["title"]
    doc_stats[title]["chunk_count"] += 1
    doc_stats[title]["chunks"].append({
        "index": i + 1,
        "content": chunk.page_content,
        "length": len(chunk.page_content),
    })

# 生成 HTML
html_parts = []
html_parts.append("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>知识库分块可视化</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif;
         background: #f5f5f5; padding: 30px; color: #333; }
  h1 { font-size: 24px; margin-bottom: 8px; }
  .subtitle { color: #666; margin-bottom: 24px; font-size: 14px; }

  /* 统计卡片 */
  .stats { display: flex; gap: 12px; margin-bottom: 30px; flex-wrap: wrap; }
  .stat-card { flex: 1; min-width: 180px; background: white; border-radius: 10px;
               padding: 16px 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .stat-card .label { font-size: 12px; color: #888; }
  .stat-card .value { font-size: 28px; font-weight: 700; margin-top: 4px; }

  /* 文档摘要 */
  .doc-summary { display: flex; gap: 12px; margin-bottom: 30px; flex-wrap: wrap; }
  .doc-card { flex: 1; min-width: 200px; border-radius: 10px; padding: 14px 18px;
              box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .doc-card .doc-title { font-size: 15px; font-weight: 600; margin-bottom: 6px; }
  .doc-card .doc-stat { font-size: 12px; color: #555; line-height: 1.6; }

  /* 搜索过滤 */
  .toolbar { margin-bottom: 20px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .toolbar input { padding: 8px 14px; border: 1px solid #ddd; border-radius: 6px;
                   font-size: 14px; width: 240px; }
  .toolbar input:focus { outline: none; border-color: #2196F3; }
  .filter-btn { padding: 6px 14px; border-radius: 20px; border: 1px solid #ddd;
                background: white; cursor: pointer; font-size: 13px; transition: all 0.2s; }
  .filter-btn:hover { background: #f0f0f0; }
  .filter-btn.active { background: #2196F3; color: white; border-color: #2196F3; }

  /* 分块列表 */
  .chunk { margin-bottom: 12px; border-radius: 10px; overflow: hidden;
           box-shadow: 0 1px 4px rgba(0,0,0,0.08); transition: box-shadow 0.2s; }
  .chunk:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.12); }
  .chunk-header { display: flex; justify-content: space-between; align-items: center;
                  padding: 10px 16px; font-size: 13px; cursor: pointer;
                  border-bottom: 1px solid rgba(0,0,0,0.06); }
  .chunk-header .chunk-title { font-weight: 600; }
  .chunk-header .chunk-meta { color: #888; font-size: 12px; }
  .chunk-body { padding: 14px 16px; font-size: 14px; line-height: 1.8;
                white-space: pre-wrap; display: none; }
  .chunk-body.open { display: block; }

  /* 关键词高亮 */
  .highlight { background: #FFEB3B; padding: 0 2px; border-radius: 2px; }

  /* 重叠提示 */
  .overlap-badge { display: inline-block; font-size: 11px; padding: 1px 8px;
                   border-radius: 10px; margin-left: 8px; }
  .overlap-badge.yes { background: #E8F5E9; color: #2E7D32; }
  .overlap-badge.no { background: #FFEBEE; color: #C62828; }
  .overlap-highlight { background: #FFF9C4; }

  /* 进度条 */
  .progress-bar { height: 4px; border-radius: 2px; background: #eee;
                  margin-top: 6px; overflow: hidden; }
  .progress-bar .fill { height: 100%; border-radius: 2px; transition: width 0.3s; }

  @media (max-width: 600px) {
    body { padding: 16px; }
    .stats { flex-direction: column; }
    .doc-summary { flex-direction: column; }
  }
</style>
</head>
<body>
""")

html_parts.append(f"""
<h1>📄 知识库分块可视化</h1>
<p class="subtitle">分块参数: chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP} | 共 {len(chunks)} 个块</p>

<div class="stats">
  <div class="stat-card">
    <div class="label">文档总数</div>
    <div class="value">{len(docs)}</div>
  </div>
  <div class="stat-card">
    <div class="label">分块总数</div>
    <div class="value">{len(chunks)}</div>
  </div>
  <div class="stat-card">
    <div class="label">总字符数</div>
    <div class="value">{sum(len(d.page_content) for d in docs):,}</div>
  </div>
  <div class="stat-card">
    <div class="label">平均块大小</div>
    <div class="value">{sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)}</div>
  </div>
</div>

<div class="doc-summary">
""")

for title, stats in doc_stats.items():
    color = DOC_COLORS.get(title, "#f5f5f5")
    border = DOC_BORDERS.get(title, "#ccc")
    chunk_count = stats["chunk_count"]
    total_chars = stats["total_chars"]
    html_parts.append(f"""
  <div class="doc-card" style="background:{color}; border-left:4px solid {border};">
    <div class="doc-title">{title}</div>
    <div class="doc-stat">{total_chars} 字 → {chunk_count} 个块</div>
    <div class="progress-bar">
      <div class="fill" style="width:{chunk_count / max(len(chunks), 1) * 100}%;background:{border};"></div>
    </div>
  </div>
""")

html_parts.append("</div>")

# 工具栏
html_parts.append("""
<div class="toolbar">
  <input type="text" id="searchInput" placeholder="🔍 搜索关键词..." oninput="filterChunks()">
  <span style="color:#888;font-size:13px;">文档筛选:</span>
""")
for title in doc_stats:
    color = DOC_BORDERS.get(title, "#999")
    html_parts.append(f'  <button class="filter-btn" data-doc="{title}" style="border-color:{color};" onclick="toggleDocFilter(this)">{title}</button>')
html_parts.append("""</div>

<div id="chunkList">
""")

# 分块列表
for i, chunk in enumerate(chunks):
    title = chunk.metadata["title"]
    content = chunk.page_content
    length = len(content)
    color = DOC_COLORS.get(title, "#f5f5f5")
    border = DOC_BORDERS.get(title, "#ccc")

    # 检测是否包含重叠内容（以前一块末尾匹配）
    overlap_hint = ""
    overlap_badge = ""
    if i > 0 and chunks[i - 1].metadata["title"] == title:
        prev_content = chunks[i - 1].page_content
        tail = prev_content[-CHUNK_OVERLAP:].strip()
        if tail and tail in content[:CHUNK_OVERLAP * 2]:
            overlap_hint = " ✅ 含重叠"
            overlap_badge = "<span class='overlap-badge yes'>✅ 重叠</span>"
        else:
            overlap_badge = "<span class='overlap-badge no'>❌ 无重叠</span>"

    # 安全转义 HTML
    content_html = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 在内容中标记重叠区域（用黄色高亮）
    if overlap_hint and i > 0 and chunks[i - 1].metadata["title"] == title:
        prev_content = chunks[i - 1].page_content
        tail = prev_content[-CHUNK_OVERLAP:].strip()
        if tail and tail in content[:CHUNK_OVERLAP * 2]:
            # 用 span 包裹重叠部分（仅开头第一次出现）
            overlap_escaped = tail.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if overlap_escaped in content_html[:CHUNK_OVERLAP * 2]:
                content_html = content_html.replace(
                    overlap_escaped,
                    "<span class='overlap-highlight'>" + overlap_escaped + "</span>",
                    1
                )

    html_parts.append(f"""
  <div class="chunk" data-doc="{title}">
    <div class="chunk-header" style="background:{color};" onclick="toggleChunk(this)">
      <span class="chunk-title">#{i+1} {title}</span>
      <span class="chunk-meta">{length} 字{overlap_hint} {overlap_badge}</span>
    </div>
    <div class="chunk-body">{content_html}</div>
  </div>
""")

html_parts.append("""
</div>

<script>
function toggleChunk(header) {
  var body = header.nextElementSibling;
  body.classList.toggle('open');
}

function filterChunks() {
  var query = document.getElementById('searchInput').value.toLowerCase();
  var chunks = document.querySelectorAll('.chunk');
  chunks.forEach(function(chunk) {
    var text = chunk.textContent.toLowerCase();
    chunk.style.display = text.includes(query) ? '' : 'none';
  });
}

function toggleDocFilter(btn) {
  btn.classList.toggle('active');
  applyFilters();
}

function applyFilters() {
  var activeDocs = Array.from(document.querySelectorAll('.filter-btn.active')).map(function(b) { return b.dataset.doc; });
  var allChunks = document.querySelectorAll('.chunk');
  allChunks.forEach(function(chunk) {
    if (activeDocs.length === 0) {
      chunk.style.display = '';
    } else {
      chunk.style.display = activeDocs.includes(chunk.dataset.doc) ? '' : 'none';
    }
  });
}

// 默认展开所有
document.querySelectorAll('.chunk-body').forEach(function(b) { b.classList.add('open'); });
</script>
</body>
</html>
""")

# 写入文件
output_path = Path("backend/chunks_visual.html")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("".join(html_parts))

print(f"\n✅ 可视化报告已生成: {output_path.resolve()}")
print(f"   浏览器打开即可查看")