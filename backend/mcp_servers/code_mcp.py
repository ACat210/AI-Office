"""Code MCP Server - 技术多的代码工具

提供文件读写、代码搜索、命令执行等能力，
让 Agent 能通过 MCP 协议操作真实代码。
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Optional

from mcp.server.mcpserver import MCPServer

# 创建 MCP Server
mcp = MCPServer("code-mcp", instructions="代码操作工具集：读文件、写文件、搜索代码、运行命令")

# 安全限制：只允许操作这些目录
ALLOWED_DIRS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),  # backend/
]

# 允许执行的命令白名单
ALLOWED_COMMANDS = {
    "python": ["python", "python3"],
    "test": ["pytest"],
    "lint": ["flake8", "ruff"],
    "format": ["black"],
    "shell": ["ls", "cat", "head", "tail", "wc", "find", "grep"],
}


def _is_path_allowed(filepath: str) -> bool:
    """检查路径是否在允许范围内"""
    abs_path = os.path.abspath(filepath)
    for allowed in ALLOWED_DIRS:
        if abs_path.startswith(allowed):
            return True
    return False


def _safe_path(filepath: str) -> Optional[str]:
    """安全解析路径，返回绝对路径或None"""
    # 如果是相对路径，相对于 backend/ 目录
    if not os.path.isabs(filepath):
        base = ALLOWED_DIRS[0]
        filepath = os.path.join(base, filepath)
    filepath = os.path.abspath(filepath)
    if _is_path_allowed(filepath):
        return filepath
    return None


@mcp.tool()
def read_file(filepath: str, max_length: int = 5000) -> str:
    """读取文件内容

    Args:
        filepath: 文件路径 (相对或绝对)
        max_length: 最大读取字符数，默认5000
    """
    safe = _safe_path(filepath)
    if not safe:
        return f"错误: 路径 '{filepath}' 不在允许范围内"

    if not os.path.exists(safe):
        return f"错误: 文件 '{filepath}' 不存在"

    if not os.path.isfile(safe):
        return f"错误: '{filepath}' 不是文件"

    try:
        with open(safe, "r", encoding="utf-8") as f:
            content = f.read(max_length)
        if len(content) >= max_length:
            content += f"\n\n...(文件过长，仅显示前 {max_length} 字符)"
        return content
    except Exception as e:
        return f"错误: 读取文件失败 - {str(e)}"


@mcp.tool()
def write_file(filepath: str, content: str) -> str:
    """写入/创建文件

    Args:
        filepath: 文件路径 (相对或绝对)
        content: 文件内容
    """
    safe = _safe_path(filepath)
    if not safe:
        return f"错误: 路径 '{filepath}' 不在允许范围内"

    try:
        os.makedirs(os.path.dirname(safe), exist_ok=True)
        with open(safe, "w", encoding="utf-8") as f:
            f.write(content)
        return f"成功: 已写入文件 '{filepath}' ({len(content)} 字符)"
    except Exception as e:
        return f"错误: 写入文件失败 - {str(e)}"


@mcp.tool()
def search_code(keyword: str, file_pattern: str = "*.py") -> str:
    """搜索代码中的关键词

    Args:
        keyword: 要搜索的关键词（支持正则表达式）
        file_pattern: 文件匹配模式，默认 *.py
    """
    import glob
    import re

    safe_dir = ALLOWED_DIRS[0]
    try:
        # 递归搜索匹配的文件
        full_pattern = os.path.join(safe_dir, "**", file_pattern)
        files = glob.glob(full_pattern, recursive=True)

        # 排除 __pycache__ 和 .venv
        files = [f for f in files if "__pycache__" not in f and ".venv" not in f]

        if not files:
            return f"未找到匹配 '{file_pattern}' 的文件"

        results = []
        max_results = 50
        for filepath in sorted(files):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, 1):
                        if re.search(keyword, line):
                            rel = os.path.relpath(filepath, safe_dir)
                            results.append(f"{rel}:{line_no}:{line.rstrip()[:200]}")
                            if len(results) >= max_results:
                                break
            except (UnicodeDecodeError, IOError):
                continue
            if len(results) >= max_results:
                break

        if not results:
            return f"未找到匹配 '{keyword}' 的结果"
        if len(results) >= max_results:
            results.append(f"... (共超过 {max_results} 行，仅显示前 {max_results} 行)")
        return "\n".join(results)
    except Exception as e:
        return f"错误: 搜索失败 - {str(e)}"


@mcp.tool()
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """列出目录中的文件

    Args:
        directory: 目录路径 (相对或绝对)
        pattern: 文件匹配模式，默认显示所有文件
    """
    safe = _safe_path(directory)
    if not safe:
        return f"错误: 路径 '{directory}' 不在允许范围内"

    if not os.path.exists(safe):
        return f"错误: 目录 '{directory}' 不存在"

    if not os.path.isdir(safe):
        return f"错误: '{directory}' 不是目录"

    try:
        import glob
        full_pattern = os.path.join(safe, pattern)
        files = glob.glob(full_pattern)
        files = [f for f in files if not os.path.basename(f).startswith("__pycache__")]

        if not files:
            return f"目录 '{directory}' 中没有匹配 '{pattern}' 的文件"

        lines = []
        for f in sorted(files):
            name = os.path.relpath(f, ALLOWED_DIRS[0]) if f.startswith(ALLOWED_DIRS[0]) else f
            if os.path.isdir(f):
                lines.append(f"  📁 {name}/")
            else:
                size = os.path.getsize(f)
                lines.append(f"  📄 {name} ({size} bytes)")

        return "\n".join(lines[:100])
    except Exception as e:
        return f"错误: 列出目录失败 - {str(e)}"


@mcp.tool()
def run_command(command: str, args: str = "") -> str:
    """运行命令 (仅限白名单内的命令)

    Args:
        command: 命令类型 (python, test, lint, format)
        args: 命令参数
    """
    if command not in ALLOWED_COMMANDS:
        allowed = ", ".join(ALLOWED_COMMANDS.keys())
        return f"错误: 不允许的命令 '{command}'。允许的命令: {allowed}"

    cmd_list = [ALLOWED_COMMANDS[command][0]] + args.split()
    safe_dir = ALLOWED_DIRS[0]

    try:
        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=safe_dir,
        )
        output = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr[-1000:]}"
        return output if output else "(无输出)"
    except subprocess.TimeoutExpired:
        return "错误: 命令执行超时 (30秒)"
    except FileNotFoundError:
        return f"错误: 命令 '{ALLOWED_COMMANDS[command][0]}' 未找到，请确认已安装"
    except Exception as e:
        return f"错误: 命令执行失败 - {str(e)}"


def get_code_mcp() -> MCPServer:
    """获取 Code MCP Server 实例"""
    return mcp