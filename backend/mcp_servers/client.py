"""MCP 客户端包装器

让 Agent 节点能够直接调用 MCP 工具，
不需要通过标准输入输出。
"""

from typing import Dict, Any, Optional

from mcp_servers.code_mcp import (
    read_file, write_file, search_code,
    list_files, run_command
)


class CodeMCPClient:
    """代码 MCP 工具客户端

    包装 MCP Server 的工具函数，供 Agent 调用。
    每个工具返回字符串结果，方便 LLM 理解。
    """

    @staticmethod
    def call_tool(tool_name: str, **kwargs) -> str:
        """调用 MCP 工具

        Args:
            tool_name: 工具名 (read_file, write_file, search_code, list_files, run_command)
            **kwargs: 工具参数

        Returns:
            工具执行结果 (字符串)
        """
        tool_map = {
            "read_file": read_file,
            "write_file": write_file,
            "search_code": search_code,
            "list_files": list_files,
            "run_command": run_command,
        }

        func = tool_map.get(tool_name)
        if not func:
            return f"错误: 未知工具 '{tool_name}'，可用工具: {', '.join(tool_map.keys())}"

        try:
            return func(**kwargs)
        except Exception as e:
            return f"错误: 工具 '{tool_name}' 执行失败 - {str(e)}"

    @staticmethod
    def get_tool_descriptions() -> str:
        """获取工具描述，供 Agent 参考"""
        return """可用工具:
1. read_file(filepath, max_length=5000) - 读取文件内容
2. write_file(filepath, content) - 写入/创建文件
3. search_code(keyword, file_pattern="*.py") - 搜索代码
4. list_files(directory=".", pattern="*") - 列出文件
5. run_command(command, args="") - 运行命令 (python/test/lint/format)"""