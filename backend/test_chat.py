#!/usr/bin/env python
"""快速测试工具 - 直接调 Agent，不用启动服务器

用法:
    python test_chat.py "帮我设计一个登录页面"
    python test_chat.py --agent "帮我设计开发一个登录页面"          # 测试多Agent
    python test_chat.py --single "你好"          # 测试单NPC对话
    python test_chat.py --list                   # 查看NPC列表
"""

import sys
import os
import json
import argparse

# 修复代理和编码
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def test_agent_chat(message: str):
    """测试多Agent协作"""
    from agent_workflow.workflow import get_workflow

    print(f"\n{'='*60}")
    print(f"  🎯 测试多Agent协作")
    print(f"  💬 用户: {message}")
    print(f"{'='*60}\n")

    wf = get_workflow()

    result = wf.invoke({
        "messages": [],
        "user_input": message,
        "active_agent": "",
        "task_analysis": None,
        "shared_context": {},
        "tool_results": [],
        "rag_context": None,
        "final_response": None,
        "agent_history": [],
    }, config={"configurable": {"thread_id": "test_cli"}})

    print(f"\n{'='*60}")
    print(f"  📋 各角色输出")
    print(f"{'='*60}\n")

    for h in result.get("agent_history", []):
        agent = h.get("agent", "?")
        npc = h.get("npc_name", "?")
        output = h.get("output", "")

        if agent == "dev_tool":
            print(f"  🔧 [{npc}] 工具调用:")
            print(f"       {output[:200]}")
            print()
        elif agent == "pm":
            print(f"  📋 [{agent}] {npc}:")
            print(f"  {output}")
            print()
        elif agent == "designer":
            print(f"  🎨 [{agent}] {npc}:")
            print(f"  {output}")
            print()
        elif agent == "dev":
            print(f"  💻 [{agent}] {npc}:")
            print(f"  {output}")
            print()
        else:
            print(f"  [{agent}] {npc}:")
            print(f"  {output[:500]}")
            print()

    final = result.get("final_response", "")
    if final:
        print(f"{'='*60}")
        print(f"  📝 最终回复:")
        print(f"{'='*60}")
        print(f"  {final}")
        print()


def test_single_chat(message: str):
    """测试单NPC对话"""
    from agents import get_npc_manager

    print(f"\n{'='*60}")
    print(f"  🎯 测试单NPC对话")
    print(f"  💬 用户: {message}")
    print(f"{'='*60}\n")

    mgr = get_npc_manager()

    for npc_name in ["需求多", "技术多", "设计多"]:
        print(f"  ── {npc_name} ──")
        try:
            response = mgr.chat(npc_name, message)
            print(f"  {response[:200]}")
        except Exception as e:
            print(f"  ❌ 错误: {e}")
        print()


def list_npcs():
    """查看NPC列表"""
    from agents import get_npc_manager, NPC_NAMES, NPC_ROLES

    mgr = get_npc_manager()

    print(f"\n{'='*60}")
    print(f"  📋 NPC列表")
    print(f"{'='*60}\n")

    for key, name in NPC_NAMES.items():
        role = NPC_ROLES[name]
        print(f"  {name:6s} | {role['title']:8s} | {role['location']} | {role['activity']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="数字办公室测试工具")
    parser.add_argument("message", nargs="?", default="", help="测试消息")
    parser.add_argument("--agent", action="store_true", help="测试多Agent协作")
    parser.add_argument("--single", action="store_true", help="测试单NPC对话")
    parser.add_argument("--list", action="store_true", help="查看NPC列表")
    parser.add_argument("--stream", action="store_true", help="测试流式响应")

    args = parser.parse_args()

    if args.list:
        list_npcs()
        return

    if args.single or (args.message and not args.agent and not args.stream):
        test_single_chat(args.message)
        return

    if args.stream and args.message:
        test_stream_chat(args.message)
        return

    if args.agent or (args.message and not args.single and not args.stream):
        test_agent_chat(args.message)
        return

    parser.print_help()


def test_stream_chat(message: str):
    """测试流式对话"""
    import requests as req

    print(f"\n{'='*60}")
    print(f"  🎯 测试流式对话")
    print(f"  💬 用户: {message}")
    print(f"{'='*60}\n")

    try:
        resp = req.post(
            "http://localhost:8000/chat/stream",
            json={"npc_name": "需求多", "message": message},
            stream=True,
            timeout=30,
        )

        print(f"  回复: ", end="", flush=True)
        for line in resp.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    import json
                    try:
                        d = json.loads(data)
                        if d.get("type") == "chunk":
                            print(d["text"], end="", flush=True)
                        elif d.get("type") == "done":
                            print()
                    except json.JSONDecodeError:
                        pass
        print(f"\n  ✅ 流式响应完成")
    except Exception as e:
        print(f"\n  ❌ 错误: {e}")


if __name__ == "__main__":
    main()