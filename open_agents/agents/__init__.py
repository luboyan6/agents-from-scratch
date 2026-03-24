"""
Agent实现模块 - OpenAgents原生Agent范式

该模块导出了多种 Agent 实现，每种实现代表不同的智能体范式：

导出内容:
    - SimpleAgent: 简单对话 Agent，支持可选的工具调用
    - FunctionCallAgent: 基于 OpenAI 原生函数调用的 Agent
    - ReActAgent: 推理与行动结合的 Agent
    - ReflectionAgent: 自我反思与迭代优化的 Agent
    - PlanAndSolveAgent: 分解规划与逐步执行的 Agent
    - ToolAwareSimpleAgent: 带工具调用监听的 SimpleAgent
"""

from .simple_agent import SimpleAgent
from .function_call_agent import FunctionCallAgent
from .react_agent import ReActAgent
from .reflection_agent import ReflectionAgent
from .plan_solve_agent import PlanAndSolveAgent
from .tool_aware_agent import ToolAwareSimpleAgent

# ==================== 模块公共 API ====================

__all__ = [
    "SimpleAgent",
    "FunctionCallAgent",
    "ReActAgent",
    "ReflectionAgent",
    "PlanAndSolveAgent",
    "ToolAwareSimpleAgent"
]
