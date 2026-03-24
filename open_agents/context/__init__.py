"""
上下文工程模块

为 OpenAgents 框架提供上下文工程能力：
- ContextBuilder: GSSC 流水线（Gather-Select-Structure-Compress）
- Compactor: 对话压缩整合
- NotesManager: 结构化笔记管理
- ContextObserver: 可观测性与指标追踪

GSSC 流水线说明:
1. Gather: 从多源收集候选信息（历史、记忆、RAG、工具结果）
2. Select: 基于优先级、相关性、多样性筛选
3. Structure: 组织成结构化上下文模板
4. Compress: 在预算内压缩与规范化
"""

from .builder import ContextBuilder, ContextConfig, ContextPacket

# ==================== 模块公共 API ====================

__all__ = [
    "ContextBuilder",
    "ContextConfig",
    "ContextPacket",
]
