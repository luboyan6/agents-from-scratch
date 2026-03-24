"""
ContextBuilder - GSSC 流水线实现

实现 Gather-Select-Structure-Compress 上下文构建流程：
1. Gather: 从多源收集候选信息（历史、记忆、RAG、工具结果）
2. Select: 基于优先级、相关性、多样性筛选
3. Structure: 组织成结构化上下文模板
4. Compress: 在预算内压缩与规范化

该模块提供智能上下文管理能力，帮助 Agent 在有限的 token 预算内
构建最相关的上下文，提高响应质量。
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import tiktoken
import math

from ..core.message import Message
from ..tools import MemoryTool, RAGTool


# ==================== 数据类 ====================

@dataclass
class ContextPacket:
    """
    上下文信息包

    用于封装单个上下文信息单元，包含内容、元数据和评分信息。

    属性:
        content: 上下文内容文本
        timestamp: 创建时间戳
        metadata: 元数据字典（如类型、重要性等）
        token_count: token 数量
        relevance_score: 相关性分数（0.0-1.0）
    """

    # ==================== 成员变量 ====================
    content: str                                               # 上下文内容
    timestamp: datetime = field(default_factory=datetime.now) # 创建时间
    metadata: Dict[str, Any] = field(default_factory=dict)    # 元数据
    token_count: int = 0                                       # token 数量
    relevance_score: float = 0.0                               # 相关性分数

    # ==================== 魔法方法 ====================

    def __post_init__(self):
        """初始化后自动计算 token 数"""
        if self.token_count == 0:
            self.token_count = count_tokens(self.content)


@dataclass
class ContextConfig:
    """
    上下文构建配置

    用于配置 ContextBuilder 的行为参数。

    属性:
        max_tokens: 总 token 预算
        reserve_ratio: 生成余量比例（10-20%）
        min_relevance: 最小相关性阈值
        enable_mmr: 是否启用最大边际相关性（多样性）
        mmr_lambda: MMR 平衡参数
        system_prompt_template: 系统提示模板
        enable_compression: 是否启用压缩
    """

    # ==================== 成员变量 ====================
    max_tokens: int = 8000                     # 总 token 预算
    reserve_ratio: float = 0.15                # 生成余量比例
    min_relevance: float = 0.3                 # 最小相关性阈值
    enable_mmr: bool = True                    # 启用 MMR（多样性）
    mmr_lambda: float = 0.7                    # MMR 平衡参数
    system_prompt_template: str = ""           # 系统提示模板
    enable_compression: bool = True            # 启用压缩

    # ==================== 公有方法 ====================

    def get_available_tokens(self) -> int:
        """
        获取可用 token 预算（扣除余量）

        Returns:
            可用的 token 数量
        """
        return int(self.max_tokens * (1 - self.reserve_ratio))


# ==================== 主类 ====================

class ContextBuilder:
    """
    上下文构建器 - GSSC 流水线

    实现四阶段上下文构建流程，在有限预算内构建最优上下文。

    流程:
        1. Gather: 从记忆、RAG、对话历史等多源收集信息
        2. Select: 基于相关性、新近性筛选信息
        3. Structure: 组织成结构化模板
        4. Compress: 压缩超预算内容

    用法示例:
        ```python
        builder = ContextBuilder(
            memory_tool=memory_tool,
            rag_tool=rag_tool,
            config=ContextConfig(max_tokens=8000)
        )

        context = builder.build(
            user_query="用户问题",
            conversation_history=[...],
            system_instructions="系统指令"
        )
        ```

    成员变量:
        memory_tool: 记忆工具实例
        rag_tool: RAG 工具实例
        config: 上下文配置
        _encoding: tiktoken 编码器
    """

    # ==================== 构造方法 ====================

    def __init__(
        self,
        memory_tool: Optional[MemoryTool] = None,
        rag_tool: Optional[RAGTool] = None,
        config: Optional[ContextConfig] = None
    ):
        """
        初始化上下文构建器

        Args:
            memory_tool: 记忆工具实例（可选）
            rag_tool: RAG 工具实例（可选）
            config: 上下文配置（可选，使用默认配置）
        """
        # ==================== 成员变量 ====================
        self.memory_tool = memory_tool                    # 记忆工具
        self.rag_tool = rag_tool                          # RAG 工具
        self.config = config or ContextConfig()           # 配置对象
        self._encoding = tiktoken.get_encoding("cl100k_base")  # tiktoken 编码器

    # ==================== 公有方法 ====================

    def build(
        self,
        user_query: str,
        conversation_history: Optional[List[Message]] = None,
        system_instructions: Optional[str] = None,
        additional_packets: Optional[List[ContextPacket]] = None
    ) -> str:
        """
        构建完整上下文（主入口方法）

        执行完整的 GSSC 流水线：
        1. Gather: 收集候选信息
        2. Select: 筛选与排序
        3. Structure: 组织成结构化模板
        4. Compress: 压缩与规范化

        Args:
            user_query: 用户查询
            conversation_history: 对话历史（可选）
            system_instructions: 系统指令（可选）
            additional_packets: 额外的上下文包（可选）

        Returns:
            结构化上下文字符串
        """
        # 1. Gather: 收集候选信息
        packets = self._gather(
            user_query=user_query,
            conversation_history=conversation_history or [],
            system_instructions=system_instructions,
            additional_packets=additional_packets or []
        )

        # 2. Select: 筛选与排序
        selected_packets = self._select(packets, user_query)

        # 3. Structure: 组织成结构化模板
        structured_context = self._structure(
            selected_packets=selected_packets,
            user_query=user_query,
            system_instructions=system_instructions
        )

        # 4. Compress: 压缩与规范化（如果超预算）
        final_context = self._compress(structured_context)

        return final_context

    # ==================== 私有方法（GSSC 流水线各阶段）====================

    def _gather(
        self,
        user_query: str,
        conversation_history: List[Message],
        system_instructions: Optional[str],
        additional_packets: List[ContextPacket]
    ) -> List[ContextPacket]:
        """
        Gather 阶段：收集候选信息

        从多个来源收集上下文信息：
        - P0: 系统指令（强约束）
        - P1: 记忆中的任务状态与关键结论
        - P2: RAG 中的事实证据
        - P3: 对话历史（辅助材料）

        Args:
            user_query: 用户查询
            conversation_history: 对话历史
            system_instructions: 系统指令
            additional_packets: 额外的上下文包

        Returns:
            收集到的上下文包列表
        """
        packets = []

        # P0: 系统指令（强约束）
        if system_instructions:
            packets.append(ContextPacket(
                content=system_instructions,
                metadata={"type": "instructions"}
            ))

        # P1: 从记忆中获取任务状态与关键结论
        if self.memory_tool:
            try:
                # 搜索任务状态相关记忆
                state_results = self.memory_tool.execute(
                    "search",
                    query="(任务状态 OR 子目标 OR 结论 OR 阻塞)",
                    min_importance=0.7,
                    limit=5
                )
                if state_results and "未找到" not in state_results:
                    packets.append(ContextPacket(
                        content=state_results,
                        metadata={"type": "task_state", "importance": "high"}
                    ))

                # 搜索与当前查询相关的记忆
                related_results = self.memory_tool.execute(
                    "search",
                    query=user_query,
                    limit=5
                )
                if related_results and "未找到" not in related_results:
                    packets.append(ContextPacket(
                        content=related_results,
                        metadata={"type": "related_memory"}
                    ))
            except Exception as e:
                print(f"⚠️ 记忆检索失败: {e}")

        # P2: 从 RAG 中获取事实证据
        if self.rag_tool:
            try:
                rag_results = self.rag_tool.run({
                    "action": "search",
                    "query": user_query,
                    "limit": 5
                })
                if rag_results and "未找到" not in rag_results and "错误" not in rag_results:
                    packets.append(ContextPacket(
                        content=rag_results,
                        metadata={"type": "knowledge_base"}
                    ))
            except Exception as e:
                print(f"⚠️ RAG检索失败: {e}")

        # P3: 对话历史（辅助材料）
        if conversation_history:
            # 只保留最近 N 条
            recent_history = conversation_history[-10:]
            history_text = "\n".join([
                f"[{msg.role}] {msg.content}"
                for msg in recent_history
            ])
            packets.append(ContextPacket(
                content=history_text,
                metadata={"type": "history", "count": len(recent_history)}
            ))

        # 添加额外包
        packets.extend(additional_packets)

        return packets

    def _select(
        self,
        packets: List[ContextPacket],
        user_query: str
    ) -> List[ContextPacket]:
        """
        Select 阶段：基于分数与预算的筛选

        筛选步骤:
        1. 计算相关性分数（关键词重叠）
        2. 计算新近性分数（指数衰减）
        3. 计算复合分数
        4. 按预算填充

        Args:
            packets: 候选上下文包列表
            user_query: 用户查询

        Returns:
            筛选后的上下文包列表
        """
        # 1) 计算相关性（关键词重叠）
        query_tokens = set(user_query.lower().split())
        for packet in packets:
            content_tokens = set(packet.content.lower().split())
            if len(query_tokens) > 0:
                overlap = len(query_tokens & content_tokens)
                packet.relevance_score = overlap / len(query_tokens)
            else:
                packet.relevance_score = 0.0

        # 2) 计算新近性（指数衰减）
        def recency_score(ts: datetime) -> float:
            delta = max((datetime.now() - ts).total_seconds(), 0)
            tau = 3600  # 1小时时间尺度
            return math.exp(-delta / tau)

        # 3) 计算复合分：0.7*相关性 + 0.3*新近性
        scored_packets: List[Tuple[float, ContextPacket]] = []
        for p in packets:
            rec = recency_score(p.timestamp)
            score = 0.7 * p.relevance_score + 0.3 * rec
            scored_packets.append((score, p))

        # 4) 系统指令单独拿出，固定纳入
        system_packets = [p for (_, p) in scored_packets if p.metadata.get("type") == "instructions"]
        remaining = [p for (s, p) in sorted(scored_packets, key=lambda x: x[0], reverse=True)
                     if p.metadata.get("type") != "instructions"]

        # 5) 依据 min_relevance 过滤（对非系统包）
        filtered = [p for p in remaining if p.relevance_score >= self.config.min_relevance]

        # 6) 按预算填充
        available_tokens = self.config.get_available_tokens()
        selected: List[ContextPacket] = []
        used_tokens = 0

        # 先放入系统指令（不排序）
        for p in system_packets:
            if used_tokens + p.token_count <= available_tokens:
                selected.append(p)
                used_tokens += p.token_count

        # 再按分数加入其余
        for p in filtered:
            if used_tokens + p.token_count > available_tokens:
                continue
            selected.append(p)
            used_tokens += p.token_count

        return selected

    def _structure(
        self,
        selected_packets: List[ContextPacket],
        user_query: str,
        system_instructions: Optional[str]
    ) -> str:
        """
        Structure 阶段：组织成结构化上下文模板

        将筛选后的上下文包组织成结构化格式：
        - [Role & Policies]: 系统指令
        - [Task]: 当前任务
        - [State]: 任务状态
        - [Evidence]: 事实证据
        - [Context]: 辅助材料
        - [Output]: 输出约束

        Args:
            selected_packets: 筛选后的上下文包列表
            user_query: 用户查询
            system_instructions: 系统指令

        Returns:
            结构化上下文字符串
        """
        sections = []

        # [Role & Policies] - 系统指令
        p0_packets = [p for p in selected_packets if p.metadata.get("type") == "instructions"]
        if p0_packets:
            role_section = "[Role & Policies]\n"
            role_section += "\n".join([p.content for p in p0_packets])
            sections.append(role_section)

        # [Task] - 当前任务
        sections.append(f"[Task]\n用户问题：{user_query}")

        # [State] - 任务状态
        p1_packets = [p for p in selected_packets if p.metadata.get("type") == "task_state"]
        if p1_packets:
            state_section = "[State]\n关键进展与未决问题：\n"
            state_section += "\n".join([p.content for p in p1_packets])
            sections.append(state_section)

        # [Evidence] - 事实证据
        p2_packets = [
            p for p in selected_packets
            if p.metadata.get("type") in {"related_memory", "knowledge_base", "retrieval", "tool_result"}
        ]
        if p2_packets:
            evidence_section = "[Evidence]\n事实与引用：\n"
            for p in p2_packets:
                evidence_section += f"\n{p.content}\n"
            sections.append(evidence_section)

        # [Context] - 辅助材料（历史等）
        p3_packets = [p for p in selected_packets if p.metadata.get("type") == "history"]
        if p3_packets:
            context_section = "[Context]\n对话历史与背景：\n"
            context_section += "\n".join([p.content for p in p3_packets])
            sections.append(context_section)

        # [Output] - 输出约束
        output_section = """[Output]
                            请按以下格式回答：
                            1. 结论（简洁明确）
                            2. 依据（列出支撑证据及来源）
                            3. 风险与假设（如有）
                            4. 下一步行动建议（如适用）"""
        sections.append(output_section)

        return "\n\n".join(sections)

    def _compress(self, context: str) -> str:
        """
        Compress 阶段：压缩与规范化

        当上下文超过预算时进行压缩，采用按段落截断策略。

        Args:
            context: 待压缩的上下文字符串

        Returns:
            压缩后的上下文字符串
        """
        if not self.config.enable_compression:
            return context

        current_tokens = count_tokens(context)
        available_tokens = self.config.get_available_tokens()

        if current_tokens <= available_tokens:
            return context

        # 简单截断策略（保留前 N 个 token）
        # 实际应用中可用 LLM 做高保真摘要
        print(f"⚠️ 上下文超预算 ({current_tokens} > {available_tokens})，执行截断")

        # 按段落截断，保留结构
        lines = context.split("\n")
        compressed_lines = []
        used_tokens = 0

        for line in lines:
            line_tokens = count_tokens(line)
            if used_tokens + line_tokens > available_tokens:
                break
            compressed_lines.append(line)
            used_tokens += line_tokens

        return "\n".join(compressed_lines)


# ==================== 模块级辅助函数 ====================

def count_tokens(text: str) -> int:
    """
    计算文本 token 数（使用 tiktoken）

    Args:
        text: 待计算的文本

    Returns:
        token 数量
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # 降级方案：粗略估算（1 token ≈ 4 字符）
        return len(text) // 4
