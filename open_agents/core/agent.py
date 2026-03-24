"""
Agent基类

定义了所有 Agent 的抽象基类，提供基本的消息管理和 LLM 调用能力。
子类需要实现 run 方法来定义具体的 Agent 行为。
"""

from abc import ABC, abstractmethod
from typing import Optional
from .message import Message
from .llm import OpenAgentsLLM
from .config import Config


class Agent(ABC):
    """
    Agent抽象基类

    继承自 ABC (Abstract Base Class)，要求子类必须实现 run 方法。
    提供消息历史管理、LLM 调用等核心功能。

    设计理念:
        - 抽象基类定义接口规范
        - 内置消息历史管理
        - 支持自定义配置
    """

    # ==================== 构造方法 ====================

    def __init__(
        self,
        name: str,
        llm: OpenAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        """
        初始化 Agent 实例

        Args:
            name: Agent 名称，用于标识和日志
            llm: LLM 客户端实例，用于调用大语言模型
            system_prompt: 系统提示词，定义 Agent 的角色和行为
            config: 配置对象，如果未提供则使用默认配置
        """
        # ==================== 公有成员变量 ====================
        self.name = name                    # Agent 名称
        self.llm = llm                      # LLM 客户端
        self.system_prompt = system_prompt  # 系统提示词
        self.config = config or Config()    # 配置对象（默认配置）

        # ==================== 私有成员变量 ====================
        self._history: list[Message] = []   # 消息历史记录（私有）

    # ==================== 抽象方法（子类必须实现）====================

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """
        运行 Agent（抽象方法）

        子类必须实现此方法来定义 Agent 的核心行为。

        Args:
            input_text: 用户输入文本
            **kwargs: 额外的关键字参数

        Returns:
            Agent 的响应文本
        """
        pass

    # ==================== 消息管理方法 ====================

    def add_message(self, message: Message):
        """
        添加消息到历史记录

        Args:
            message: 要添加的消息对象
        """
        self._history.append(message)

    def clear_history(self):
        """清空消息历史记录"""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """
        获取消息历史记录的副本

        Returns:
            消息列表的浅拷贝，修改不会影响原始历史记录
        """
        return self._history.copy()

    # ==================== 魔法方法 ====================

    def __str__(self) -> str:
        """
        重写字符串表示方法

        Returns:
            格式为 "Agent(name=xxx, provider=xxx)" 的字符串
        """
        return f"Agent(name={self.name}, provider={self.llm.provider})"

    def __repr__(self) -> str:
        """
        重写对象表示方法

        Returns:
            与 __str__ 相同，便于调试输出
        """
        return self.__str__()
