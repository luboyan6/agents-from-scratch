"""消息系统"""

from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel

# 消息角色类型：仅允许 "user"、"assistant"、"system"、"tool" 四种值
MessageRole = Literal["user", "assistant", "system", "tool"]

class Message(BaseModel):
    """
    消息类

    继承自 pydantic.BaseModel，提供数据验证和序列化功能
    用于表示对话中的单条消息
    """

    # ==================== 成员变量 ====================

    content: str                    # 消息内容
    role: MessageRole               # 消息角色（user/assistant/system/tool）
    timestamp: datetime = None      # 时间戳，创建时自动设置
    metadata: Optional[Dict[str, Any]] = None  # 可选的元数据字典

    # ==================== 构造方法 ====================

    def __init__(self, content: str, role: MessageRole, **kwargs):
        """
        初始化消息实例

        Args:
            content: 消息内容
            role: 消息角色
            **kwargs: 可选关键字参数
                - timestamp: 自定义时间戳，默认为当前时间
                - metadata: 自定义元数据，默认为空字典

        Note:
            重写父类 BaseModel.__init__，为 timestamp 和 metadata 提供默认值
        """
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get('timestamp', datetime.now()),  # 默认当前时间
            metadata=kwargs.get('metadata', {})                  # 默认空字典
        )

    # ==================== 实例方法 ====================

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式（OpenAI API 格式）

        Returns:
            包含 role 和 content 的字典，用于调用 OpenAI API

        Note:
            仅输出 role 和 content，不包含 timestamp 和 metadata
        """
        return {
            "role": self.role,
            "content": self.content
        }

    # ==================== 魔法方法 ====================

    def __str__(self) -> str:
        """
        重写字符串表示方法

        Returns:
            格式为 "[role] content" 的字符串

        Example:
            >>> msg = Message("Hello", "user")
            >>> print(msg)
            [user] Hello
        """
        return f"[{self.role}] {self.content}"
