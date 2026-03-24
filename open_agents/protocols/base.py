"""协议基类（概念性）

本模块定义了协议的基本接口概念。
实际实现中，各协议根据自己的特点独立实现，不强制继承这个基类。

协议接口概念：
- 协议标识：每个协议有唯一的名称和版本
- 消息传递：支持发送和接收消息
- 信息查询：可以获取协议的基本信息

实际使用：
- MCP: 使用 fastmcp 库实现
- A2A: 使用官方 a2a 库实现
- ANP: 使用概念性实现

注意：这个基类主要用于文档说明，实际协议实现不需要继承它。
"""

from enum import Enum


# ==================== 枚举类 ====================

class ProtocolType(Enum):
    """
    协议类型枚举

    定义支持的协议类型。
    """

    MCP = "mcp"      # Model Context Protocol
    A2A = "a2a"      # Agent-to-Agent Protocol
    ANP = "anp"      # Agent Network Protocol


# ==================== 主类 ====================

# 为了向后兼容，保留 Protocol 类的定义
# 但标记为概念性，不建议实际使用
class Protocol:
    """
    协议基类（概念性，不建议继承）

    这个类定义了协议的基本概念，但实际实现不需要继承它。
    各协议根据自己的特点独立实现。

    继承:
        无

    成员变量:
        _protocol_type: 协议类型
        _version: 协议版本
    """

    # ==================== 构造方法 ====================

    def __init__(self, protocol_type: ProtocolType, version: str = "1.0.0"):
        """
        初始化协议

        Args:
            protocol_type: 协议类型
            version: 协议版本
        """
        # ==================== 成员变量 ====================
        self._protocol_type = protocol_type                 # 协议类型
        self._version = version                             # 协议版本

    # ==================== 属性方法 ====================

    @property
    def protocol_name(self) -> str:
        """获取协议名称"""
        return self._protocol_type.value

    @property
    def version(self) -> str:
        """获取协议版本"""
        return self._version

    # ==================== 魔法方法 ====================

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(protocol={self.protocol_name}, version={self.version})"

    def __repr__(self) -> str:
        return self.__str__()

