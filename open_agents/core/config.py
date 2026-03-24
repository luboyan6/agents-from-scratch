"""
配置管理模块

提供 OpenAgents 框架的配置管理功能，支持环境变量和默认值。
使用 Pydantic 进行数据验证和序列化。
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel


class Config(BaseModel):
    """
    OpenAgents 配置类

    继承自 pydantic.BaseModel，提供配置管理和验证功能。
    支持从环境变量加载配置。

    配置分类:
        - LLM 配置: 模型名称、提供商、温度参数等
        - 系统配置: 调试模式、日志级别
        - 其他配置: 历史记录长度限制
    """

    # ==================== 成员变量 ====================

    # LLM 配置
    default_model: str = "gpt-3.5-turbo"    # 默认模型名称
    default_provider: str = "openai"        # 默认 LLM 提供商
    temperature: float = 0.7                # 温度参数（控制随机性）
    max_tokens: Optional[int] = None        # 最大 token 数（None 表示不限制）

    # 系统配置
    debug: bool = False                     # 调试模式开关
    log_level: str = "INFO"                 # 日志级别

    # 其他配置
    max_history_length: int = 100           # 最大历史记录长度

    # ==================== 类方法 ====================

    @classmethod
    def from_env(cls) -> "Config":
        """
        从环境变量创建配置实例

        支持的环境变量:
            - DEBUG: 调试模式 ("true" / "false")
            - LOG_LEVEL: 日志级别
            - TEMPERATURE: 温度参数
            - MAX_TOKENS: 最大 token 数

        Returns:
            从环境变量加载的 Config 实例
        """
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )

    # ==================== 实例方法 ====================

    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典格式

        Returns:
            包含所有配置项的字典
        """
        return self.model_dump()
