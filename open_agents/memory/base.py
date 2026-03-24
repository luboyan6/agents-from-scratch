"""
记忆系统基础类和配置

按照第8章架构设计的基础组件：
- MemoryItem: 记忆项数据结构
- MemoryConfig: 记忆系统配置
- BaseMemory: 记忆基类
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel


# ==================== 数据类 ====================

class MemoryItem(BaseModel):
    """
    记忆项数据结构

    用于封装单个记忆条目，包含内容、类型、时间戳等元数据。

    继承:
        BaseModel: Pydantic基类

    成员变量:
        id: 记忆项唯一标识
        content: 记忆内容文本
        memory_type: 记忆类型（working/episodic/semantic/perceptual）
        user_id: 用户标识
        timestamp: 创建时间戳
        importance: 重要性分数（0.0-1.0）
        metadata: 元数据字典
    """

    # ==================== 成员变量 ====================
    id: str                                           # 记忆项唯一标识
    content: str                                      # 记忆内容文本
    memory_type: str                                  # 记忆类型
    user_id: str                                      # 用户标识
    timestamp: datetime                               # 创建时间戳
    importance: float = 0.5                           # 重要性分数
    metadata: Dict[str, Any] = {}                     # 元数据字典

    # ==================== Pydantic配置 ====================

    class Config:
        arbitrary_types_allowed = True


class MemoryConfig(BaseModel):
    """
    记忆系统配置

    用于配置各类记忆系统的参数设置。

    继承:
        BaseModel: Pydantic基类

    成员变量:
        storage_path: 存储路径
        max_capacity: 最大容量（统计显示用）
        importance_threshold: 重要性阈值
        decay_factor: 衰减因子
        working_memory_capacity: 工作记忆容量
        working_memory_tokens: 工作记忆token限制
        working_memory_ttl_minutes: 工作记忆TTL（分钟）
        perceptual_memory_modalities: 感知记忆支持的模态
    """

    # ==================== 成员变量 ====================

    # 存储路径
    storage_path: str = "./memory_data"

    # 统计显示用的基础配置（仅用于展示）
    max_capacity: int = 100
    importance_threshold: float = 0.1
    decay_factor: float = 0.95

    # 工作记忆特定配置
    working_memory_capacity: int = 10
    working_memory_tokens: int = 2000
    working_memory_ttl_minutes: int = 120

    # 感知记忆特定配置
    perceptual_memory_modalities: List[str] = ["text", "image", "audio", "video"]


# ==================== 抽象基类 ====================

class BaseMemory(ABC):
    """
    记忆基类

    定义所有记忆类型的通用接口和行为。

    继承:
        ABC: 抽象基类

    成员变量:
        config: 记忆配置对象
        storage: 存储后端
        memory_type: 记忆类型名称
    """

    # ==================== 构造方法 ====================

    def __init__(self, config: MemoryConfig, storage_backend=None):
        """
        初始化记忆基类

        Args:
            config: 记忆配置对象
            storage_backend: 存储后端实例
        """
        # ==================== 成员变量 ====================
        self.config = config                        # 记忆配置
        self.storage = storage_backend              # 存储后端
        self.memory_type = self.__class__.__name__.lower().replace("memory", "")  # 记忆类型名称

    # ==================== 抽象方法 ====================

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """
        添加记忆项

        Args:
            memory_item: 记忆项对象

        Returns:
            记忆ID
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """
        检索相关记忆

        Args:
            query: 查询内容
            limit: 返回数量限制
            **kwargs: 其他检索参数

        Returns:
            相关记忆列表
        """
        pass

    @abstractmethod
    def update(self, memory_id: str, content: str = None,
               importance: float = None, metadata: Dict[str, Any] = None) -> bool:
        """
        更新记忆

        Args:
            memory_id: 记忆ID
            content: 新内容
            importance: 新重要性
            metadata: 新元数据

        Returns:
            是否更新成功
        """
        pass

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """
        删除记忆

        Args:
            memory_id: 记忆ID

        Returns:
            是否删除成功
        """
        pass

    @abstractmethod
    def has_memory(self, memory_id: str) -> bool:
        """
        检查记忆是否存在

        Args:
            memory_id: 记忆ID

        Returns:
            是否存在
        """
        pass

    @abstractmethod
    def clear(self):
        """清空所有记忆"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        获取记忆统计信息

        Returns:
            统计信息字典
        """
        pass

    # ==================== 受保护方法 ====================

    def _generate_id(self) -> str:
        """生成记忆ID"""
        import uuid
        return str(uuid.uuid4())

    def _calculate_importance(self, content: str, base_importance: float = 0.5) -> float:
        """
        计算记忆重要性

        Args:
            content: 记忆内容
            base_importance: 基础重要性

        Returns:
            计算后的重要性分数
        """
        importance = base_importance

        # 基于内容长度
        if len(content) > 100:
            importance += 0.1

        # 基于关键词
        important_keywords = ["重要", "关键", "必须", "注意", "警告", "错误"]
        if any(keyword in content for keyword in important_keywords):
            importance += 0.2

        return max(0.0, min(1.0, importance))

    # ==================== 魔法方法 ====================

    def __str__(self) -> str:
        stats = self.get_stats()
        return f"{self.__class__.__name__}(count={stats.get('count', 0)})"

    def __repr__(self) -> str:
        return self.__str__()
