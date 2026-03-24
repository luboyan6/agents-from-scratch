"""
数据库配置管理模块

支持 Qdrant 向量数据库和 Neo4j 图数据库的配置管理。
提供环境变量加载、配置验证和连接测试功能。
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

# 配置日志记录器
logger = logging.getLogger(__name__)

# 加载环境变量（确保在配置类使用前加载）
load_dotenv()


class QdrantConfig(BaseModel):
    """
    Qdrant 向量数据库配置类

    用于配置 Qdrant 向量数据库的连接参数和集合设置。
    支持云服务和本地部署。

    配置分类:
        - 连接配置: URL、API 密钥、超时时间
        - 集合配置: 集合名称、向量维度、距离度量
    """

    # ==================== 成员变量 ====================

    # 连接配置
    url: Optional[str] = Field(
        default=None,
        description="Qdrant服务URL (云服务或自定义URL)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API密钥 (云服务需要)"
    )

    # 集合配置
    collection_name: str = Field(
        default="open_agents_vectors",
        description="向量集合名称"
    )
    vector_size: int = Field(
        default=384,
        description="向量维度"
    )
    distance: str = Field(
        default="cosine",
        description="距离度量方式 (cosine, dot, euclidean)"
    )

    # 连接配置
    timeout: int = Field(
        default=30,
        description="连接超时时间(秒)"
    )

    # ==================== 类方法 ====================

    @classmethod
    def from_env(cls) -> "QdrantConfig":
        """
        从环境变量创建 Qdrant 配置

        支持的环境变量:
            - QDRANT_URL: 服务地址
            - QDRANT_API_KEY: API 密钥
            - QDRANT_COLLECTION: 集合名称
            - QDRANT_VECTOR_SIZE: 向量维度
            - QDRANT_DISTANCE: 距离度量方式
            - QDRANT_TIMEOUT: 超时时间

        Returns:
            从环境变量加载的 QdrantConfig 实例
        """
        return cls(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv("QDRANT_COLLECTION", "open_agents_vectors"),
            vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", "384")),
            distance=os.getenv("QDRANT_DISTANCE", "cosine"),
            timeout=int(os.getenv("QDRANT_TIMEOUT", "30"))
        )

    # ==================== 实例方法 ====================

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式

        Returns:
            排除 None 值的配置字典
        """
        return self.model_dump(exclude_none=True)


class Neo4jConfig(BaseModel):
    """
    Neo4j 图数据库配置类

    用于配置 Neo4j 图数据库的连接参数和连接池设置。

    配置分类:
        - 连接配置: URI、用户名、密码、数据库名称
        - 连接池配置: 最大连接生命周期、连接池大小、超时时间
    """

    # ==================== 成员变量 ====================

    # 连接配置
    uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j连接URI"
    )
    username: str = Field(
        default="neo4j",
        description="用户名"
    )
    password: str = Field(
        default="hello-agents-password",
        description="密码"
    )
    database: str = Field(
        default="neo4j",
        description="数据库名称"
    )

    # 连接池配置
    max_connection_lifetime: int = Field(
        default=3600,
        description="最大连接生命周期(秒)"
    )
    max_connection_pool_size: int = Field(
        default=50,
        description="最大连接池大小"
    )
    connection_acquisition_timeout: int = Field(
        default=60,
        description="连接获取超时(秒)"
    )

    # ==================== 类方法 ====================

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """
        从环境变量创建 Neo4j 配置

        支持的环境变量:
            - NEO4J_URI: 连接 URI
            - NEO4J_USERNAME: 用户名
            - NEO4J_PASSWORD: 密码
            - NEO4J_DATABASE: 数据库名称
            - NEO4J_MAX_CONNECTION_LIFETIME: 最大连接生命周期
            - NEO4J_MAX_CONNECTION_POOL_SIZE: 最大连接池大小
            - NEO4J_CONNECTION_TIMEOUT: 连接获取超时

        Returns:
            从环境变量加载的 Neo4jConfig 实例
        """
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "hello-agents-password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            max_connection_lifetime=int(os.getenv("NEO4J_MAX_CONNECTION_LIFETIME", "3600")),
            max_connection_pool_size=int(os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", "50")),
            connection_acquisition_timeout=int(os.getenv("NEO4J_CONNECTION_TIMEOUT", "60"))
        )

    # ==================== 实例方法 ====================

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式

        Returns:
            包含所有配置项的字典
        """
        return self.model_dump()


class DatabaseConfig(BaseModel):
    """
    数据库配置管理器

    统一管理 Qdrant 和 Neo4j 数据库配置。
    提供配置加载、获取和连接验证功能。
    """

    # ==================== 成员变量 ====================

    qdrant: QdrantConfig = Field(
        default_factory=QdrantConfig,
        description="Qdrant向量数据库配置"
    )
    neo4j: Neo4jConfig = Field(
        default_factory=Neo4jConfig,
        description="Neo4j图数据库配置"
    )

    # ==================== 类方法 ====================

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """
        从环境变量创建数据库配置

        Returns:
            从环境变量加载的 DatabaseConfig 实例
        """
        return cls(
            qdrant=QdrantConfig.from_env(),
            neo4j=Neo4jConfig.from_env()
        )

    # ==================== 实例方法 ====================

    def get_qdrant_config(self) -> Dict[str, Any]:
        """
        获取 Qdrant 配置字典

        Returns:
            Qdrant 配置字典
        """
        return self.qdrant.to_dict()

    def get_neo4j_config(self) -> Dict[str, Any]:
        """
        获取 Neo4j 配置字典

        Returns:
            Neo4j 配置字典
        """
        return self.neo4j.to_dict()

    def validate_connections(self) -> Dict[str, bool]:
        """
        验证数据库连接配置

        尝试连接各个数据库并返回连接状态。

        Returns:
            字典，键为数据库名称，值为连接是否成功
        """
        results = {}

        # 验证 Qdrant 配置
        try:
            from ..memory.storage.qdrant_store import QdrantVectorStore
            qdrant_store = QdrantVectorStore(**self.get_qdrant_config())
            results["qdrant"] = qdrant_store.health_check()
            logger.info(f"✅ Qdrant连接验证: {'成功' if results['qdrant'] else '失败'}")
        except Exception as e:
            results["qdrant"] = False
            logger.error(f"❌ Qdrant连接验证失败: {e}")

        # 验证 Neo4j 配置
        try:
            from ..memory.storage.neo4j_store import Neo4jGraphStore
            neo4j_store = Neo4jGraphStore(**self.get_neo4j_config())
            results["neo4j"] = neo4j_store.health_check()
            logger.info(f"✅ Neo4j连接验证: {'成功' if results['neo4j'] else '失败'}")
        except Exception as e:
            results["neo4j"] = False
            logger.error(f"❌ Neo4j连接验证失败: {e}")

        return results


# ==================== 全局配置实例 ====================

# 全局数据库配置实例（启动时从环境变量加载）
db_config = DatabaseConfig.from_env()


# ==================== 模块级辅助函数 ====================

def get_database_config() -> DatabaseConfig:
    """
    获取数据库配置实例

    Returns:
        全局数据库配置实例
    """
    return db_config


def update_database_config(**kwargs) -> None:
    """
    更新数据库配置

    Args:
        **kwargs: 配置更新项
            - qdrant: QdrantConfig 配置字典
            - neo4j: Neo4jConfig 配置字典
    """
    global db_config

    if "qdrant" in kwargs:
        db_config.qdrant = QdrantConfig(**kwargs["qdrant"])

    if "neo4j" in kwargs:
        db_config.neo4j = Neo4jConfig(**kwargs["neo4j"])

    logger.info("✅ 数据库配置已更新")
