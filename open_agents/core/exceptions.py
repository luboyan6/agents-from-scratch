"""
异常体系模块

定义了 OpenAgents 框架的异常层次结构。
所有框架异常都继承自 OpenAgentsException 基类。
"""


class OpenAgentsException(Exception):
    """
    OpenAgents 基础异常类

    所有框架自定义异常的基类，便于统一捕获和处理。

    继承关系:
        OpenAgentsException (基类)
        ├── LLMException (LLM 相关异常)
        ├── AgentException (Agent 相关异常)
        ├── ConfigException (配置相关异常)
        └── ToolException (工具相关异常)
    """
    pass


class LLMException(OpenAgentsException):
    """
    LLM 相关异常

    当 LLM 调用、响应处理等发生错误时抛出。

    使用场景:
        - API 调用失败
        - 响应解析错误
        - 模型不可用
    """
    pass


class AgentException(OpenAgentsException):
    """
    Agent 相关异常

    当 Agent 执行、状态管理等发生错误时抛出。

    使用场景:
        - Agent 初始化失败
        - 运行时错误
        - 消息处理错误
    """
    pass


class ConfigException(OpenAgentsException):
    """
    配置相关异常

    当配置加载、验证等发生错误时抛出。

    使用场景:
        - 配置文件解析错误
        - 必要配置缺失
        - 配置值无效
    """
    pass


class ToolException(OpenAgentsException):
    """
    工具相关异常

    当工具调用、执行等发生错误时抛出。

    使用场景:
        - 工具不存在
        - 工具参数错误
        - 工具执行失败
    """
    pass
