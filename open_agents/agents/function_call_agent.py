"""
FunctionCallAgent - 使用OpenAI函数调用范式的Agent实现

该模块实现了基于 OpenAI 原生函数调用机制的 Agent，支持：
- 自动将工具转换为 OpenAI 函数调用格式
- 多轮工具调用迭代
- 参数类型自动转换
- 流式调用支持
"""

from __future__ import annotations

import json
from typing import Iterator, Optional, Union, TYPE_CHECKING, Any, Dict

from ..core.agent import Agent
from ..core.config import Config
from ..core.llm import OpenAgentsLLM
from ..core.message import Message

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry


# ==================== 模块级辅助函数 ====================

def _map_parameter_type(param_type: str) -> str:
    """
    将工具参数类型映射为 JSON Schema 允许的类型

    Args:
        param_type: 原始参数类型字符串

    Returns:
        JSON Schema 兼容的类型字符串
    """
    normalized = (param_type or "").lower()
    if normalized in {"string", "number", "integer", "boolean", "array", "object"}:
        return normalized
    return "string"


class FunctionCallAgent(Agent):
    """
    基于 OpenAI 原生函数调用机制的 Agent

    该 Agent 使用 OpenAI 的函数调用（Function Calling）能力，
    自动将注册的工具转换为 OpenAI 兼容的函数格式，并处理多轮工具调用。

    特性:
        - 自动生成工具的 JSON Schema
        - 支持多轮工具调用迭代
        - 参数类型自动转换
        - 支持自定义工具选择策略

    继承:
        Agent: 抽象基类
    """

    # ==================== 构造方法 ====================

    def __init__(
        self,
        name: str,
        llm: OpenAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        enable_tool_calling: bool = True,
        default_tool_choice: Union[str, dict] = "auto",
        max_tool_iterations: int = 3,
    ):
        """
        初始化 FunctionCallAgent

        Args:
            name: Agent 名称
            llm: LLM 客户端实例
            system_prompt: 系统提示词（可选）
            config: 配置对象（可选）
            tool_registry: 工具注册表（可选）
            enable_tool_calling: 是否启用工具调用
            default_tool_choice: 默认工具选择策略（"auto"/"none"/工具定义）
            max_tool_iterations: 最大工具调用迭代次数
        """
        super().__init__(name, llm, system_prompt, config)

        # ==================== 成员变量 ====================
        self.tool_registry = tool_registry                                    # 工具注册表
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None  # 是否启用工具调用
        self.default_tool_choice = default_tool_choice                         # 默认工具选择策略
        self.max_tool_iterations = max_tool_iterations                         # 最大迭代次数

    # ==================== 私有方法 ====================

    def _get_system_prompt(self) -> str:
        """
        构建系统提示词，注入工具描述

        Returns:
            包含工具信息的系统提示词
        """
        base_prompt = self.system_prompt or "你是一个可靠的AI助理，能够在需要时调用工具完成任务。"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt

        prompt = base_prompt + "\n\n## 可用工具\n"
        prompt += "当你判断需要外部信息或执行动作时，可以直接通过函数调用使用以下工具：\n"
        prompt += tools_description + "\n"
        prompt += "\n请主动决定是否调用工具，合理利用多次调用来获得完备答案。"
        return prompt

    def _build_tool_schemas(self) -> list[dict[str, Any]]:
        """
        构建工具的 JSON Schema 格式定义

        Returns:
            OpenAI 函数调用格式的工具定义列表
        """
        if not self.enable_tool_calling or not self.tool_registry:
            return []

        schemas: list[dict[str, Any]] = []

        # 处理 Tool 对象
        for tool in self.tool_registry.get_all_tools():
            properties: Dict[str, Any] = {}
            required: list[str] = []
            try:
                parameters = tool.get_parameters()
            except Exception:
                parameters = []

            for param in parameters:
                properties[param.name] = {
                    "type": _map_parameter_type(param.type),
                    "description": param.description or ""
                }
                if param.default is not None:
                    properties[param.name]["default"] = param.default
                if getattr(param, "required", True):
                    required.append(param.name)

            schema: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties
                    }
                }
            }
            if required:
                schema["function"]["parameters"]["required"] = required
            schemas.append(schema)

        # 处理通过 register_function 注册的工具
        function_map = getattr(self.tool_registry, "_functions", {})
        for name, info in function_map.items():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": info.get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "输入文本"
                                }
                            },
                            "required": ["input"]
                        }
                    }
                }
            )

        return schemas

    @staticmethod
    def _extract_message_content(raw_content: Any) -> str:
        """
        从 OpenAI 响应的 message.content 中安全提取文本

        Args:
            raw_content: 原始响应内容

        Returns:
            提取的文本字符串
        """
        if raw_content is None:
            return ""
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                text = getattr(item, "text", None)
                if text is None and isinstance(item, dict):
                    text = item.get("text")
                if text:
                    parts.append(text)
            return "".join(parts)
        return str(raw_content)

    @staticmethod
    def _parse_function_call_arguments(arguments: Optional[str]) -> dict[str, Any]:
        """
        解析模型返回的 JSON 字符串参数

        Args:
            arguments: JSON 格式的参数字符串

        Returns:
            解析后的参数字典
        """
        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _convert_parameter_types(self, tool_name: str, param_dict: dict[str, Any]) -> dict[str, Any]:
        """
        根据工具定义尽可能转换参数类型

        Args:
            tool_name: 工具名称
            param_dict: 原始参数字典

        Returns:
            类型转换后的参数字典
        """
        if not self.tool_registry:
            return param_dict

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return param_dict

        try:
            tool_params = tool.get_parameters()
        except Exception:
            return param_dict

        type_mapping = {param.name: param.type for param in tool_params}
        converted: dict[str, Any] = {}

        for key, value in param_dict.items():
            param_type = type_mapping.get(key)
            if not param_type:
                converted[key] = value
                continue

            try:
                normalized = param_type.lower()
                if normalized in {"number", "float"}:
                    converted[key] = float(value)
                elif normalized in {"integer", "int"}:
                    converted[key] = int(value)
                elif normalized in {"boolean", "bool"}:
                    if isinstance(value, bool):
                        converted[key] = value
                    elif isinstance(value, (int, float)):
                        converted[key] = bool(value)
                    elif isinstance(value, str):
                        converted[key] = value.lower() in {"true", "1", "yes"}
                    else:
                        converted[key] = bool(value)
                else:
                    converted[key] = value
            except (TypeError, ValueError):
                converted[key] = value

        return converted

    def _execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        执行工具调用并返回字符串结果

        Args:
            tool_name: 工具名称
            arguments: 工具参数字典

        Returns:
            工具执行结果字符串
        """
        if not self.tool_registry:
            return "❌ 错误：未配置工具注册表"

        tool = self.tool_registry.get_tool(tool_name)
        if tool:
            try:
                typed_arguments = self._convert_parameter_types(tool_name, arguments)
                return tool.run(typed_arguments)
            except Exception as exc:
                return f"❌ 工具调用失败：{exc}"

        func = self.tool_registry.get_function(tool_name)
        if func:
            try:
                input_text = arguments.get("input", "")
                return func(input_text)
            except Exception as exc:
                return f"❌ 工具调用失败：{exc}"

        return f"❌ 错误：未找到工具 '{tool_name}'"

    def _invoke_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: Union[str, dict], **kwargs):
        """
        调用底层 OpenAI 客户端执行函数调用

        Args:
            messages: 消息列表
            tools: 工具定义列表
            tool_choice: 工具选择策略
            **kwargs: 额外参数

        Returns:
            OpenAI API 响应对象

        Raises:
            RuntimeError: 客户端未正确初始化
        """
        client = getattr(self.llm, "_client", None)
        if client is None:
            raise RuntimeError("OpenAgentsLLM 未正确初始化客户端，无法执行函数调用。")

        client_kwargs = dict(kwargs)
        client_kwargs.setdefault("temperature", self.llm.temperature)
        if self.llm.max_tokens is not None:
            client_kwargs.setdefault("max_tokens", self.llm.max_tokens)

        return client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **client_kwargs,
        )

    # ==================== 重写父类方法 ====================

    def run(
        self,
        input_text: str,
        *,
        max_tool_iterations: Optional[int] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs,
    ) -> str:
        """
        执行函数调用范式的对话流程（重写父类方法）

        Args:
            input_text: 用户输入文本
            max_tool_iterations: 最大工具调用迭代次数（可选）
            tool_choice: 工具选择策略（可选）
            **kwargs: 传递给 LLM 的额外参数

        Returns:
            Agent 的响应文本
        """
        messages: list[dict[str, Any]] = []
        system_prompt = self._get_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        tool_schemas = self._build_tool_schemas()
        if not tool_schemas:
            response_text = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response_text, "assistant"))
            return response_text

        iterations_limit = max_tool_iterations if max_tool_iterations is not None else self.max_tool_iterations
        effective_tool_choice: Union[str, dict] = tool_choice if tool_choice is not None else self.default_tool_choice

        current_iteration = 0
        final_response = ""

        while current_iteration < iterations_limit:
            response = self._invoke_with_tools(
                messages,
                tools=tool_schemas,
                tool_choice=effective_tool_choice,
                **kwargs,
            )

            choice = response.choices[0]
            assistant_message = choice.message
            content = self._extract_message_content(assistant_message.content)
            tool_calls = list(assistant_message.tool_calls or [])

            if tool_calls:
                assistant_payload: dict[str, Any] = {"role": "assistant", "content": content}
                assistant_payload["tool_calls"] = []

                for tool_call in tool_calls:
                    assistant_payload["tool_calls"].append(
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )
                messages.append(assistant_payload)

                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    arguments = self._parse_function_call_arguments(tool_call.function.arguments)
                    result = self._execute_tool_call(tool_name, arguments)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": result,
                        }
                    )

                current_iteration += 1
                continue

            final_response = content
            messages.append({"role": "assistant", "content": final_response})
            break

        if current_iteration >= iterations_limit and not final_response:
            final_choice = self._invoke_with_tools(
                messages,
                tools=tool_schemas,
                tool_choice="none",
                **kwargs,
            )
            final_response = self._extract_message_content(final_choice.choices[0].message.content)
            messages.append({"role": "assistant", "content": final_response})

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        return final_response

    # ==================== 公有方法 ====================

    def add_tool(self, tool) -> None:
        """
        便捷方法：将工具注册到当前 Agent

        Args:
            tool: 工具实例
        """
        if not self.tool_registry:
            from ..tools.registry import ToolRegistry

            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        if hasattr(tool, "auto_expand") and getattr(tool, "auto_expand"):
            expanded_tools = tool.get_expanded_tools()
            if expanded_tools:
                for expanded_tool in expanded_tools:
                    self.tool_registry.register_tool(expanded_tool)
                print(f"✅ MCP工具 '{tool.name}' 已展开为 {len(expanded_tools)} 个独立工具")
                return

        self.tool_registry.register_tool(tool)

    def remove_tool(self, tool_name: str) -> bool:
        """
        移除工具

        Args:
            tool_name: 工具名称

        Returns:
            是否成功移除
        """
        if self.tool_registry:
            before = set(self.tool_registry.list_tools())
            self.tool_registry.unregister(tool_name)
            after = set(self.tool_registry.list_tools())
            return tool_name in before and tool_name not in after
        return False

    def list_tools(self) -> list[str]:
        """
        列出所有可用工具

        Returns:
            工具名称列表
        """
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []

    def has_tools(self) -> bool:
        """
        检查是否有可用工具

        Returns:
            是否有可用工具
        """
        return self.enable_tool_calling and self.tool_registry is not None

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        流式调用（暂未实现，回退到一次性调用）

        Args:
            input_text: 用户输入文本
            **kwargs: 传递给 LLM 的额外参数

        Yields:
            响应文本片段
        """
        result = self.run(input_text, **kwargs)
        yield result
