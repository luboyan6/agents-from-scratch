"""
Plan and Solve Agent实现 - 分解规划与逐步执行的智能体

该模块实现了 Plan-and-Solve 范式的 Agent，包含：
- Planner: 规划器，将复杂问题分解为简单步骤
- Executor: 执行器，按计划逐步执行
- PlanAndSolveAgent: 组合规划器和执行器的完整 Agent

适合任务：
- 多步骤推理
- 数学问题
- 复杂分析
"""

import ast
from typing import Optional, List, Dict
from ..core.agent import Agent
from ..core.llm import OpenAgentsLLM
from ..core.config import Config
from ..core.message import Message


# ==================== 默认提示词模板 ====================

DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""


# ==================== 辅助类 ====================

class Planner:
    """
    规划器 - 负责将复杂问题分解为简单步骤

    使用 LLM 将复杂问题分解为可执行的步骤列表。

    成员变量:
        llm_client: LLM 客户端实例
        prompt_template: 规划提示词模板
    """

    def __init__(self, llm_client: OpenAgentsLLM, prompt_template: Optional[str] = None):
        """
        初始化规划器

        Args:
            llm_client: LLM 客户端实例
            prompt_template: 自定义提示词模板（可选）
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else DEFAULT_PLANNER_PROMPT

    def plan(self, question: str, **kwargs) -> List[str]:
        """
        生成执行计划

        Args:
            question: 要解决的问题
            **kwargs: LLM 调用参数

        Returns:
            步骤列表
        """
        prompt = self.prompt_template.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成计划 ---")
        response_text = self.llm_client.invoke(messages, **kwargs) or ""
        print(f"✅ 计划已生成:\n{response_text}")

        try:
            # 提取 Python 代码块中的列表
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []


class Executor:
    """
    执行器 - 负责按计划逐步执行

    按照规划器生成的计划，逐步执行每个步骤。

    成员变量:
        llm_client: LLM 客户端实例
        prompt_template: 执行提示词模板
    """

    def __init__(self, llm_client: OpenAgentsLLM, prompt_template: Optional[str] = None):
        """
        初始化执行器

        Args:
            llm_client: LLM 客户端实例
            prompt_template: 自定义提示词模板（可选）
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else DEFAULT_EXECUTOR_PROMPT

    def execute(self, question: str, plan: List[str], **kwargs) -> str:
        """
        按计划执行任务

        Args:
            question: 原始问题
            plan: 执行计划
            **kwargs: LLM 调用参数

        Returns:
            最终答案
        """
        history = ""
        final_answer = ""

        print("\n--- 正在执行计划 ---")
        for i, step in enumerate(plan, 1):
            print(f"\n-> 正在执行步骤 {i}/{len(plan)}: {step}")
            prompt = self.prompt_template.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step
            )
            messages = [{"role": "user", "content": prompt}]

            response_text = self.llm_client.invoke(messages, **kwargs) or ""

            history += f"步骤 {i}: {step}\n结果: {response_text}\n\n"
            final_answer = response_text
            print(f"✅ 步骤 {i} 已完成，结果: {final_answer}")

        return final_answer


# ==================== 主类 ====================

class PlanAndSolveAgent(Agent):
    """
    Plan and Solve Agent - 分解规划与逐步执行的智能体

    该 Agent 能够：
    1. 将复杂问题分解为简单步骤
    2. 按照计划逐步执行
    3. 维护执行历史和上下文
    4. 得出最终答案

    特别适合多步骤推理、数学问题、复杂分析等任务。

    继承:
        Agent: 抽象基类

    成员变量:
        planner: 规划器实例
        executor: 执行器实例
    """

    def __init__(
        self,
        name: str,
        llm: OpenAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        """
        初始化 PlanAndSolveAgent

        Args:
            name: Agent 名称
            llm: LLM 客户端实例
            system_prompt: 系统提示词（可选）
            config: 配置对象（可选）
            custom_prompts: 自定义提示词模板 {"planner": "", "executor": ""}
        """
        super().__init__(name, llm, system_prompt, config)

        # 设置提示词模板：用户自定义优先，否则使用默认模板
        if custom_prompts:
            planner_prompt = custom_prompts.get("planner")
            executor_prompt = custom_prompts.get("executor")
        else:
            planner_prompt = None
            executor_prompt = None

        # ==================== 成员变量 ====================
        self.planner = Planner(self.llm, planner_prompt)     # 规划器
        self.executor = Executor(self.llm, executor_prompt)  # 执行器

    # ==================== 重写父类方法 ====================

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行 Plan and Solve Agent（重写父类方法）

        Args:
            input_text: 要解决的问题
            **kwargs: 其他参数

        Returns:
            最终答案
        """
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        # 1. 生成计划
        plan = self.planner.plan(input_text, **kwargs)
        if not plan:
            final_answer = "无法生成有效的行动计划，任务终止。"
            print(f"\n--- 任务终止 ---\n{final_answer}")

            # 保存到历史记录
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))

            return final_answer

        # 2. 执行计划
        final_answer = self.executor.execute(input_text, plan, **kwargs)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))

        return final_answer
