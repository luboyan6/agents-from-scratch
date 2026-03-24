"""
Reflection Agent实现 - 自我反思与迭代优化的智能体

该模块实现了 Reflection 范式的 Agent，通过自我反思和迭代优化来改进输出质量。

工作流程:
1. 初始执行：生成初始回答
2. 反思：审查回答，找出问题和改进空间
3. 优化：根据反思结果改进回答
4. 迭代：重复反思-优化直到满意或达到最大迭代次数

适合任务：
- 代码生成
- 文档写作
- 分析报告
- 需要高质量输出的场景
"""

from typing import Optional, List, Dict, Any
from ..core.agent import Agent
from ..core.llm import OpenAgentsLLM
from ..core.config import Config
from ..core.message import Message


# ==================== 默认提示词模板 ====================

DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务：

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间：

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答：

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""
}

# ==================== 辅助类 ====================

class Memory:
    """
    简单的短期记忆模块，用于存储智能体的行动与反思轨迹。

    成员变量:
        records: 记录列表，每条记录包含类型和内容
    """

    def __init__(self):
        """初始化记忆模块"""
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        向记忆中添加一条新记录

        Args:
            record_type: 记录类型（如 "execution"、"reflection"）
            content: 记录内容
        """
        self.records.append({"type": record_type, "content": content})
        print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")

    def get_trajectory(self) -> str:
        """
        将所有记忆记录格式化为一个连贯的字符串文本

        Returns:
            格式化后的轨迹文本
        """
        trajectory = ""
        for record in self.records:
            if record['type'] == 'execution':
                trajectory += f"--- 上一轮尝试 (代码) ---\n{record['content']}\n\n"
            elif record['type'] == 'reflection':
                trajectory += f"--- 评审员反馈 ---\n{record['content']}\n\n"
        return trajectory.strip()

    def get_last_execution(self) -> str:
        """
        获取最近一次的执行结果

        Returns:
            最近的执行结果，如果没有则返回空字符串
        """
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return ""


# ==================== 主类 ====================

class ReflectionAgent(Agent):
    """
    Reflection Agent - 自我反思与迭代优化的智能体

    该 Agent 能够：
    1. 执行初始任务
    2. 对结果进行自我反思
    3. 根据反思结果进行优化
    4. 迭代改进直到满意

    特别适合代码生成、文档写作、分析报告等需要迭代优化的任务。

    支持多种专业领域的提示词模板，用户可以自定义或使用内置模板。

    继承:
        Agent: 抽象基类

    成员变量:
        max_iterations: 最大迭代次数
        memory: 记忆模块
        prompts: 提示词模板字典
    """

    # ==================== 构造方法 ====================

    def __init__(
        self,
        name: str,
        llm: OpenAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        """
        初始化 ReflectionAgent

        Args:
            name: Agent 名称
            llm: LLM 客户端实例
            system_prompt: 系统提示词（可选）
            config: 配置对象（可选）
            max_iterations: 最大迭代次数
            custom_prompts: 自定义提示词模板 {"initial": "", "reflect": "", "refine": ""}
        """
        super().__init__(name, llm, system_prompt, config)

        # ==================== 成员变量 ====================
        self.max_iterations = max_iterations                         # 最大迭代次数
        self.memory = Memory()                                       # 记忆模块

        # 设置提示词模板：用户自定义优先，否则使用默认模板
        self.prompts = custom_prompts if custom_prompts else DEFAULT_PROMPTS

    # ==================== 重写父类方法 ====================

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行 Reflection Agent（重写父类方法）

        Args:
            input_text: 任务描述
            **kwargs: 其他参数

        Returns:
            最终优化后的结果
        """
        print(f"\n🤖 {self.name} 开始处理任务: {input_text}")

        # 重置记忆
        self.memory = Memory()

        # 1. 初始执行
        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = self.prompts["initial"].format(task=input_text)
        initial_result = self._get_llm_response(initial_prompt, **kwargs)
        self.memory.add_record("execution", initial_result)

        # 2. 迭代循环：反思与优化
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")

            # a. 反思
            print("\n-> 正在进行反思...")
            last_result = self.memory.get_last_execution()
            reflect_prompt = self.prompts["reflect"].format(
                task=input_text,
                content=last_result
            )
            feedback = self._get_llm_response(reflect_prompt, **kwargs)
            self.memory.add_record("reflection", feedback)

            # b. 检查是否需要停止
            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                print("\n✅ 反思认为结果已无需改进，任务完成。")
                break

            # c. 优化
            print("\n-> 正在进行优化...")
            refine_prompt = self.prompts["refine"].format(
                task=input_text,
                last_attempt=last_result,
                feedback=feedback
            )
            refined_result = self._get_llm_response(refine_prompt, **kwargs)
            self.memory.add_record("execution", refined_result)

        final_result = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终结果:\n{final_result}")

        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_result, "assistant"))

        return final_result

    # ==================== 私有方法 ====================

    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """
        调用 LLM 并获取完整响应

        Args:
            prompt: 提示词
            **kwargs: LLM 调用参数

        Returns:
            LLM 响应文本
        """
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, **kwargs) or ""
