"""
OpenAgents 智能体评估模块

本模块提供了完整的智能体评估框架,包括:
- BFCL (Berkeley Function Calling Leaderboard): 工具调用能力评估
- GAIA (General AI Assistants): 通用AI助手能力评估
- Data Generation: 数据生成质量评估（LLM Judge & Win Rate）

主要组件:
- benchmarks: 各种评估基准测试
  - bfcl: BFCL评估（包含专用metrics）
  - gaia: GAIA评估（包含专用metrics）
  - data_generation: 数据生成质量评估

使用示例:
    >>> from open_agents.evaluation import BFCLDataset, BFCLEvaluator
    >>> from open_agents import SimpleAgent
    >>>
    >>> agent = SimpleAgent(name="TestAgent")
    >>> dataset = BFCLDataset(category="simple_python")
    >>> evaluator = BFCLEvaluator(dataset=dataset)
    >>> results = evaluator.evaluate(agent, max_samples=5)
    >>> print(f"准确率: {results['overall_accuracy']:.2%}")
"""

# ==================== 模块公共 API ====================

# 导出benchmark评估器和数据集
from open_agents.evaluation.benchmarks.bfcl.dataset import BFCLDataset
from open_agents.evaluation.benchmarks.bfcl.evaluator import BFCLEvaluator
from open_agents.evaluation.benchmarks.gaia.dataset import GAIADataset
from open_agents.evaluation.benchmarks.gaia.evaluator import GAIAEvaluator
from open_agents.evaluation.benchmarks.data_generation.dataset import AIDataset
from open_agents.evaluation.benchmarks.data_generation.llm_judge import LLMJudgeEvaluator
from open_agents.evaluation.benchmarks.data_generation.win_rate import WinRateEvaluator

# ==================== 模块常量 ====================

__version__ = "0.1.0"

# ==================== 模块公共接口 ====================

__all__ = [
    # Benchmark数据集
    "BFCLDataset",
    "GAIADataset",
    "AIDataset",

    # Benchmark评估器
    "BFCLEvaluator",
    "GAIAEvaluator",
    "LLMJudgeEvaluator",
    "WinRateEvaluator",
]

