"""
Benchmarks 模块

包含各种智能体评估基准测试:
- BFCL: Berkeley Function Calling Leaderboard
- GAIA: General AI Assistants Benchmark
- Data Generation: 数据生成质量评估
"""

# ==================== 模块公共 API ====================

from open_agents.evaluation.benchmarks.bfcl.evaluator import BFCLEvaluator
from open_agents.evaluation.benchmarks.gaia.evaluator import GAIAEvaluator
from open_agents.evaluation.benchmarks.data_generation.llm_judge import LLMJudgeEvaluator
from open_agents.evaluation.benchmarks.data_generation.win_rate import WinRateEvaluator

# ==================== 模块公共接口 ====================

__all__ = [
    "BFCLEvaluator",
    "GAIAEvaluator",
    "LLMJudgeEvaluator",
    "WinRateEvaluator",
]

