"""
TxAgent 失败经验优化系统

基于失败经验的 Agent Skills 优化系统，包括：
- 失败捕获和存储
- 失败分析和模式识别
- 优化建议生成
- 在线自适应优化应用
"""

from .plugin import PluginManager, OptimizationPlugin
from .failure_models import (
    FailureRecord,
    FailureContext,
    FailureDetails,
    RecoveryAttempt,
    FailureType,
    SeverityLevel,
    FailurePattern,
    OptimizationSuggestion,
)
from .failure_capture import FailureCapture
from .failure_store import FailureStore
from .failure_analyzer import FailureAnalyzer
from .pattern_recognizer import PatternRecognizer
from .optimization_generator import OptimizationGenerator
from .optimization_applier import OptimizationApplier
from .optimization_engine import OptimizationEngine, OptimizationCache
from .optimization_plugin import TxAgentOptimizationPlugin
from .optimization_monitor import (
    OptimizationMonitor,
    FeedbackLoop,
    OptimizationDashboard,
)

__all__ = [
    "PluginManager",
    "OptimizationPlugin",
    "FailureRecord",
    "FailureContext",
    "FailureDetails",
    "RecoveryAttempt",
    "FailureType",
    "SeverityLevel",
    "FailurePattern",
    "OptimizationSuggestion",
    "FailureCapture",
    "FailureStore",
    "FailureAnalyzer",
    "PatternRecognizer",
    "OptimizationGenerator",
    "OptimizationApplier",
    "OptimizationEngine",
    "OptimizationCache",
    "TxAgentOptimizationPlugin",
    "OptimizationMonitor",
    "FeedbackLoop",
    "OptimizationDashboard",
]
