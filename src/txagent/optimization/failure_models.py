"""
失败数据模型定义

定义所有失败相关的数据结构，用于捕获、存储和分析失败信息。
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid


class FailureType(Enum):
    """失败类型分类"""
    # 工具相关
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    TOOL_NOT_FOUND = "tool_not_found"
    INVALID_PARAMETERS = "invalid_parameters"
    TOOL_TIMEOUT = "tool_timeout"

    # 推理相关
    INFINITE_LOOP = "infinite_loop"
    REASONING_DEAD_END = "reasoning_dead_end"
    INVALID_TOOL_SELECTION = "invalid_tool_selection"
    CONTEXT_MISMATCH = "context_mismatch"

    # 系统相关
    TOKEN_OVERFLOW = "token_overflow"
    MEMORY_ERROR = "memory_error"
    RATE_LIMIT = "rate_limit"
    EXTERNAL_SERVICE_ERROR = "external_service_error"


class SeverityLevel(Enum):
    """严重程度级别"""
    CRITICAL = 5      # 任务完全失败
    HIGH = 4          # 严重影响结果
    MEDIUM = 3        # 中等影响
    LOW = 2           # 轻微影响
    INFO = 1          # 信息性


@dataclass
class FailureContext:
    """失败发生时的上下文信息"""
    # 推理状态
    reasoning_step: int                # 第几步
    reasoning_depth: int = 0           # 推理深度

    # 工具信息
    tool_name: Optional[str] = None    # 工具名称
    tool_parameters: Dict[str, Any] = field(default_factory=dict)

    # 医疗特定上下文
    medical_domain: str = "general"    # 医疗领域
    case_complexity: str = "medium"    # 案例复杂度

    # 模型状态
    model_temperature: float = 0.3
    model_max_tokens: int = 1024
    rag_top_k: int = 10
    rag_extra_factor: float = 30.0

    # 提示词信息
    system_prompt_version: str = "default"
    prompt_template_version: str = "default"

    # 资源状态
    available_memory: int = 0
    token_usage: int = 0
    elapsed_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class FailureDetails:
    """失败的详细信息"""
    # 错误信息
    error_message: str
    error_code: Optional[str] = None
    error_traceback: Optional[str] = None

    # 失败原因分析
    root_cause: Optional[str] = None   # 根本原因
    contributing_factors: List[str] = field(default_factory=list)

    # 失败特征
    is_repeatable: bool = False        # 是否可重复
    is_systematic: bool = False        # 是否系统性

    # 相关信息
    related_failures: List[str] = field(default_factory=list)
    similar_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class RecoveryAttempt:
    """恢复尝试记录"""
    attempt_number: int
    strategy: str                      # 恢复策略
    parameters_adjusted: Dict[str, Any] = field(default_factory=dict)
    result: str = "pending"            # 恢复结果
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class FailureRecord:
    """失败事件的完整记录"""
    # 标识信息
    failure_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = ""
    task_id: str = ""

    # 失败分类
    failure_type: FailureType = FailureType.TOOL_EXECUTION_ERROR
    severity: SeverityLevel = SeverityLevel.MEDIUM

    # 失败上下文
    context: FailureContext = field(default_factory=lambda: FailureContext(reasoning_step=0))

    # 失败详情
    details: FailureDetails = field(default_factory=lambda: FailureDetails(error_message=""))

    # 恢复信息
    recovery_attempted: bool = False
    recovery_success: bool = False
    recovery_attempts: List[RecoveryAttempt] = field(default_factory=list)

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 JSON 序列化）"""
        return {
            'failure_id': self.failure_id,
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id,
            'task_id': self.task_id,
            'failure_type': self.failure_type.value,
            'severity': self.severity.value,
            'context': self.context.to_dict(),
            'details': self.details.to_dict(),
            'recovery_attempted': self.recovery_attempted,
            'recovery_success': self.recovery_success,
            'recovery_attempts': [r.to_dict() for r in self.recovery_attempts],
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FailureRecord':
        """从字典创建实例（用于 JSON 反序列化）"""
        data = data.copy()

        # 转换时间戳
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])

        # 转换枚举
        data['failure_type'] = FailureType(data['failure_type'])
        data['severity'] = SeverityLevel(data['severity'])

        # 转换上下文
        context_data = data.pop('context')
        data['context'] = FailureContext(**context_data)

        # 转换详情
        details_data = data.pop('details')
        data['details'] = FailureDetails(**details_data)

        # 转换恢复尝试
        recovery_attempts_data = data.pop('recovery_attempts', [])
        recovery_attempts = []
        for attempt_data in recovery_attempts_data:
            if isinstance(attempt_data['timestamp'], str):
                attempt_data['timestamp'] = datetime.fromisoformat(attempt_data['timestamp'])
            recovery_attempts.append(RecoveryAttempt(**attempt_data))
        data['recovery_attempts'] = recovery_attempts

        return cls(**data)


@dataclass
class FailurePattern:
    """失败模式识别结果"""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: str = "unknown"      # "infinite_loop", "tool_chain_failure" 等

    # 模式特征
    trigger_conditions: List[str] = field(default_factory=list)
    failure_sequence: List[str] = field(default_factory=list)
    frequency: int = 0

    # 统计信息
    total_occurrences: int = 0
    success_rate_after: float = 0.0    # 后续成功率
    average_recovery_time: float = 0.0

    # 影响范围
    affected_tools: List[str] = field(default_factory=list)
    affected_domains: List[str] = field(default_factory=list)
    affected_complexity_levels: List[str] = field(default_factory=list)

    # 优化建议
    recommended_optimizations: List[str] = field(default_factory=list)

    # 时间信息
    first_occurrence: datetime = field(default_factory=datetime.now)
    last_occurrence: datetime = field(default_factory=datetime.now)
    trend: str = "stable"              # "increasing"/"decreasing"/"stable"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'trigger_conditions': self.trigger_conditions,
            'failure_sequence': self.failure_sequence,
            'frequency': self.frequency,
            'total_occurrences': self.total_occurrences,
            'success_rate_after': self.success_rate_after,
            'average_recovery_time': self.average_recovery_time,
            'affected_tools': self.affected_tools,
            'affected_domains': self.affected_domains,
            'affected_complexity_levels': self.affected_complexity_levels,
            'recommended_optimizations': self.recommended_optimizations,
            'first_occurrence': self.first_occurrence.isoformat(),
            'last_occurrence': self.last_occurrence.isoformat(),
            'trend': self.trend,
        }


@dataclass
class OptimizationSuggestion:
    """优化建议"""
    suggestion_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_id: str = ""               # 关联的失败模式

    # 建议内容
    optimization_type: str = "unknown" # "tool_selection"/"rag"/"prompt"/"loop_detection"
    description: str = ""

    # 预期效果
    expected_improvement: float = 0.0  # 预期改进百分比
    confidence: float = 0.0            # 建议置信度 (0-1)

    # 实施信息
    implementation_effort: str = "medium"  # "low"/"medium"/"high"
    risk_level: str = "low"            # "low"/"medium"/"high"

    # 优先级
    priority: int = 0

    # 验证信息
    validation_status: str = "pending" # "pending"/"validated"/"rejected"
    validation_results: Dict[str, Any] = field(default_factory=dict)

    # 应用信息
    applied_count: int = 0
    success_rate: float = 0.0
    last_applied: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'suggestion_id': self.suggestion_id,
            'pattern_id': self.pattern_id,
            'optimization_type': self.optimization_type,
            'description': self.description,
            'expected_improvement': self.expected_improvement,
            'confidence': self.confidence,
            'implementation_effort': self.implementation_effort,
            'risk_level': self.risk_level,
            'priority': self.priority,
            'validation_status': self.validation_status,
            'validation_results': self.validation_results,
            'applied_count': self.applied_count,
            'success_rate': self.success_rate,
            'last_applied': self.last_applied.isoformat() if self.last_applied else None,
        }
