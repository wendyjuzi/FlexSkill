"""
模式识别引擎

识别失败模式，包括循环、级联、链式失败等。
"""

from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import logging

from .failure_models import FailureRecord, FailurePattern

logger = logging.getLogger(__name__)


class PatternRecognizer:
    """模式识别引擎"""

    def __init__(self, min_pattern_frequency: int = 3):
        self.min_pattern_frequency = min_pattern_frequency

    def recognize_patterns(
        self,
        failure_records: List[FailureRecord],
        time_window_days: int = 7,
    ) -> List[Dict[str, Any]]:
        """
        识别失败模式

        返回模式列表，每个模式包含：
        {
            'pattern_id': str,
            'pattern_type': str,
            'trigger_conditions': List[str],
            'frequency': int,
            'affected_tools': List[str],
            'affected_domains': List[str],
            'success_rate_after': float,
            'recommended_optimizations': List[str],
        }
        """

        if not failure_records:
            return []

        # 按时间窗口过滤
        cutoff_time = datetime.now() - timedelta(days=time_window_days)
        recent_failures = [
            f for f in failure_records
            if f.timestamp > cutoff_time
        ]

        patterns = []

        # 识别循环模式
        loop_patterns = self._detect_loop_patterns(recent_failures)
        patterns.extend(loop_patterns)

        # 识别级联模式
        cascade_patterns = self._detect_cascade_patterns(recent_failures)
        patterns.extend(cascade_patterns)

        # 识别工具链失败模式
        chain_patterns = self._detect_chain_patterns(recent_failures)
        patterns.extend(chain_patterns)

        # 识别参数不匹配模式
        param_patterns = self._detect_parameter_mismatch_patterns(recent_failures)
        patterns.extend(param_patterns)

        return patterns

    def _detect_loop_patterns(
        self,
        failure_records: List[FailureRecord],
    ) -> List[Dict[str, Any]]:
        """检测循环模式"""
        patterns = []
        loop_sequences = defaultdict(int)

        # 按会话分组
        sessions = defaultdict(list)
        for failure in failure_records:
            sessions[failure.session_id].append(failure)

        for session_id, failures in sessions.items():
            # 按时间排序
            failures.sort(key=lambda f: f.timestamp)

            # 提取工具序列
            tool_sequence = [
                f.context.tool_name
                for f in failures
                if f.context.tool_name
            ]

            # 检测重复模式
            for i in range(len(tool_sequence) - 1):
                for j in range(i + 1, len(tool_sequence)):
                    if tool_sequence[i] == tool_sequence[j]:
                        # 找到重复
                        repeat_pattern = tuple(tool_sequence[i:j])
                        if len(repeat_pattern) >= 2:
                            loop_sequences[repeat_pattern] += 1

        # 生成循环模式
        for pattern, frequency in loop_sequences.items():
            if frequency >= self.min_pattern_frequency:
                patterns.append({
                    'pattern_id': f"loop_{hash(pattern)}",
                    'pattern_type': 'infinite_loop',
                    'trigger_conditions': [f'Tool sequence: {" -> ".join(pattern)}'],
                    'frequency': frequency,
                    'affected_tools': list(set(pattern)),
                    'affected_domains': [],
                    'success_rate_after': 0.0,
                    'recommended_optimizations': [
                        'Increase loop detection sensitivity',
                        'Add tool blacklist',
                        'Optimize prompt to avoid repetition',
                    ],
                })

        return patterns

    def _detect_cascade_patterns(
        self,
        failure_records: List[FailureRecord],
    ) -> List[Dict[str, Any]]:
        """检测级联失败模式"""
        patterns = []
        cascade_chains = defaultdict(int)

        # 按会话分组
        sessions = defaultdict(list)
        for failure in failure_records:
            sessions[failure.session_id].append(failure)

        for session_id, failures in sessions.items():
            # 按时间排序
            failures.sort(key=lambda f: f.timestamp)

            # 检测因果关系
            for i in range(len(failures) - 1):
                current = failures[i]
                next_failure = failures[i + 1]

                # 检查是否相关
                if self._are_failures_related(current, next_failure):
                    chain = (
                        current.failure_type.value,
                        next_failure.failure_type.value
                    )
                    cascade_chains[chain] += 1

        # 生成级联模式
        for chain, frequency in cascade_chains.items():
            if frequency >= self.min_pattern_frequency:
                patterns.append({
                    'pattern_id': f"cascade_{hash(chain)}",
                    'pattern_type': 'cascading_failure',
                    'trigger_conditions': [f'Failure chain: {chain[0]} -> {chain[1]}'],
                    'frequency': frequency,
                    'affected_tools': [],
                    'affected_domains': [],
                    'success_rate_after': 0.0,
                    'recommended_optimizations': [
                        'Add error recovery mechanism',
                        'Implement fallback strategies',
                        'Improve error handling',
                    ],
                })

        return patterns

    def _detect_chain_patterns(
        self,
        failure_records: List[FailureRecord],
    ) -> List[Dict[str, Any]]:
        """检测工具链失败模式"""
        patterns = []
        tool_chains = defaultdict(int)

        # 按会话分组
        sessions = defaultdict(list)
        for failure in failure_records:
            sessions[failure.session_id].append(failure)

        for session_id, failures in sessions.items():
            # 按时间排序
            failures.sort(key=lambda f: f.timestamp)

            # 提取工具序列
            tool_sequence = [
                f.context.tool_name
                for f in failures
                if f.context.tool_name
            ]

            # 检测连续失败的工具
            for i in range(len(tool_sequence) - 2):
                chain = tuple(tool_sequence[i:i+3])
                tool_chains[chain] += 1

        # 生成工具链模式
        for chain, frequency in tool_chains.items():
            if frequency >= self.min_pattern_frequency:
                patterns.append({
                    'pattern_id': f"chain_{hash(chain)}",
                    'pattern_type': 'tool_chain_failure',
                    'trigger_conditions': [f'Tool chain: {" -> ".join(chain)}'],
                    'frequency': frequency,
                    'affected_tools': list(set(chain)),
                    'affected_domains': [],
                    'success_rate_after': 0.0,
                    'recommended_optimizations': [
                        'Adjust RAG parameters for better tool selection',
                        'Add alternative tool chains',
                        'Improve tool compatibility checks',
                    ],
                })

        return patterns

    def _detect_parameter_mismatch_patterns(
        self,
        failure_records: List[FailureRecord],
    ) -> List[Dict[str, Any]]:
        """检测参数不匹配模式"""
        patterns = []
        param_mismatches = defaultdict(int)

        for failure in failure_records:
            if 'parameter' in failure.details.error_message.lower():
                tool_name = failure.context.tool_name or 'unknown'
                param_mismatches[tool_name] += 1

        # 生成参数不匹配模式
        for tool_name, frequency in param_mismatches.items():
            if frequency >= self.min_pattern_frequency:
                patterns.append({
                    'pattern_id': f"param_{hash(tool_name)}",
                    'pattern_type': 'parameter_mismatch',
                    'trigger_conditions': [f'Tool: {tool_name}'],
                    'frequency': frequency,
                    'affected_tools': [tool_name],
                    'affected_domains': [],
                    'success_rate_after': 0.0,
                    'recommended_optimizations': [
                        f'Validate parameters for {tool_name}',
                        'Add parameter type checking',
                        'Improve tool documentation',
                    ],
                })

        return patterns

    def _are_failures_related(
        self,
        failure1: FailureRecord,
        failure2: FailureRecord,
    ) -> bool:
        """检查两个失败是否相关"""
        # 同一工具
        if failure1.context.tool_name == failure2.context.tool_name:
            return True

        # 同一医疗领域
        if failure1.context.medical_domain == failure2.context.medical_domain:
            return True

        # 时间接近（5 分钟内）
        time_diff = (failure2.timestamp - failure1.timestamp).total_seconds()
        if 0 < time_diff < 300:
            return True

        return False

    def get_pattern_statistics(
        self,
        patterns: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """获取模式统计"""
        if not patterns:
            return {}

        pattern_types = Counter(p['pattern_type'] for p in patterns)
        total_frequency = sum(p['frequency'] for p in patterns)

        return {
            'total_patterns': len(patterns),
            'pattern_types': dict(pattern_types),
            'total_frequency': total_frequency,
            'average_frequency': total_frequency / len(patterns) if patterns else 0,
            'top_patterns': sorted(
                patterns,
                key=lambda p: p['frequency'],
                reverse=True
            )[:5],
        }
