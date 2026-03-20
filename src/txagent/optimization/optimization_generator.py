"""
优化建议生成器

基于失败模式和分析结果生成优化建议。
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from collections import defaultdict

from .failure_models import FailureRecord, FailurePattern, OptimizationSuggestion

logger = logging.getLogger(__name__)


class OptimizationGenerator:
    """优化建议生成器"""

    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
        self.suggestion_cache: Dict[str, OptimizationSuggestion] = {}

    def generate_suggestions(
        self,
        patterns: List[Dict[str, Any]],
        failure_records: List[FailureRecord],
    ) -> List[Dict[str, Any]]:
        """
        生成优化建议

        返回建议列表，每个建议包含：
        {
            'suggestion_id': str,
            'pattern_id': str,
            'optimization_type': str,
            'description': str,
            'expected_improvement': float,
            'confidence': float,
            'implementation_effort': str,
            'risk_level': str,
            'priority': int,
            'affected_tools': List[str],
            'affected_domains': List[str],
        }
        """

        if not patterns:
            return []

        suggestions = []

        for pattern in patterns:
            pattern_suggestions = self._generate_for_pattern(
                pattern,
                failure_records
            )
            suggestions.extend(pattern_suggestions)

        # 按优先级排序
        suggestions.sort(key=lambda s: s['priority'], reverse=True)

        return suggestions

    def _generate_for_pattern(
        self,
        pattern: Dict[str, Any],
        failure_records: List[FailureRecord],
    ) -> List[Dict[str, Any]]:
        """为单个模式生成优化建议"""
        suggestions = []
        pattern_type = pattern.get('pattern_type', '')

        if pattern_type == 'infinite_loop':
            suggestions.extend(self._generate_loop_optimizations(pattern))

        elif pattern_type == 'cascading_failure':
            suggestions.extend(self._generate_cascade_optimizations(pattern))

        elif pattern_type == 'tool_chain_failure':
            suggestions.extend(self._generate_chain_optimizations(pattern))

        elif pattern_type == 'parameter_mismatch':
            suggestions.extend(self._generate_parameter_optimizations(pattern))

        return suggestions

    def _generate_loop_optimizations(
        self,
        pattern: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """生成循环模式的优化建议"""
        suggestions = []
        pattern_id = pattern.get('pattern_id', '')
        frequency = pattern.get('frequency', 0)
        affected_tools = pattern.get('affected_tools', [])

        # 建议1: 增加循环检测灵敏度
        suggestions.append({
            'suggestion_id': f"loop_detect_{pattern_id}",
            'pattern_id': pattern_id,
            'optimization_type': 'loop_detection',
            'description': 'Increase loop detection sensitivity to catch repetitions earlier',
            'expected_improvement': 0.15,
            'confidence': 0.85,
            'implementation_effort': 'low',
            'risk_level': 'low',
            'priority': 90,
            'affected_tools': affected_tools,
            'affected_domains': pattern.get('affected_domains', []),
            'parameters': {
                'loop_detection_threshold': 2,  # 从 3 降低到 2
                'loop_check_interval': 1,
            }
        })

        # 建议2: 添加工具黑名单
        suggestions.append({
            'suggestion_id': f"tool_blacklist_{pattern_id}",
            'pattern_id': pattern_id,
            'optimization_type': 'tool_selection',
            'description': f'Add tool blacklist for problematic tools: {", ".join(affected_tools[:3])}',
            'expected_improvement': 0.20,
            'confidence': 0.75,
            'implementation_effort': 'medium',
            'risk_level': 'medium',
            'priority': 85,
            'affected_tools': affected_tools,
            'affected_domains': pattern.get('affected_domains', []),
            'parameters': {
                'blacklisted_tools': affected_tools,
                'blacklist_duration': 300,  # 5 分钟
            }
        })

        # 建议3: 优化提示词避免重复
        suggestions.append({
            'suggestion_id': f"prompt_opt_{pattern_id}",
            'pattern_id': pattern_id,
            'optimization_type': 'prompt',
            'description': 'Optimize system prompt to avoid tool repetition',
            'expected_improvement': 0.10,
            'confidence': 0.70,
            'implementation_effort': 'high',
            'risk_level': 'medium',
            'priority': 75,
            'affected_tools': affected_tools,
            'affected_domains': pattern.get('affected_domains', []),
            'parameters': {
                'add_repetition_warning': True,
                'max_tool_reuse': 1,
            }
        })

        return suggestions

    def _generate_cascade_optimizations(
        self,
        pattern: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """生成级联失败模式的优化建议"""
        suggestions = []
        pattern_id = pattern.get('pattern_id', '')

        # 建议1: 添加错误恢复机制
        suggestions.append({
            'suggestion_id': f"recovery_{pattern_id}",
            'pattern_id': pattern_id,
            'optimization_type': 'error_recovery',
            'description': 'Add error recovery mechanism to prevent cascade',
            'expected_improvement': 0.25,
            'confidence': 0.80,
            'implementation_effort': 'medium',
            'risk_level': 'low',
            'priority': 88,
            'affected_tools': pattern.get('affected_tools', []),
            'affected_domains': pattern.get('affected_domains', []),
            'parameters': {
                'enable_recovery': True,
                'recovery_max_attempts': 3,
                'recovery_backoff': 'exponential',
            }
        })

        # 建议2: 实现备选策略
        suggestions.append({
            'suggestion_id': f"fallback_{pattern_id}",
            'pattern_id': pattern_id,
            'optimization_type': 'tool_selection',
            'description': 'Implement fallback strategies for failed tools',
            'expected_improvement': 0.18,
            'confidence': 0.75,
            'implementation_effort': 'medium',
            'risk_level': 'medium',
            'priority': 82,
            'affected_tools': pattern.get('affected_tools', []),
            'affected_domains': pattern.get('affected_domains', []),
            'parameters': {
                'enable_fallback': True,
                'fallback_strategy': 'alternative_tools',
            }
        })

        # 建议3: 改进错误处理
        suggestions.append({
            'suggestion_id': f"error_handling_{pattern_id}",
            'pattern_id': pattern_id,
            'optimization_type': 'error_handling',
            'description': 'Improve error handling and logging',
            'expected_improvement': 0.12,
            'confidence': 0.70,
            'implementation_effort': 'low',
            'risk_level': 'low',
            'priority': 70,
            'affected_tools': pattern.get('affected_tools', []),
            'affected_domains': pattern.get('affected_domains', []),
            'parameters': {
                'verbose_error_logging': True,
                'error_context_depth': 5,
            }
        })

        return suggestions

    def _generate_chain_optimizations(
        self,
        pattern: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """生成工具链失败模式的优化建议"""
        suggestions = []
        pattern_id = pattern.get('pattern_id', '')
        affected_tools = pattern.get('affected_tools', [])

        # 建议1: 调整 RAG 参数
        suggestions.append({
            'suggestion_id': f"rag_opt_{pattern_id}",
            'pattern_id': pattern_id,
            'optimization_type': 'rag',
            'description': 'Adjust RAG parameters for better tool selection',
            'expected_improvement': 0.20,
            'confidence': 0.78,
            'implementation_effort': 'medium',
            'risk_level': 'low',
            'priority': 85,
            'affected_tools': affected_tools,
            'affected_domains': pattern.get('affected_domains', []),
            'parameters': {
                'rag_top_k': 5,
                'rag_similarity_threshold': 0.7,
                'rag_rerank': True,
            }
        })

        # 建议2: 添加备选工具链
        suggestions.append({
            'suggestion_id': f"alt_chain_{pattern_id}",
            'pattern_id': pattern_id,
            'optimization_type': 'tool_selection',
            'description': f'Add alternative tool chains for {", ".join(affected_tools[:2])}',
            'expected_improvement': 0.15,
            'confidence': 0.72,
            'implementation_effort': 'high',
            'risk_level': 'medium',
            'priority': 80,
            'affected_tools': affected_tools,
            'affected_domains': pattern.get('affected_domains', []),
            'parameters': {
                'enable_alternative_chains': True,
                'chain_diversity': 'high',
            }
        })

        # 建议3: 改进工具兼容性检查
        suggestions.append({
            'suggestion_id': f"compat_check_{pattern_id}",
            'pattern_id': pattern_id,
            'optimization_type': 'tool_selection',
            'description': 'Improve tool compatibility checks',
            'expected_improvement': 0.10,
            'confidence': 0.68,
            'implementation_effort': 'medium',
            'risk_level': 'low',
            'priority': 75,
            'affected_tools': affected_tools,
            'affected_domains': pattern.get('affected_domains', []),
            'parameters': {
                'enable_compatibility_check': True,
                'check_output_format': True,
            }
        })

        return suggestions

    def _generate_parameter_optimizations(
        self,
        pattern: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """生成参数不匹配模式的优化建议"""
        suggestions = []
        pattern_id = pattern.get('pattern_id', '')
        affected_tools = pattern.get('affected_tools', [])

        # 建议1: 参数验证
        suggestions.append({
            'suggestion_id': f"param_validate_{pattern_id}",
            'pattern_id': pattern_id,
            'optimization_type': 'parameter_validation',
            'description': f'Add parameter validation for {", ".join(affected_tools)}',
            'expected_improvement': 0.30,
            'confidence': 0.90,
            'implementation_effort': 'low',
            'risk_level': 'low',
            'priority': 95,
            'affected_tools': affected_tools,
            'affected_domains': pattern.get('affected_domains', []),
            'parameters': {
                'enable_param_validation': True,
                'validation_strict': True,
            }
        })

        # 建议2: 参数类型检查
        suggestions.append({
            'suggestion_id': f"param_type_{pattern_id}",
            'pattern_id': pattern_id,
            'optimization_type': 'parameter_validation',
            'description': 'Add parameter type checking',
            'expected_improvement': 0.25,
            'confidence': 0.85,
            'implementation_effort': 'low',
            'risk_level': 'low',
            'priority': 92,
            'affected_tools': affected_tools,
            'affected_domains': pattern.get('affected_domains', []),
            'parameters': {
                'enable_type_checking': True,
                'auto_convert': True,
            }
        })

        # 建议3: 改进工具文档
        suggestions.append({
            'suggestion_id': f"doc_improve_{pattern_id}",
            'pattern_id': pattern_id,
            'optimization_type': 'documentation',
            'description': 'Improve tool documentation and examples',
            'expected_improvement': 0.15,
            'confidence': 0.70,
            'implementation_effort': 'high',
            'risk_level': 'low',
            'priority': 70,
            'affected_tools': affected_tools,
            'affected_domains': pattern.get('affected_domains', []),
            'parameters': {
                'add_examples': True,
                'add_constraints': True,
            }
        })

        return suggestions

    def prioritize_suggestions(
        self,
        suggestions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        优先级排序

        考虑因素：
        - 预期改进
        - 置信度
        - 实现难度
        - 风险等级
        """

        def calculate_priority_score(s: Dict[str, Any]) -> float:
            improvement = s.get('expected_improvement', 0)
            confidence = s.get('confidence', 0)
            effort_map = {'low': 1.0, 'medium': 0.7, 'high': 0.4}
            effort = effort_map.get(s.get('implementation_effort', 'medium'), 0.7)
            risk_map = {'low': 1.0, 'medium': 0.8, 'high': 0.5}
            risk = risk_map.get(s.get('risk_level', 'medium'), 0.8)

            # 综合评分
            score = (improvement * 0.4 + confidence * 0.3 + effort * 0.2 + risk * 0.1)
            return score

        # 计算优先级分数
        for suggestion in suggestions:
            suggestion['priority_score'] = calculate_priority_score(suggestion)

        # 按分数排序
        suggestions.sort(key=lambda s: s['priority_score'], reverse=True)

        # 重新分配优先级（1-100）
        for i, suggestion in enumerate(suggestions):
            suggestion['priority'] = max(1, 100 - i * 5)

        return suggestions

    def filter_suggestions(
        self,
        suggestions: List[Dict[str, Any]],
        min_confidence: Optional[float] = None,
        max_effort: Optional[str] = None,
        max_risk: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        过滤建议

        参数：
        - min_confidence: 最小置信度
        - max_effort: 最大实现难度
        - max_risk: 最大风险等级
        """

        if min_confidence is None:
            min_confidence = self.min_confidence

        effort_levels = {'low': 1, 'medium': 2, 'high': 3}
        risk_levels = {'low': 1, 'medium': 2, 'high': 3}

        max_effort_level = effort_levels.get(max_effort, 3) if max_effort else 3
        max_risk_level = risk_levels.get(max_risk, 3) if max_risk else 3

        filtered = []
        for suggestion in suggestions:
            if suggestion.get('confidence', 0) < min_confidence:
                continue

            effort_level = effort_levels.get(
                suggestion.get('implementation_effort', 'medium'),
                2
            )
            if effort_level > max_effort_level:
                continue

            risk_level = risk_levels.get(
                suggestion.get('risk_level', 'medium'),
                2
            )
            if risk_level > max_risk_level:
                continue

            filtered.append(suggestion)

        return filtered

    def get_suggestion_statistics(
        self,
        suggestions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """获取建议统计"""
        if not suggestions:
            return {}

        optimization_types = defaultdict(int)
        effort_distribution = defaultdict(int)
        risk_distribution = defaultdict(int)
        total_improvement = 0.0
        avg_confidence = 0.0

        for suggestion in suggestions:
            optimization_types[suggestion.get('optimization_type', 'unknown')] += 1
            effort_distribution[suggestion.get('implementation_effort', 'unknown')] += 1
            risk_distribution[suggestion.get('risk_level', 'unknown')] += 1
            total_improvement += suggestion.get('expected_improvement', 0)
            avg_confidence += suggestion.get('confidence', 0)

        return {
            'total_suggestions': len(suggestions),
            'optimization_types': dict(optimization_types),
            'effort_distribution': dict(effort_distribution),
            'risk_distribution': dict(risk_distribution),
            'total_expected_improvement': total_improvement,
            'average_confidence': avg_confidence / len(suggestions) if suggestions else 0,
            'high_priority_count': sum(1 for s in suggestions if s.get('priority', 0) >= 80),
            'low_risk_count': sum(1 for s in suggestions if s.get('risk_level') == 'low'),
        }
