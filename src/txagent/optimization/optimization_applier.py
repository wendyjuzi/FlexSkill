"""
优化应用器

将优化建议应用到 TxAgent 的参数和策略中。
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class OptimizationApplier:
    """优化应用器"""

    def __init__(self):
        self.applied_optimizations: Dict[str, Dict[str, Any]] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        self.parameter_overrides: Dict[str, Any] = {}

    def apply_optimization(
        self,
        suggestion: Dict[str, Any],
        tx_agent: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        应用单个优化建议

        返回应用结果：
        {
            'suggestion_id': str,
            'applied': bool,
            'timestamp': datetime,
            'parameters_changed': Dict[str, Any],
            'status': str,
            'error': Optional[str],
        }
        """

        suggestion_id = suggestion.get('suggestion_id', '')
        optimization_type = suggestion.get('optimization_type', '')
        parameters = suggestion.get('parameters', {})

        result = {
            'suggestion_id': suggestion_id,
            'applied': False,
            'timestamp': datetime.now().isoformat(),
            'parameters_changed': {},
            'status': 'pending',
            'error': None,
        }

        try:
            # 根据优化类型应用
            if optimization_type == 'loop_detection':
                self._apply_loop_detection(parameters, tx_agent)
                result['parameters_changed'] = parameters

            elif optimization_type == 'tool_selection':
                self._apply_tool_selection(parameters, tx_agent)
                result['parameters_changed'] = parameters

            elif optimization_type == 'prompt':
                self._apply_prompt_optimization(parameters, tx_agent)
                result['parameters_changed'] = parameters

            elif optimization_type == 'error_recovery':
                self._apply_error_recovery(parameters, tx_agent)
                result['parameters_changed'] = parameters

            elif optimization_type == 'rag':
                self._apply_rag_optimization(parameters, tx_agent)
                result['parameters_changed'] = parameters

            elif optimization_type == 'parameter_validation':
                self._apply_parameter_validation(parameters, tx_agent)
                result['parameters_changed'] = parameters

            elif optimization_type == 'error_handling':
                self._apply_error_handling(parameters, tx_agent)
                result['parameters_changed'] = parameters

            else:
                result['error'] = f'Unknown optimization type: {optimization_type}'
                result['status'] = 'failed'
                return result

            # 记录应用
            self.applied_optimizations[suggestion_id] = {
                'suggestion': suggestion,
                'applied_at': datetime.now(),
                'parameters': parameters,
            }

            result['applied'] = True
            result['status'] = 'applied'

        except Exception as e:
            logger.error(f"Error applying optimization {suggestion_id}: {e}")
            result['error'] = str(e)
            result['status'] = 'failed'

        # 记录历史
        self.optimization_history.append(result)

        return result

    def apply_optimizations(
        self,
        suggestions: List[Dict[str, Any]],
        tx_agent: Optional[Any] = None,
        max_concurrent: int = 5,
    ) -> List[Dict[str, Any]]:
        """批量应用优化建议"""
        results = []

        # 按优先级排序
        sorted_suggestions = sorted(
            suggestions,
            key=lambda s: s.get('priority', 0),
            reverse=True
        )

        # 应用前 max_concurrent 个
        for suggestion in sorted_suggestions[:max_concurrent]:
            result = self.apply_optimization(suggestion, tx_agent)
            results.append(result)

        return results

    def _apply_loop_detection(
        self,
        parameters: Dict[str, Any],
        tx_agent: Optional[Any] = None,
    ) -> None:
        """应用循环检测优化"""
        self.parameter_overrides['loop_detection_threshold'] = parameters.get(
            'loop_detection_threshold',
            2
        )
        self.parameter_overrides['loop_check_interval'] = parameters.get(
            'loop_check_interval',
            1
        )

        if tx_agent and hasattr(tx_agent, 'config'):
            tx_agent.config['loop_detection_threshold'] = parameters.get(
                'loop_detection_threshold',
                2
            )

    def _apply_tool_selection(
        self,
        parameters: Dict[str, Any],
        tx_agent: Optional[Any] = None,
    ) -> None:
        """应用工具选择优化"""
        if 'blacklisted_tools' in parameters:
            self.parameter_overrides['blacklisted_tools'] = parameters['blacklisted_tools']

        if 'enable_alternative_chains' in parameters:
            self.parameter_overrides['enable_alternative_chains'] = parameters[
                'enable_alternative_chains'
            ]

        if 'enable_fallback' in parameters:
            self.parameter_overrides['enable_fallback'] = parameters['enable_fallback']

        if tx_agent and hasattr(tx_agent, 'config'):
            tx_agent.config.update(self.parameter_overrides)

    def _apply_prompt_optimization(
        self,
        parameters: Dict[str, Any],
        tx_agent: Optional[Any] = None,
    ) -> None:
        """应用提示词优化"""
        if 'add_repetition_warning' in parameters:
            self.parameter_overrides['add_repetition_warning'] = parameters[
                'add_repetition_warning'
            ]

        if 'max_tool_reuse' in parameters:
            self.parameter_overrides['max_tool_reuse'] = parameters['max_tool_reuse']

        if tx_agent and hasattr(tx_agent, 'system_prompt'):
            # 可以在这里修改系统提示词
            pass

    def _apply_error_recovery(
        self,
        parameters: Dict[str, Any],
        tx_agent: Optional[Any] = None,
    ) -> None:
        """应用错误恢复优化"""
        self.parameter_overrides['enable_recovery'] = parameters.get(
            'enable_recovery',
            True
        )
        self.parameter_overrides['recovery_max_attempts'] = parameters.get(
            'recovery_max_attempts',
            3
        )
        self.parameter_overrides['recovery_backoff'] = parameters.get(
            'recovery_backoff',
            'exponential'
        )

    def _apply_rag_optimization(
        self,
        parameters: Dict[str, Any],
        tx_agent: Optional[Any] = None,
    ) -> None:
        """应用 RAG 优化"""
        self.parameter_overrides['rag_top_k'] = parameters.get('rag_top_k', 5)
        self.parameter_overrides['rag_similarity_threshold'] = parameters.get(
            'rag_similarity_threshold',
            0.7
        )
        self.parameter_overrides['rag_rerank'] = parameters.get('rag_rerank', True)

    def _apply_parameter_validation(
        self,
        parameters: Dict[str, Any],
        tx_agent: Optional[Any] = None,
    ) -> None:
        """应用参数验证优化"""
        self.parameter_overrides['enable_param_validation'] = parameters.get(
            'enable_param_validation',
            True
        )
        self.parameter_overrides['validation_strict'] = parameters.get(
            'validation_strict',
            True
        )
        self.parameter_overrides['enable_type_checking'] = parameters.get(
            'enable_type_checking',
            True
        )
        self.parameter_overrides['auto_convert'] = parameters.get('auto_convert', True)

    def _apply_error_handling(
        self,
        parameters: Dict[str, Any],
        tx_agent: Optional[Any] = None,
    ) -> None:
        """应用错误处理优化"""
        self.parameter_overrides['verbose_error_logging'] = parameters.get(
            'verbose_error_logging',
            True
        )
        self.parameter_overrides['error_context_depth'] = parameters.get(
            'error_context_depth',
            5
        )

    def get_current_parameters(self) -> Dict[str, Any]:
        """获取当前应用的参数"""
        return self.parameter_overrides.copy()

    def rollback_optimization(
        self,
        suggestion_id: str,
    ) -> bool:
        """回滚优化"""
        if suggestion_id not in self.applied_optimizations:
            return False

        optimization = self.applied_optimizations[suggestion_id]
        parameters = optimization['parameters']

        # 移除参数覆盖
        for key in parameters.keys():
            if key in self.parameter_overrides:
                del self.parameter_overrides[key]

        del self.applied_optimizations[suggestion_id]

        logger.info(f"Rolled back optimization: {suggestion_id}")
        return True

    def get_optimization_history(
        self,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """获取优化历史"""
        history = self.optimization_history.copy()

        if limit:
            history = history[-limit:]

        return history

    def get_applied_optimizations(self) -> Dict[str, Dict[str, Any]]:
        """获取已应用的优化"""
        return self.applied_optimizations.copy()

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """获取优化统计"""
        if not self.optimization_history:
            return {}

        applied_count = sum(1 for h in self.optimization_history if h['applied'])
        failed_count = sum(1 for h in self.optimization_history if h['status'] == 'failed')

        optimization_types = defaultdict(int)
        for opt in self.applied_optimizations.values():
            opt_type = opt['suggestion'].get('optimization_type', 'unknown')
            optimization_types[opt_type] += 1

        return {
            'total_attempts': len(self.optimization_history),
            'applied_count': applied_count,
            'failed_count': failed_count,
            'success_rate': applied_count / len(self.optimization_history)
            if self.optimization_history else 0,
            'optimization_types': dict(optimization_types),
            'currently_applied': len(self.applied_optimizations),
        }

    def validate_optimization(
        self,
        suggestion: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """验证优化建议的有效性"""
        required_fields = [
            'suggestion_id',
            'optimization_type',
            'confidence',
            'implementation_effort',
            'risk_level',
        ]

        for field in required_fields:
            if field not in suggestion:
                return False, f"Missing required field: {field}"

        confidence = suggestion.get('confidence', 0)
        if not 0 <= confidence <= 1:
            return False, "Confidence must be between 0 and 1"

        effort_levels = {'low', 'medium', 'high'}
        if suggestion.get('implementation_effort') not in effort_levels:
            return False, "Invalid implementation_effort"

        risk_levels = {'low', 'medium', 'high'}
        if suggestion.get('risk_level') not in risk_levels:
            return False, "Invalid risk_level"

        return True, None


# 类型提示
from typing import Tuple
