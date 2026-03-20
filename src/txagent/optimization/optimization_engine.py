"""
优化引擎

核心优化引擎，实现在线自适应优化和实时缓存系统。
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
import logging
import hashlib

from .failure_store import FailureStore
from .failure_analyzer import FailureAnalyzer
from .pattern_recognizer import PatternRecognizer
from .optimization_generator import OptimizationGenerator
from .optimization_applier import OptimizationApplier

logger = logging.getLogger(__name__)


class OptimizationCache:
    """LRU 缓存系统"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def put(self, key: str, value: Dict[str, Any]) -> None:
        """添加到缓存"""
        if key in self.cache:
            # 移到末尾（最近使用）
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # 移除最旧的项
                self.cache.popitem(last=False)

        self.cache[key] = {
            'value': value,
            'timestamp': datetime.now(),
        }

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """从缓存获取"""
        if key not in self.cache:
            return None

        entry = self.cache[key]
        age = (datetime.now() - entry['timestamp']).total_seconds()

        if age > self.ttl_seconds:
            # 过期
            del self.cache[key]
            return None

        # 移到末尾（最近使用）
        self.cache.move_to_end(key)
        return entry['value']

    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
        }


class OptimizationEngine:
    """优化引擎"""

    def __init__(
        self,
        failure_store: FailureStore,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
    ):
        self.failure_store = failure_store
        self.analyzer = FailureAnalyzer()
        self.recognizer = PatternRecognizer()
        self.generator = OptimizationGenerator()
        self.applier = OptimizationApplier()

        self.cache = OptimizationCache(max_size=cache_size, ttl_seconds=cache_ttl)
        self.optimization_effects: Dict[str, List[float]] = {}

    def process_failure(
        self,
        failure_record: Any,
        tx_agent: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        处理单个失败事件

        流程：
        1. 分析失败
        2. 识别模式
        3. 生成优化
        4. 应用高置信度优化
        5. 缓存中等置信度优化
        """

        result = {
            'failure_id': failure_record.failure_id,
            'timestamp': datetime.now().isoformat(),
            'analysis': None,
            'patterns': [],
            'suggestions': [],
            'applied': [],
            'cached': [],
            'status': 'processing',
        }

        try:
            # 步骤1: 分析失败
            historical_failures = self.failure_store.load_all_failures()
            analysis = self.analyzer.analyze_failure(
                failure_record,
                historical_failures
            )
            result['analysis'] = analysis

            # 步骤2: 识别模式
            patterns = self.recognizer.recognize_patterns(historical_failures)
            result['patterns'] = patterns

            # 步骤3: 生成优化
            suggestions = self.generator.generate_suggestions(
                patterns,
                historical_failures
            )
            suggestions = self.generator.prioritize_suggestions(suggestions)
            result['suggestions'] = suggestions

            # 步骤4: 应用高置信度优化（> 0.85）
            high_confidence = [s for s in suggestions if s.get('confidence', 0) > 0.85]
            for suggestion in high_confidence:
                apply_result = self.applier.apply_optimization(suggestion, tx_agent)
                result['applied'].append(apply_result)
                self._track_optimization_effect(suggestion['suggestion_id'], 1.0)

            # 步骤5: 缓存中等置信度优化（0.7-0.85）
            medium_confidence = [
                s for s in suggestions
                if 0.7 <= s.get('confidence', 0) <= 0.85
            ]
            for suggestion in medium_confidence:
                cache_key = self._generate_cache_key(suggestion)
                self.cache.put(cache_key, suggestion)
                result['cached'].append({
                    'suggestion_id': suggestion['suggestion_id'],
                    'confidence': suggestion['confidence'],
                    'cached_at': datetime.now().isoformat(),
                })

            result['status'] = 'completed'

        except Exception as e:
            logger.error(f"Error processing failure {failure_record.failure_id}: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)

        return result

    def batch_process_failures(
        self,
        failure_records: List[Any],
        tx_agent: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """批量处理失败"""
        results = []
        for failure in failure_records:
            result = self.process_failure(failure, tx_agent)
            results.append(result)

        return results

    def get_cached_optimizations(
        self,
        pattern_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """获取缓存的优化"""
        cached = []

        for key, entry in self.cache.cache.items():
            suggestion = entry['value']

            if pattern_type and suggestion.get('pattern_type') != pattern_type:
                continue

            cached.append({
                'suggestion_id': suggestion['suggestion_id'],
                'optimization_type': suggestion['optimization_type'],
                'confidence': suggestion['confidence'],
                'cached_at': entry['timestamp'].isoformat(),
            })

        return cached

    def apply_cached_optimization(
        self,
        suggestion_id: str,
        tx_agent: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """应用缓存的优化"""
        # 从缓存查找
        for key, entry in self.cache.cache.items():
            suggestion = entry['value']
            if suggestion['suggestion_id'] == suggestion_id:
                result = self.applier.apply_optimization(suggestion, tx_agent)
                self._track_optimization_effect(suggestion_id, 1.0)
                return result

        return {
            'suggestion_id': suggestion_id,
            'applied': False,
            'error': 'Suggestion not found in cache',
        }

    def _generate_cache_key(self, suggestion: Dict[str, Any]) -> str:
        """生成缓存键"""
        key_str = f"{suggestion['pattern_id']}_{suggestion['optimization_type']}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _track_optimization_effect(
        self,
        suggestion_id: str,
        effect: float,
    ) -> None:
        """跟踪优化效果"""
        if suggestion_id not in self.optimization_effects:
            self.optimization_effects[suggestion_id] = []

        self.optimization_effects[suggestion_id].append(effect)

    def get_optimization_effectiveness(
        self,
        suggestion_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """获取优化效果"""
        if suggestion_id:
            effects = self.optimization_effects.get(suggestion_id, [])
            if not effects:
                return {'suggestion_id': suggestion_id, 'effectiveness': 0.0}

            avg_effect = sum(effects) / len(effects)
            return {
                'suggestion_id': suggestion_id,
                'effectiveness': avg_effect,
                'attempts': len(effects),
            }

        # 所有优化的效果
        effectiveness = {}
        for sid, effects in self.optimization_effects.items():
            if effects:
                effectiveness[sid] = sum(effects) / len(effects)

        return effectiveness

    def get_engine_statistics(self) -> Dict[str, Any]:
        """获取引擎统计"""
        return {
            'cache_stats': self.cache.get_stats(),
            'applier_stats': self.applier.get_optimization_statistics(),
            'optimization_effects': self.get_optimization_effectiveness(),
            'cached_optimizations': len(self.cache.cache),
            'applied_optimizations': len(self.applier.applied_optimizations),
        }

    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
        logger.info("Optimization cache cleared")

    def reset_engine(self) -> None:
        """重置引擎"""
        self.cache.clear()
        self.applier.applied_optimizations.clear()
        self.applier.optimization_history.clear()
        self.applier.parameter_overrides.clear()
        self.optimization_effects.clear()
        logger.info("Optimization engine reset")

    def get_optimization_recommendations(
        self,
        min_confidence: float = 0.7,
        max_effort: str = 'medium',
        max_risk: str = 'medium',
    ) -> List[Dict[str, Any]]:
        """
        获取优化建议

        基于当前缓存和应用的优化
        """

        # 从缓存获取
        cached_suggestions = []
        for key, entry in self.cache.cache.items():
            suggestion = entry['value']
            cached_suggestions.append(suggestion)

        # 过滤
        filtered = self.generator.filter_suggestions(
            cached_suggestions,
            min_confidence=min_confidence,
            max_effort=max_effort,
            max_risk=max_risk,
        )

        return filtered

    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化总结"""
        stats = self.get_engine_statistics()
        applier_stats = stats['applier_stats']

        return {
            'total_optimizations_attempted': applier_stats.get('total_attempts', 0),
            'total_optimizations_applied': applier_stats.get('applied_count', 0),
            'success_rate': applier_stats.get('success_rate', 0),
            'currently_cached': stats['cached_optimizations'],
            'currently_applied': stats['applied_optimizations'],
            'optimization_types': applier_stats.get('optimization_types', {}),
            'average_effectiveness': (
                sum(self.optimization_effects.values()) / len(self.optimization_effects)
                if self.optimization_effects else 0
            ),
        }
