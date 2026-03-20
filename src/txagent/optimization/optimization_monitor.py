"""
监控和反馈系统

监控优化效果，收集反馈，建立反馈循环。
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import json

logger = logging.getLogger(__name__)


class OptimizationMonitor:
    """优化监控系统"""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.optimization_effects: Dict[str, Dict[str, Any]] = {}
        self.feedback_history: List[Dict[str, Any]] = []
        self.performance_baseline: Dict[str, float] = {}

    def record_metric(self, metric_name: str, value: float) -> None:
        """记录指标"""
        self.metrics[metric_name].append(value)
        logger.debug(f"Recorded metric {metric_name}: {value}")

    def record_optimization_effect(
        self,
        suggestion_id: str,
        before_metrics: Dict[str, float],
        after_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        记录优化效果

        返回效果评估：
        {
            'suggestion_id': str,
            'improvement': float,
            'metrics_changed': Dict[str, float],
            'timestamp': datetime,
        }
        """

        effect = {
            'suggestion_id': suggestion_id,
            'before_metrics': before_metrics,
            'after_metrics': after_metrics,
            'timestamp': datetime.now().isoformat(),
            'metrics_changed': {},
            'improvement': 0.0,
        }

        # 计算改进
        total_improvement = 0.0
        metric_count = 0

        for metric_name, before_value in before_metrics.items():
            if metric_name not in after_metrics:
                continue

            after_value = after_metrics[metric_name]

            if before_value == 0:
                change = 0
            else:
                change = (after_value - before_value) / before_value

            effect['metrics_changed'][metric_name] = change
            total_improvement += change
            metric_count += 1

        if metric_count > 0:
            effect['improvement'] = total_improvement / metric_count

        self.optimization_effects[suggestion_id] = effect
        logger.info(f"Optimization effect recorded: {suggestion_id}, improvement: {effect['improvement']}")

        return effect

    def get_metric_statistics(
        self,
        metric_name: str,
        window_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """获取指标统计"""
        if metric_name not in self.metrics:
            return {}

        values = self.metrics[metric_name]

        if window_size:
            values = values[-window_size:]

        if not values:
            return {}

        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'latest': values[-1] if values else 0,
        }

    def get_all_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """获取所有指标总结"""
        summary = {}

        for metric_name in self.metrics.keys():
            summary[metric_name] = self.get_metric_statistics(metric_name)

        return summary

    def record_feedback(
        self,
        suggestion_id: str,
        feedback_type: str,
        feedback_value: float,
        notes: Optional[str] = None,
    ) -> None:
        """
        记录反馈

        feedback_type: 'helpful', 'harmful', 'neutral'
        feedback_value: 0-1 的反馈值
        """

        feedback = {
            'suggestion_id': suggestion_id,
            'feedback_type': feedback_type,
            'feedback_value': feedback_value,
            'notes': notes,
            'timestamp': datetime.now().isoformat(),
        }

        self.feedback_history.append(feedback)
        logger.info(f"Feedback recorded: {suggestion_id}, type: {feedback_type}, value: {feedback_value}")

    def get_feedback_summary(
        self,
        suggestion_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """获取反馈总结"""
        if suggestion_id:
            feedbacks = [f for f in self.feedback_history if f['suggestion_id'] == suggestion_id]
        else:
            feedbacks = self.feedback_history

        if not feedbacks:
            return {}

        feedback_types = defaultdict(int)
        total_value = 0.0

        for feedback in feedbacks:
            feedback_types[feedback['feedback_type']] += 1
            total_value += feedback['feedback_value']

        return {
            'total_feedbacks': len(feedbacks),
            'feedback_types': dict(feedback_types),
            'average_value': total_value / len(feedbacks),
            'latest_feedback': feedbacks[-1] if feedbacks else None,
        }

    def set_performance_baseline(self, metrics: Dict[str, float]) -> None:
        """设置性能基线"""
        self.performance_baseline = metrics.copy()
        logger.info(f"Performance baseline set: {metrics}")

    def get_performance_improvement(self) -> Dict[str, float]:
        """获取性能改进"""
        if not self.performance_baseline:
            return {}

        improvement = {}

        for metric_name, baseline_value in self.performance_baseline.items():
            current_stats = self.get_metric_statistics(metric_name)

            if not current_stats or 'mean' not in current_stats:
                continue

            current_value = current_stats['mean']

            if baseline_value == 0:
                improvement[metric_name] = 0
            else:
                improvement[metric_name] = (current_value - baseline_value) / baseline_value

        return improvement

    def get_optimization_effectiveness_report(self) -> Dict[str, Any]:
        """获取优化效果报告"""
        if not self.optimization_effects:
            return {}

        improvements = []
        for suggestion_id, effect in self.optimization_effects.items():
            improvements.append({
                'suggestion_id': suggestion_id,
                'improvement': effect['improvement'],
                'timestamp': effect['timestamp'],
            })

        improvements.sort(key=lambda x: x['improvement'], reverse=True)

        total_improvement = sum(e['improvement'] for e in self.optimization_effects.values())
        avg_improvement = total_improvement / len(self.optimization_effects) if self.optimization_effects else 0

        return {
            'total_optimizations': len(self.optimization_effects),
            'average_improvement': avg_improvement,
            'top_optimizations': improvements[:10],
            'bottom_optimizations': improvements[-10:] if len(improvements) > 10 else [],
        }

    def export_monitoring_data(self, output_path: str) -> None:
        """导出监控数据"""
        data = {
            'metrics': {k: v for k, v in self.metrics.items()},
            'optimization_effects': self.optimization_effects,
            'feedback_history': self.feedback_history,
            'performance_baseline': self.performance_baseline,
            'export_time': datetime.now().isoformat(),
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Monitoring data exported to {output_path}")


class FeedbackLoop:
    """反馈循环系统"""

    def __init__(self, monitor: OptimizationMonitor):
        self.monitor = monitor
        self.optimization_history: List[Dict[str, Any]] = []
        self.adjustment_history: List[Dict[str, Any]] = []

    def record_optimization_attempt(
        self,
        suggestion_id: str,
        optimization_type: str,
        confidence: float,
        applied: bool,
    ) -> None:
        """记录优化尝试"""
        attempt = {
            'suggestion_id': suggestion_id,
            'optimization_type': optimization_type,
            'confidence': confidence,
            'applied': applied,
            'timestamp': datetime.now().isoformat(),
        }

        self.optimization_history.append(attempt)

    def evaluate_optimization(
        self,
        suggestion_id: str,
        success: bool,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float],
    ) -> Dict[str, Any]:
        """评估优化"""
        effect = self.monitor.record_optimization_effect(
            suggestion_id,
            metrics_before,
            metrics_after,
        )

        # 记录反馈
        feedback_type = 'helpful' if success else 'harmful'
        feedback_value = effect['improvement'] if success else -effect['improvement']

        self.monitor.record_feedback(
            suggestion_id,
            feedback_type,
            feedback_value,
        )

        return effect

    def adjust_optimization_parameters(
        self,
        suggestion_id: str,
        current_confidence: float,
        feedback_summary: Dict[str, Any],
    ) -> Tuple[float, str]:
        """
        调整优化参数

        返回：(新置信度, 调整原因)
        """

        new_confidence = current_confidence
        reason = ""

        # 基于反馈调整置信度
        avg_feedback = feedback_summary.get('average_value', 0)

        if avg_feedback > 0.8:
            # 反馈很好，增加置信度
            new_confidence = min(current_confidence + 0.1, 1.0)
            reason = "Positive feedback, increased confidence"

        elif avg_feedback < 0.3:
            # 反馈不好，降低置信度
            new_confidence = max(current_confidence - 0.2, 0.0)
            reason = "Negative feedback, decreased confidence"

        else:
            # 反馈中等，保持或微调
            if avg_feedback > 0.5:
                new_confidence = min(current_confidence + 0.05, 1.0)
                reason = "Moderate positive feedback, slightly increased confidence"
            else:
                new_confidence = max(current_confidence - 0.05, 0.0)
                reason = "Moderate negative feedback, slightly decreased confidence"

        adjustment = {
            'suggestion_id': suggestion_id,
            'old_confidence': current_confidence,
            'new_confidence': new_confidence,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
        }

        self.adjustment_history.append(adjustment)

        return new_confidence, reason

    def get_feedback_loop_status(self) -> Dict[str, Any]:
        """获取反馈循环状态"""
        return {
            'total_attempts': len(self.optimization_history),
            'total_adjustments': len(self.adjustment_history),
            'optimization_history': self.optimization_history[-10:],
            'adjustment_history': self.adjustment_history[-10:],
        }

    def get_optimization_recommendations_based_on_feedback(
        self,
        min_confidence: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """基于反馈获取优化建议"""
        recommendations = []

        # 分析优化历史
        optimization_stats = defaultdict(lambda: {'count': 0, 'success': 0})

        for attempt in self.optimization_history:
            opt_type = attempt['optimization_type']
            optimization_stats[opt_type]['count'] += 1

            # 检查反馈
            feedback_summary = self.monitor.get_feedback_summary(attempt['suggestion_id'])
            if feedback_summary.get('average_value', 0) > 0.5:
                optimization_stats[opt_type]['success'] += 1

        # 生成建议
        for opt_type, stats in optimization_stats.items():
            if stats['count'] == 0:
                continue

            success_rate = stats['success'] / stats['count']

            if success_rate > 0.7:
                recommendations.append({
                    'optimization_type': opt_type,
                    'recommendation': 'Continue using this optimization',
                    'success_rate': success_rate,
                    'confidence': min(success_rate, 1.0),
                })

            elif success_rate < 0.3:
                recommendations.append({
                    'optimization_type': opt_type,
                    'recommendation': 'Consider discontinuing this optimization',
                    'success_rate': success_rate,
                    'confidence': 1.0 - success_rate,
                })

        return recommendations


class OptimizationDashboard:
    """优化仪表板"""

    def __init__(self, monitor: OptimizationMonitor, feedback_loop: FeedbackLoop):
        self.monitor = monitor
        self.feedback_loop = feedback_loop

    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        return {
            'metrics_summary': self.monitor.get_all_metrics_summary(),
            'performance_improvement': self.monitor.get_performance_improvement(),
            'optimization_effectiveness': self.monitor.get_optimization_effectiveness_report(),
            'feedback_loop_status': self.feedback_loop.get_feedback_loop_status(),
            'recommendations': self.feedback_loop.get_optimization_recommendations_based_on_feedback(),
            'timestamp': datetime.now().isoformat(),
        }

    def generate_report(self, output_path: str) -> None:
        """生成报告"""
        dashboard_data = self.get_dashboard_data()

        with open(output_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2)

        logger.info(f"Dashboard report generated: {output_path}")

    def print_summary(self) -> str:
        """打印总结"""
        data = self.get_dashboard_data()

        summary = []
        summary.append("=" * 60)
        summary.append("OPTIMIZATION DASHBOARD SUMMARY")
        summary.append("=" * 60)

        # 性能改进
        improvement = data['performance_improvement']
        if improvement:
            summary.append("\nPerformance Improvement:")
            for metric, value in improvement.items():
                summary.append(f"  {metric}: {value:.2%}")

        # 优化效果
        effectiveness = data['optimization_effectiveness']
        if effectiveness:
            summary.append(f"\nOptimization Effectiveness:")
            summary.append(f"  Total Optimizations: {effectiveness.get('total_optimizations', 0)}")
            summary.append(f"  Average Improvement: {effectiveness.get('average_improvement', 0):.2%}")

        # 反馈循环
        feedback_status = data['feedback_loop_status']
        summary.append(f"\nFeedback Loop Status:")
        summary.append(f"  Total Attempts: {feedback_status.get('total_attempts', 0)}")
        summary.append(f"  Total Adjustments: {feedback_status.get('total_adjustments', 0)}")

        # 建议
        recommendations = data['recommendations']
        if recommendations:
            summary.append(f"\nRecommendations:")
            for rec in recommendations[:5]:
                summary.append(f"  {rec['optimization_type']}: {rec['recommendation']}")

        summary.append("=" * 60)

        return "\n".join(summary)
