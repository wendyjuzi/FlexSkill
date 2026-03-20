"""
失败分析引擎

分析失败的根本原因、贡献因素和相似失败。
"""

from typing import Dict, List, Any
from datetime import datetime
import logging
from collections import defaultdict

from .failure_models import FailureRecord

logger = logging.getLogger(__name__)


class FailureAnalyzer:
    """失败分析引擎"""

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.failure_cache: Dict[str, FailureRecord] = {}

    def analyze_failure(
        self,
        failure_record: FailureRecord,
        historical_failures: List[FailureRecord],
    ) -> Dict[str, Any]:
        """
        分析单个失败事件

        返回：
        {
            'failure_id': str,
            'root_cause': str,
            'contributing_factors': List[str],
            'similar_failures': List[str],
            'severity_assessment': str,
            'recommendations': List[str],
            'confidence': float,
        }
        """

        # 步骤1: 特征提取
        features = self._extract_features(failure_record)

        # 步骤2: 查询相似失败
        similar_failures = self._find_similar_failures(
            features,
            historical_failures,
            top_k=10
        )

        # 步骤3: 分析根本原因
        root_cause = self._analyze_root_cause(
            failure_record,
            similar_failures
        )

        # 步骤4: 识别贡献因素
        contributing_factors = self._identify_contributing_factors(
            failure_record,
            root_cause
        )

        # 步骤5: 生成建议
        recommendations = self._generate_recommendations(
            failure_record,
            root_cause,
            contributing_factors
        )

        # 步骤6: 计算置信度
        confidence = self._calculate_confidence(
            failure_record,
            similar_failures,
            root_cause
        )

        return {
            'failure_id': failure_record.failure_id,
            'root_cause': root_cause,
            'contributing_factors': contributing_factors,
            'similar_failures': [f.failure_id for f in similar_failures],
            'severity_assessment': failure_record.severity.name,
            'recommendations': recommendations,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
        }

    def batch_analyze(
        self,
        failure_records: List[FailureRecord],
        historical_failures: List[FailureRecord],
    ) -> List[Dict[str, Any]]:
        """批量分析失败"""
        results = []
        for failure in failure_records:
            try:
                result = self.analyze_failure(failure, historical_failures)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing failure {failure.failure_id}: {e}")

        return results

    # 私有方法

    def _extract_features(self, failure_record: FailureRecord) -> Dict[str, Any]:
        """提取失败特征"""
        return {
            'failure_type': failure_record.failure_type.value,
            'tool_name': failure_record.context.tool_name or 'unknown',
            'medical_domain': failure_record.context.medical_domain,
            'error_pattern': self._extract_error_pattern(
                failure_record.details.error_message
            ),
            'is_systematic': failure_record.details.is_systematic,
            'is_repeatable': failure_record.details.is_repeatable,
        }

    def _extract_error_pattern(self, error_message: str) -> str:
        """从错误消息中提取模式"""
        error_lower = error_message.lower()

        patterns = {
            'parameter': ['parameter', 'argument', 'invalid', 'mismatch'],
            'not_found': ['not found', 'does not exist', 'missing'],
            'timeout': ['timeout', 'timed out', 'exceeded'],
            'permission': ['permission', 'denied', 'unauthorized'],
            'resource': ['resource', 'memory', 'disk', 'cpu'],
            'network': ['network', 'connection', 'timeout', 'unreachable'],
            'format': ['format', 'parse', 'invalid', 'malformed'],
        }

        for pattern_name, keywords in patterns.items():
            if any(keyword in error_lower for keyword in keywords):
                return pattern_name

        return 'unknown'

    def _find_similar_failures(
        self,
        features: Dict[str, Any],
        historical_failures: List[FailureRecord],
        top_k: int = 10,
    ) -> List[FailureRecord]:
        """查找相似失败"""
        similarities = []

        for historical in historical_failures:
            hist_features = self._extract_features(historical)
            similarity = self._calculate_similarity(features, hist_features)

            if similarity >= self.similarity_threshold:
                similarities.append((historical, similarity))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        return [f for f, _ in similarities[:top_k]]

    def _calculate_similarity(
        self,
        features1: Dict[str, Any],
        features2: Dict[str, Any],
    ) -> float:
        """计算两个失败的相似度"""
        score = 0.0
        weights = {
            'failure_type': 0.3,
            'tool_name': 0.2,
            'medical_domain': 0.2,
            'error_pattern': 0.2,
            'is_systematic': 0.05,
            'is_repeatable': 0.05,
        }

        for key, weight in weights.items():
            if features1.get(key) == features2.get(key):
                score += weight

        return score

    def _analyze_root_cause(
        self,
        failure_record: FailureRecord,
        similar_failures: List[FailureRecord],
    ) -> str:
        """分析根本原因"""

        # 如果已有根本原因，直接返回
        if failure_record.details.root_cause:
            return failure_record.details.root_cause

        # 分析直接原因
        error_msg = failure_record.details.error_message.lower()

        if 'parameter' in error_msg or 'argument' in error_msg:
            return 'Invalid parameters or arguments'

        if 'not found' in error_msg or 'does not exist' in error_msg:
            return 'Resource or tool not found'

        if 'timeout' in error_msg:
            return 'Operation timeout'

        if 'permission' in error_msg or 'denied' in error_msg:
            return 'Permission denied'

        if 'memory' in error_msg or 'resource' in error_msg:
            return 'Insufficient resources'

        # 从相似失败中推断
        if similar_failures:
            root_causes = defaultdict(int)
            for similar in similar_failures:
                if similar.details.root_cause:
                    root_causes[similar.details.root_cause] += 1

            if root_causes:
                return max(root_causes.items(), key=lambda x: x[1])[0]

        return 'Unknown root cause'

    def _identify_contributing_factors(
        self,
        failure_record: FailureRecord,
        root_cause: str,
    ) -> List[str]:
        """识别贡献因素"""
        factors = []

        # 检查参数相关因素
        if 'parameter' in root_cause.lower():
            factors.append('Invalid tool parameters')

        # 检查上下文相关因素
        if failure_record.context.case_complexity == 'high':
            factors.append('High case complexity')

        # 检查工具相关因素
        if failure_record.context.tool_name:
            factors.append(f'Tool: {failure_record.context.tool_name}')

        # 检查资源相关因素
        if failure_record.context.token_usage > 8000:
            factors.append('High token usage')

        # 检查系统性因素
        if failure_record.details.is_systematic:
            factors.append('Systematic issue')

        return factors if factors else ['Unknown factors']

    def _generate_recommendations(
        self,
        failure_record: FailureRecord,
        root_cause: str,
        contributing_factors: List[str],
    ) -> List[str]:
        """生成建议"""
        recommendations = []

        if 'parameter' in root_cause.lower():
            recommendations.append('Validate tool parameters before execution')
            recommendations.append('Check parameter types and formats')

        if 'not found' in root_cause.lower():
            recommendations.append('Verify tool availability')
            recommendations.append('Use alternative tools if available')

        if 'timeout' in root_cause.lower():
            recommendations.append('Increase timeout threshold')
            recommendations.append('Optimize tool execution')

        if 'resource' in root_cause.lower():
            recommendations.append('Reduce batch size')
            recommendations.append('Optimize memory usage')

        if 'High case complexity' in contributing_factors:
            recommendations.append('Break down complex tasks')
            recommendations.append('Use simpler reasoning steps')

        if not recommendations:
            recommendations.append('Review error logs for more details')
            recommendations.append('Consult documentation')

        return recommendations

    def _calculate_confidence(
        self,
        failure_record: FailureRecord,
        similar_failures: List[FailureRecord],
        root_cause: str,
    ) -> float:
        """计算分析置信度"""
        confidence = 0.5

        # 因素1: 相似失败数量
        similarity_factor = min(len(similar_failures) / 10, 1.0) * 0.3

        # 因素2: 系统性
        if failure_record.details.is_systematic:
            systematic_factor = 0.2
        else:
            systematic_factor = 0.05

        # 因素3: 根本原因明确性
        if root_cause != 'Unknown root cause':
            clarity_factor = 0.25
        else:
            clarity_factor = 0.05

        confidence = min(
            confidence + similarity_factor + systematic_factor + clarity_factor,
            1.0
        )

        return confidence

    def get_failure_insights(
        self,
        failure_records: List[FailureRecord],
    ) -> Dict[str, Any]:
        """获取失败洞察"""
        if not failure_records:
            return {}

        root_causes = defaultdict(int)
        error_patterns = defaultdict(int)
        tools_with_failures = defaultdict(int)

        for failure in failure_records:
            if failure.details.root_cause:
                root_causes[failure.details.root_cause] += 1

            error_pattern = self._extract_error_pattern(
                failure.details.error_message
            )
            error_patterns[error_pattern] += 1

            if failure.context.tool_name:
                tools_with_failures[failure.context.tool_name] += 1

        return {
            'total_failures': len(failure_records),
            'top_root_causes': sorted(
                root_causes.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'error_patterns': dict(error_patterns),
            'tools_with_failures': sorted(
                tools_with_failures.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
        }
