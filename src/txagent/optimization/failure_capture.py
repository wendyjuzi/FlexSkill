"""
失败捕获系统

实时捕获各类失败事件，包括工具失败、推理失败、系统失败等。
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import traceback

from .failure_models import (
    FailureRecord,
    FailureType,
    SeverityLevel,
    FailureContext,
    FailureDetails,
)

logger = logging.getLogger(__name__)


class FailureCapture:
    """失败捕获器"""

    def __init__(self):
        self.captured_failures: List[FailureRecord] = []
        self.session_id = ""
        self.task_id = ""

    def set_session_info(self, session_id: str, task_id: str) -> None:
        """设置会话信息"""
        self.session_id = session_id
        self.task_id = task_id

    def capture_tool_failure(
        self,
        tool_name: str,
        error_message: str,
        tool_parameters: Dict[str, Any],
        context: FailureContext,
        error_code: Optional[str] = None,
        error_traceback: Optional[str] = None,
    ) -> FailureRecord:
        """捕获工具执行失败"""

        # 确定失败类型
        failure_type = self._determine_tool_failure_type(error_message, error_code)

        # 确定严重程度
        severity = self._determine_severity(failure_type)

        # 创建失败记录
        failure_record = FailureRecord(
            session_id=self.session_id,
            task_id=self.task_id,
            failure_type=failure_type,
            severity=severity,
            context=context,
            details=FailureDetails(
                error_message=error_message,
                error_code=error_code,
                error_traceback=error_traceback,
                is_repeatable=self._check_repeatability(error_message),
                is_systematic=self._check_systematicity(error_message),
            ),
        )

        # 更新上下文中的工具信息
        failure_record.context.tool_name = tool_name
        failure_record.context.tool_parameters = tool_parameters

        # 记录失败
        self.captured_failures.append(failure_record)
        logger.warning(
            f"Tool failure captured: {tool_name} - {error_message} "
            f"(failure_id: {failure_record.failure_id})"
        )

        return failure_record

    def capture_loop_failure(
        self,
        tool_sequence: List[str],
        context: FailureContext,
        loop_count: int = 0,
    ) -> FailureRecord:
        """捕获无限循环失败"""

        failure_record = FailureRecord(
            session_id=self.session_id,
            task_id=self.task_id,
            failure_type=FailureType.INFINITE_LOOP,
            severity=SeverityLevel.HIGH,
            context=context,
            details=FailureDetails(
                error_message=f"Infinite loop detected: {' -> '.join(tool_sequence)}",
                is_repeatable=True,
                is_systematic=True,
            ),
        )

        failure_record.metadata['tool_sequence'] = tool_sequence
        failure_record.metadata['loop_count'] = loop_count

        self.captured_failures.append(failure_record)
        logger.warning(
            f"Loop failure captured: {' -> '.join(tool_sequence)} "
            f"(failure_id: {failure_record.failure_id})"
        )

        return failure_record

    def capture_token_overflow(
        self,
        current_tokens: int,
        max_tokens: int,
        context: FailureContext,
    ) -> FailureRecord:
        """捕获 Token 溢出失败"""

        failure_record = FailureRecord(
            session_id=self.session_id,
            task_id=self.task_id,
            failure_type=FailureType.TOKEN_OVERFLOW,
            severity=SeverityLevel.CRITICAL,
            context=context,
            details=FailureDetails(
                error_message=f"Token overflow: {current_tokens}/{max_tokens}",
                is_repeatable=False,
                is_systematic=False,
            ),
        )

        failure_record.metadata['current_tokens'] = current_tokens
        failure_record.metadata['max_tokens'] = max_tokens
        failure_record.metadata['overflow_ratio'] = current_tokens / max_tokens

        self.captured_failures.append(failure_record)
        logger.warning(
            f"Token overflow captured: {current_tokens}/{max_tokens} "
            f"(failure_id: {failure_record.failure_id})"
        )

        return failure_record

    def capture_reasoning_failure(
        self,
        failure_reason: str,
        context: FailureContext,
        error_traceback: Optional[str] = None,
    ) -> FailureRecord:
        """捕获推理失败"""

        # 确定失败类型
        failure_type = self._determine_reasoning_failure_type(failure_reason)

        failure_record = FailureRecord(
            session_id=self.session_id,
            task_id=self.task_id,
            failure_type=failure_type,
            severity=SeverityLevel.HIGH,
            context=context,
            details=FailureDetails(
                error_message=failure_reason,
                error_traceback=error_traceback,
                is_repeatable=True,
                is_systematic=True,
            ),
        )

        self.captured_failures.append(failure_record)
        logger.warning(
            f"Reasoning failure captured: {failure_reason} "
            f"(failure_id: {failure_record.failure_id})"
        )

        return failure_record

    def capture_exception(
        self,
        exception: Exception,
        context: FailureContext,
    ) -> FailureRecord:
        """捕获异常"""

        failure_record = FailureRecord(
            session_id=self.session_id,
            task_id=self.task_id,
            failure_type=FailureType.EXTERNAL_SERVICE_ERROR,
            severity=SeverityLevel.HIGH,
            context=context,
            details=FailureDetails(
                error_message=str(exception),
                error_code=type(exception).__name__,
                error_traceback=traceback.format_exc(),
                is_repeatable=False,
                is_systematic=False,
            ),
        )

        self.captured_failures.append(failure_record)
        logger.error(
            f"Exception captured: {type(exception).__name__} - {str(exception)} "
            f"(failure_id: {failure_record.failure_id})"
        )

        return failure_record

    def get_captured_failures(self) -> List[FailureRecord]:
        """获取所有捕获的失败"""
        return self.captured_failures.copy()

    def clear_failures(self) -> None:
        """清空失败记录"""
        self.captured_failures.clear()

    # 私有方法

    def _determine_tool_failure_type(
        self,
        error_message: str,
        error_code: Optional[str] = None,
    ) -> FailureType:
        """确定工具失败类型"""

        error_lower = error_message.lower()

        if "not found" in error_lower or "does not exist" in error_lower:
            return FailureType.TOOL_NOT_FOUND

        if "parameter" in error_lower or "argument" in error_lower:
            return FailureType.INVALID_PARAMETERS

        if "timeout" in error_lower:
            return FailureType.TOOL_TIMEOUT

        return FailureType.TOOL_EXECUTION_ERROR

    def _determine_reasoning_failure_type(
        self,
        failure_reason: str,
    ) -> FailureType:
        """确定推理失败类型"""

        reason_lower = failure_reason.lower()

        if "loop" in reason_lower or "repeat" in reason_lower:
            return FailureType.INFINITE_LOOP

        if "dead end" in reason_lower or "stuck" in reason_lower:
            return FailureType.REASONING_DEAD_END

        if "tool" in reason_lower and "select" in reason_lower:
            return FailureType.INVALID_TOOL_SELECTION

        if "context" in reason_lower or "mismatch" in reason_lower:
            return FailureType.CONTEXT_MISMATCH

        return FailureType.REASONING_DEAD_END

    def _determine_severity(self, failure_type: FailureType) -> SeverityLevel:
        """确定严重程度"""

        severity_map = {
            FailureType.TOOL_EXECUTION_ERROR: SeverityLevel.MEDIUM,
            FailureType.TOOL_NOT_FOUND: SeverityLevel.HIGH,
            FailureType.INVALID_PARAMETERS: SeverityLevel.MEDIUM,
            FailureType.TOOL_TIMEOUT: SeverityLevel.MEDIUM,
            FailureType.INFINITE_LOOP: SeverityLevel.HIGH,
            FailureType.REASONING_DEAD_END: SeverityLevel.HIGH,
            FailureType.INVALID_TOOL_SELECTION: SeverityLevel.MEDIUM,
            FailureType.CONTEXT_MISMATCH: SeverityLevel.MEDIUM,
            FailureType.TOKEN_OVERFLOW: SeverityLevel.CRITICAL,
            FailureType.MEMORY_ERROR: SeverityLevel.CRITICAL,
            FailureType.RATE_LIMIT: SeverityLevel.MEDIUM,
            FailureType.EXTERNAL_SERVICE_ERROR: SeverityLevel.HIGH,
        }

        return severity_map.get(failure_type, SeverityLevel.MEDIUM)

    def _check_repeatability(self, error_message: str) -> bool:
        """检查失败是否可重复"""

        non_repeatable_keywords = [
            "timeout",
            "rate limit",
            "temporary",
            "transient",
            "random",
        ]

        error_lower = error_message.lower()
        return not any(keyword in error_lower for keyword in non_repeatable_keywords)

    def _check_systematicity(self, error_message: str) -> bool:
        """检查失败是否系统性"""

        systematic_keywords = [
            "parameter",
            "invalid",
            "not found",
            "mismatch",
            "incompatible",
            "unsupported",
        ]

        error_lower = error_message.lower()
        return any(keyword in error_lower for keyword in systematic_keywords)
