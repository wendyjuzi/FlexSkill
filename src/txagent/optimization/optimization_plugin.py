"""
优化插件

将优化系统集成到 TxAgent 中。
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from .plugin import OptimizationPlugin
from .failure_capture import FailureCapture
from .failure_store import FailureStore
from .optimization_engine import OptimizationEngine

logger = logging.getLogger(__name__)


class TxAgentOptimizationPlugin(OptimizationPlugin):
    """TxAgent 优化插件"""

    def __init__(
        self,
        name: str = "txagent_optimization",
        failure_store_path: str = "data/failures.jsonl",
        cache_size: int = 1000,
        cache_ttl: int = 3600,
    ):
        super().__init__(name)

        self.failure_capture = FailureCapture()
        self.failure_store = FailureStore(failure_store_path)
        self.optimization_engine = OptimizationEngine(
            self.failure_store,
            cache_size=cache_size,
            cache_ttl=cache_ttl,
        )

        self.tx_agent = None
        self.enabled = True
        self.auto_apply = True

    def on_register(self, tx_agent: Any) -> None:
        """插件注册时调用"""
        self.tx_agent = tx_agent
        logger.info(f"Plugin {self.name} registered with TxAgent")

    def on_tool_execution_start(self, context: Dict[str, Any]) -> None:
        """工具执行开始"""
        if not self.enabled:
            return

        # 记录工具执行开始
        logger.debug(f"Tool execution started: {context.get('tool_name')}")

    def on_tool_execution_failed(self, context: Dict[str, Any]) -> None:
        """工具执行失败"""
        if not self.enabled:
            return

        try:
            # 捕获工具失败
            failure = self.failure_capture.capture_tool_failure(
                tool_name=context.get('tool_name'),
                error_message=context.get('error_message', ''),
                tool_parameters=context.get('tool_parameters', {}),
                context=context.get('failure_context'),
            )

            # 保存失败
            self.failure_store.save_failure(failure)

            # 处理失败并生成优化
            if self.auto_apply:
                optimization_result = self.optimization_engine.process_failure(
                    failure,
                    self.tx_agent,
                )

                logger.info(
                    f"Optimization processed for failure {failure.failure_id}: "
                    f"{len(optimization_result['applied'])} applied, "
                    f"{len(optimization_result['cached'])} cached"
                )

        except Exception as e:
            logger.error(f"Error handling tool execution failure: {e}")

    def on_loop_detected(self, context: Dict[str, Any]) -> None:
        """检测到循环"""
        if not self.enabled:
            return

        try:
            # 捕获循环失败
            failure = self.failure_capture.capture_loop_failure(
                tool_sequence=context.get('tool_sequence', []),
                context=context.get('failure_context'),
            )

            # 保存失败
            self.failure_store.save_failure(failure)

            # 处理失败并生成优化
            if self.auto_apply:
                optimization_result = self.optimization_engine.process_failure(
                    failure,
                    self.tx_agent,
                )

                logger.info(
                    f"Optimization processed for loop detection: "
                    f"{len(optimization_result['applied'])} applied"
                )

        except Exception as e:
            logger.error(f"Error handling loop detection: {e}")

    def on_token_overflow(self, context: Dict[str, Any]) -> None:
        """Token 溢出"""
        if not self.enabled:
            return

        try:
            # 捕获 Token 溢出
            failure = self.failure_capture.capture_token_overflow(
                current_tokens=context.get('current_tokens', 0),
                max_tokens=context.get('max_tokens', 0),
                context=context.get('failure_context'),
            )

            # 保存失败
            self.failure_store.save_failure(failure)

            # 处理失败并生成优化
            if self.auto_apply:
                optimization_result = self.optimization_engine.process_failure(
                    failure,
                    self.tx_agent,
                )

                logger.info(
                    f"Optimization processed for token overflow: "
                    f"{len(optimization_result['applied'])} applied"
                )

        except Exception as e:
            logger.error(f"Error handling token overflow: {e}")

    def on_reasoning_failure(self, context: Dict[str, Any]) -> None:
        """推理失败"""
        if not self.enabled:
            return

        try:
            # 捕获推理失败
            failure = self.failure_capture.capture_reasoning_failure(
                failure_reason=context.get('error_message', ''),
                context=context.get('failure_context'),
            )

            # 保存失败
            self.failure_store.save_failure(failure)

            # 处理失败并生成优化
            if self.auto_apply:
                optimization_result = self.optimization_engine.process_failure(
                    failure,
                    self.tx_agent,
                )

                logger.info(
                    f"Optimization processed for reasoning failure: "
                    f"{len(optimization_result['applied'])} applied"
                )

        except Exception as e:
            logger.error(f"Error handling reasoning failure: {e}")

    def on_exception(self, context: Dict[str, Any]) -> None:
        """异常发生"""
        if not self.enabled:
            return

        try:
            # 捕获异常
            failure = self.failure_capture.capture_exception(
                exception=context.get('exception'),
                context=context.get('failure_context'),
            )

            # 保存失败
            self.failure_store.save_failure(failure)

            # 处理失败并生成优化
            if self.auto_apply:
                optimization_result = self.optimization_engine.process_failure(
                    failure,
                    self.tx_agent,
                )

                logger.info(
                    f"Optimization processed for exception: "
                    f"{len(optimization_result['applied'])} applied"
                )

        except Exception as e:
            logger.error(f"Error handling exception: {e}")

    def on_reasoning_step_complete(self, context: Dict[str, Any]) -> None:
        """推理步骤完成"""
        if not self.enabled:
            return

        # 可以在这里进行定期的分析和优化
        logger.debug("Reasoning step completed")

    def on_session_start(self, context: Dict[str, Any]) -> None:
        """会话开始"""
        if not self.enabled:
            return

        session_id = context.get('session_id')
        task_id = context.get('task_id')

        self.failure_capture.set_session_info(session_id, task_id)
        logger.info(f"Session started: {session_id}")

    def on_session_end(self, context: Dict[str, Any]) -> None:
        """会话结束"""
        if not self.enabled:
            return

        session_id = context.get('session_id')
        logger.info(f"Session ended: {session_id}")

    def enable(self) -> None:
        """启用插件"""
        self.enabled = True
        logger.info(f"Plugin {self.name} enabled")

    def disable(self) -> None:
        """禁用插件"""
        self.enabled = False
        logger.info(f"Plugin {self.name} disabled")

    def get_status(self) -> Dict[str, Any]:
        """获取插件状态"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'auto_apply': self.auto_apply,
            'engine_stats': self.optimization_engine.get_engine_statistics(),
            'failure_store_size': self.failure_store.get_store_size(),
        }

    def get_failure_statistics(self) -> Dict[str, Any]:
        """获取失败统计"""
        failure_records = self.failure_store.load_all_failures()
        return self.failure_store.get_failure_statistics()

    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化总结"""
        return self.optimization_engine.get_optimization_summary()

    def get_cached_optimizations(self) -> List[Dict[str, Any]]:
        """获取缓存的优化"""
        return self.optimization_engine.get_cached_optimizations()

    def apply_cached_optimization(self, suggestion_id: str) -> Dict[str, Any]:
        """应用缓存的优化"""
        return self.optimization_engine.apply_cached_optimization(
            suggestion_id,
            self.tx_agent,
        )

    def clear_cache(self) -> None:
        """清空缓存"""
        self.optimization_engine.clear_cache()
        logger.info("Optimization cache cleared")

    def reset_engine(self) -> None:
        """重置引擎"""
        self.optimization_engine.reset_engine()
        logger.info("Optimization engine reset")

    def export_failures(self, output_path: str) -> None:
        """导出失败数据"""
        self.failure_store.export_to_json(output_path)
        logger.info(f"Failures exported to {output_path}")

    def cleanup_old_failures(self, days: int = 30) -> int:
        """清理旧失败数据"""
        count = self.failure_store.clear_old_failures(days=days)
        logger.info(f"Cleaned up {count} old failures")
        return count

    def get_recommendations(
        self,
        min_confidence: float = 0.7,
        max_effort: str = 'medium',
        max_risk: str = 'medium',
    ) -> List[Dict[str, Any]]:
        """获取优化建议"""
        return self.optimization_engine.get_optimization_recommendations(
            min_confidence=min_confidence,
            max_effort=max_effort,
            max_risk=max_risk,
        )
