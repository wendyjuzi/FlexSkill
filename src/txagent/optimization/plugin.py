"""
插件管理系统

提供插件注册、事件触发等基础设施。
"""

from typing import Dict, List, Callable, Any, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class PluginManager:
    """插件管理器"""

    def __init__(self):
        self.plugins: Dict[str, 'BasePlugin'] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}

    def register_plugin(self, plugin_name: str, plugin: 'BasePlugin') -> None:
        """注册插件"""
        if plugin_name in self.plugins:
            logger.warning(f"Plugin {plugin_name} already registered, overwriting")

        self.plugins[plugin_name] = plugin
        logger.info(f"Plugin {plugin_name} registered")

    def unregister_plugin(self, plugin_name: str) -> None:
        """注销插件"""
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
            logger.info(f"Plugin {plugin_name} unregistered")

    def register_event_handler(
        self,
        event_name: str,
        handler: Callable
    ) -> None:
        """注册事件处理器"""
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []

        self.event_handlers[event_name].append(handler)

    def trigger_event(
        self,
        event_name: str,
        *args,
        **kwargs
    ) -> List[Any]:
        """触发事件"""
        results = []

        if event_name not in self.event_handlers:
            return results

        for handler in self.event_handlers[event_name]:
            try:
                result = handler(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in event handler for {event_name}: {e}")

        return results

    def get_plugin(self, plugin_name: str) -> Optional['BasePlugin']:
        """获取插件"""
        return self.plugins.get(plugin_name)

    def list_plugins(self) -> List[str]:
        """列出所有插件"""
        return list(self.plugins.keys())


class BasePlugin(ABC):
    """插件基类"""

    def __init__(self, name: str):
        self.name = name
        self.enabled = True

    @abstractmethod
    def on_register(self, tx_agent: Any) -> None:
        """插件注册时调用"""
        pass

    @abstractmethod
    def on_unregister(self) -> None:
        """插件注销时调用"""
        pass

    def enable(self) -> None:
        """启用插件"""
        self.enabled = True
        logger.info(f"Plugin {self.name} enabled")

    def disable(self) -> None:
        """禁用插件"""
        self.enabled = False
        logger.info(f"Plugin {self.name} disabled")

    def is_enabled(self) -> bool:
        """检查插件是否启用"""
        return self.enabled


class OptimizationPlugin(BasePlugin):
    """优化插件基类"""

    def __init__(self, name: str = "optimization_plugin"):
        super().__init__(name)
        self.tx_agent = None

    def on_register(self, tx_agent: Any) -> None:
        """插件注册时初始化"""
        self.tx_agent = tx_agent
        logger.info(f"Optimization plugin {self.name} registered with TxAgent")

    def on_unregister(self) -> None:
        """插件注销时清理"""
        self.tx_agent = None
        logger.info(f"Optimization plugin {self.name} unregistered")

    # 事件处理方法（子类可覆盖）

    def on_tool_execution_start(self, context: Dict[str, Any]) -> None:
        """工具执行开始"""
        pass

    def on_tool_execution_success(self, context: Dict[str, Any]) -> None:
        """工具执行成功"""
        pass

    def on_tool_execution_failed(self, context: Dict[str, Any]) -> None:
        """工具执行失败"""
        pass

    def on_reasoning_step_start(self, context: Dict[str, Any]) -> None:
        """推理步骤开始"""
        pass

    def on_reasoning_step_complete(self, context: Dict[str, Any]) -> None:
        """推理步骤完成"""
        pass

    def on_loop_detected(self, context: Dict[str, Any]) -> None:
        """检测到循环"""
        pass

    def on_token_overflow(self, context: Dict[str, Any]) -> None:
        """Token 溢出"""
        pass

    def on_task_complete(self, context: Dict[str, Any]) -> None:
        """任务完成"""
        pass

    def on_task_failed(self, context: Dict[str, Any]) -> None:
        """任务失败"""
        pass
