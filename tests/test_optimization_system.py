"""
单元测试和集成测试

测试失败经验优化系统的各个模块。
"""

import unittest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

from txagent.optimization import (
    FailureCapture,
    FailureStore,
    FailureContext,
    FailureType,
    SeverityLevel,
    FailureAnalyzer,
    PatternRecognizer,
    OptimizationGenerator,
    OptimizationApplier,
    OptimizationEngine,
    TxAgentOptimizationPlugin,
)


class TestFailureCapture(unittest.TestCase):
    """失败捕获测试"""

    def setUp(self):
        self.capture = FailureCapture()
        self.capture.set_session_info("test_session", "test_task")

    def test_capture_tool_failure(self):
        """测试工具失败捕获"""
        context = FailureContext(reasoning_step=1, tool_name="TestTool")
        failure = self.capture.capture_tool_failure(
            tool_name="TestTool",
            error_message="Parameter error",
            tool_parameters={"param1": "value1"},
            context=context,
        )

        self.assertIsNotNone(failure)
        self.assertEqual(failure.context.tool_name, "TestTool")
        self.assertIn("Parameter", failure.details.error_message)

    def test_capture_loop_failure(self):
        """测试循环失败捕获"""
        context = FailureContext(reasoning_step=5)
        failure = self.capture.capture_loop_failure(
            tool_sequence=["Tool_A", "Tool_B", "Tool_A", "Tool_B"],
            context=context,
        )

        self.assertIsNotNone(failure)
        self.assertEqual(failure.failure_type, FailureType.INFINITE_LOOP)

    def test_capture_token_overflow(self):
        """测试 Token 溢出捕获"""
        context = FailureContext(reasoning_step=3)
        failure = self.capture.capture_token_overflow(
            current_tokens=9000,
            max_tokens=8000,
            context=context,
        )

        self.assertIsNotNone(failure)
        self.assertEqual(failure.failure_type, FailureType.TOKEN_OVERFLOW)


class TestFailureStore(unittest.TestCase):
    """失败存储测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.temp_dir, "test_failures.jsonl")
        self.store = FailureStore(self.store_path)
        self.capture = FailureCapture()
        self.capture.set_session_info("test_session", "test_task")

    def tearDown(self):
        # 清理临时文件
        if os.path.exists(self.store_path):
            os.remove(self.store_path)
        os.rmdir(self.temp_dir)

    def test_save_and_load_failure(self):
        """测试保存和加载失败"""
        context = FailureContext(reasoning_step=1, tool_name="TestTool")
        failure = self.capture.capture_tool_failure(
            tool_name="TestTool",
            error_message="Test error",
            tool_parameters={},
            context=context,
        )

        self.store.save_failure(failure)

        # 加载并验证
        failures = self.store.load_all_failures()
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].context.tool_name, "TestTool")

    def test_query_by_tool(self):
        """测试按工具查询"""
        context1 = FailureContext(reasoning_step=1, tool_name="Tool_A")
        context2 = FailureContext(reasoning_step=2, tool_name="Tool_B")

        failure1 = self.capture.capture_tool_failure(
            tool_name="Tool_A",
            error_message="Error 1",
            tool_parameters={},
            context=context1,
        )
        failure2 = self.capture.capture_tool_failure(
            tool_name="Tool_B",
            error_message="Error 2",
            tool_parameters={},
            context=context2,
        )

        self.store.save_failures([failure1, failure2])

        # 查询
        tool_a_failures = self.store.query_by_tool("Tool_A")
        self.assertEqual(len(tool_a_failures), 1)
        self.assertEqual(tool_a_failures[0].context.tool_name, "Tool_A")

    def test_get_statistics(self):
        """测试统计"""
        context = FailureContext(reasoning_step=1, tool_name="TestTool")
        failure = self.capture.capture_tool_failure(
            tool_name="TestTool",
            error_message="Test error",
            tool_parameters={},
            context=context,
        )

        self.store.save_failure(failure)

        stats = self.store.get_failure_statistics()
        self.assertEqual(stats['total_failures'], 1)
        self.assertIn('failure_type_distribution', stats)


class TestFailureAnalyzer(unittest.TestCase):
    """失败分析测试"""

    def setUp(self):
        self.analyzer = FailureAnalyzer()
        self.capture = FailureCapture()
        self.capture.set_session_info("test_session", "test_task")

    def test_analyze_failure(self):
        """测试失败分析"""
        context = FailureContext(reasoning_step=1, tool_name="TestTool")
        failure = self.capture.capture_tool_failure(
            tool_name="TestTool",
            error_message="Parameter mismatch",
            tool_parameters={},
            context=context,
        )

        analysis = self.analyzer.analyze_failure(failure, [])

        self.assertIsNotNone(analysis)
        self.assertIn('root_cause', analysis)
        self.assertIn('confidence', analysis)
        self.assertGreater(analysis['confidence'], 0)

    def test_batch_analyze(self):
        """测试批量分析"""
        failures = []
        for i in range(3):
            context = FailureContext(reasoning_step=i, tool_name="TestTool")
            failure = self.capture.capture_tool_failure(
                tool_name="TestTool",
                error_message="Parameter error",
                tool_parameters={},
                context=context,
            )
            failures.append(failure)

        results = self.analyzer.batch_analyze(failures, [])
        self.assertEqual(len(results), 3)


class TestPatternRecognizer(unittest.TestCase):
    """模式识别测试"""

    def setUp(self):
        self.recognizer = PatternRecognizer()
        self.capture = FailureCapture()
        self.capture.set_session_info("test_session", "test_task")

    def test_recognize_loop_patterns(self):
        """测试循环模式识别"""
        failures = []

        # 创建循环失败
        for i in range(5):
            context = FailureContext(reasoning_step=i, session_id="test_session")
            failure = self.capture.capture_loop_failure(
                tool_sequence=["Tool_A", "Tool_B", "Tool_A"],
                context=context,
            )
            failures.append(failure)

        patterns = self.recognizer.recognize_patterns(failures)

        # 应该识别到循环模式
        loop_patterns = [p for p in patterns if p['pattern_type'] == 'infinite_loop']
        self.assertGreater(len(loop_patterns), 0)

    def test_get_pattern_statistics(self):
        """测试模式统计"""
        patterns = [
            {
                'pattern_id': 'p1',
                'pattern_type': 'infinite_loop',
                'frequency': 5,
            },
            {
                'pattern_id': 'p2',
                'pattern_type': 'cascading_failure',
                'frequency': 3,
            },
        ]

        stats = self.recognizer.get_pattern_statistics(patterns)

        self.assertEqual(stats['total_patterns'], 2)
        self.assertIn('infinite_loop', stats['pattern_types'])


class TestOptimizationGenerator(unittest.TestCase):
    """优化生成测试"""

    def setUp(self):
        self.generator = OptimizationGenerator()

    def test_generate_loop_optimizations(self):
        """测试循环优化生成"""
        patterns = [
            {
                'pattern_id': 'loop_1',
                'pattern_type': 'infinite_loop',
                'frequency': 5,
                'affected_tools': ['Tool_A', 'Tool_B'],
                'affected_domains': ['pharmacology'],
            }
        ]

        suggestions = self.generator.generate_suggestions(patterns, [])

        self.assertGreater(len(suggestions), 0)
        self.assertTrue(any(s['optimization_type'] == 'loop_detection' for s in suggestions))

    def test_prioritize_suggestions(self):
        """测试优化优先级排序"""
        suggestions = [
            {
                'suggestion_id': 's1',
                'expected_improvement': 0.1,
                'confidence': 0.6,
                'implementation_effort': 'high',
                'risk_level': 'high',
            },
            {
                'suggestion_id': 's2',
                'expected_improvement': 0.3,
                'confidence': 0.9,
                'implementation_effort': 'low',
                'risk_level': 'low',
            },
        ]

        prioritized = self.generator.prioritize_suggestions(suggestions)

        # s2 应该优先级更高
        self.assertEqual(prioritized[0]['suggestion_id'], 's2')

    def test_filter_suggestions(self):
        """测试优化过滤"""
        suggestions = [
            {
                'suggestion_id': 's1',
                'confidence': 0.9,
                'implementation_effort': 'low',
                'risk_level': 'low',
            },
            {
                'suggestion_id': 's2',
                'confidence': 0.5,
                'implementation_effort': 'high',
                'risk_level': 'high',
            },
        ]

        filtered = self.generator.filter_suggestions(
            suggestions,
            min_confidence=0.7,
            max_effort='medium',
            max_risk='medium',
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['suggestion_id'], 's1')


class TestOptimizationApplier(unittest.TestCase):
    """优化应用测试"""

    def setUp(self):
        self.applier = OptimizationApplier()

    def test_apply_optimization(self):
        """测试优化应用"""
        suggestion = {
            'suggestion_id': 'test_1',
            'optimization_type': 'loop_detection',
            'parameters': {
                'loop_detection_threshold': 2,
                'loop_check_interval': 1,
            },
        }

        result = self.applier.apply_optimization(suggestion)

        self.assertTrue(result['applied'])
        self.assertEqual(result['status'], 'applied')

    def test_get_current_parameters(self):
        """测试获取当前参数"""
        suggestion = {
            'suggestion_id': 'test_1',
            'optimization_type': 'loop_detection',
            'parameters': {
                'loop_detection_threshold': 2,
            },
        }

        self.applier.apply_optimization(suggestion)
        params = self.applier.get_current_parameters()

        self.assertEqual(params['loop_detection_threshold'], 2)

    def test_rollback_optimization(self):
        """测试回滚优化"""
        suggestion = {
            'suggestion_id': 'test_1',
            'optimization_type': 'loop_detection',
            'parameters': {
                'loop_detection_threshold': 2,
            },
        }

        self.applier.apply_optimization(suggestion)
        self.assertTrue(self.applier.rollback_optimization('test_1'))

        params = self.applier.get_current_parameters()
        self.assertNotIn('loop_detection_threshold', params)


class TestOptimizationEngine(unittest.TestCase):
    """优化引擎测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.temp_dir, "test_failures.jsonl")
        self.store = FailureStore(self.store_path)
        self.engine = OptimizationEngine(self.store)
        self.capture = FailureCapture()
        self.capture.set_session_info("test_session", "test_task")

    def tearDown(self):
        if os.path.exists(self.store_path):
            os.remove(self.store_path)
        os.rmdir(self.temp_dir)

    def test_process_failure(self):
        """测试失败处理"""
        context = FailureContext(reasoning_step=1, tool_name="TestTool")
        failure = self.capture.capture_tool_failure(
            tool_name="TestTool",
            error_message="Parameter error",
            tool_parameters={},
            context=context,
        )

        result = self.engine.process_failure(failure)

        self.assertEqual(result['status'], 'completed')
        self.assertIsNotNone(result['analysis'])

    def test_get_engine_statistics(self):
        """测试引擎统计"""
        stats = self.engine.get_engine_statistics()

        self.assertIn('cache_stats', stats)
        self.assertIn('applier_stats', stats)


class TestTxAgentOptimizationPlugin(unittest.TestCase):
    """TxAgent 优化插件测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.temp_dir, "test_failures.jsonl")
        self.plugin = TxAgentOptimizationPlugin(
            failure_store_path=self.store_path
        )

    def tearDown(self):
        if os.path.exists(self.store_path):
            os.remove(self.store_path)
        os.rmdir(self.temp_dir)

    def test_plugin_registration(self):
        """测试插件注册"""
        mock_agent = type('MockAgent', (), {})()
        self.plugin.on_register(mock_agent)

        self.assertEqual(self.plugin.tx_agent, mock_agent)

    def test_tool_execution_failed(self):
        """测试工具执行失败处理"""
        from txagent.optimization import FailureContext

        context = FailureContext(reasoning_step=1, tool_name="TestTool")

        self.plugin.on_tool_execution_failed({
            'tool_name': 'TestTool',
            'error_message': 'Test error',
            'tool_parameters': {},
            'failure_context': context,
        })

        # 验证失败被保存
        failures = self.plugin.failure_store.load_all_failures()
        self.assertEqual(len(failures), 1)

    def test_get_status(self):
        """测试获取插件状态"""
        status = self.plugin.get_status()

        self.assertIn('name', status)
        self.assertIn('enabled', status)
        self.assertTrue(status['enabled'])


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.temp_dir, "test_failures.jsonl")

    def tearDown(self):
        if os.path.exists(self.store_path):
            os.remove(self.store_path)
        os.rmdir(self.temp_dir)

    def test_end_to_end_optimization(self):
        """测试端到端优化流程"""
        # 初始化系统
        store = FailureStore(self.store_path)
        engine = OptimizationEngine(store)
        capture = FailureCapture()
        capture.set_session_info("test_session", "test_task")

        # 创建多个失败
        failures = []
        for i in range(5):
            context = FailureContext(reasoning_step=i, tool_name="TestTool")
            failure = capture.capture_tool_failure(
                tool_name="TestTool",
                error_message="Parameter error",
                tool_parameters={},
                context=context,
            )
            store.save_failure(failure)
            failures.append(failure)

        # 处理失败
        for failure in failures:
            result = engine.process_failure(failure)
            self.assertEqual(result['status'], 'completed')

        # 验证统计
        stats = store.get_failure_statistics()
        self.assertEqual(stats['total_failures'], 5)

        # 验证优化
        summary = engine.get_optimization_summary()
        self.assertGreater(summary['total_optimizations_attempted'], 0)


if __name__ == '__main__':
    unittest.main()
