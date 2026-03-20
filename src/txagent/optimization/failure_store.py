"""
失败存储系统

使用 JSONL 格式存储失败记录，支持流式处理、查询和分析。
"""

import json
import os
from typing import List, Dict, Any, Optional, Iterator
from datetime import datetime, timedelta
import logging
from pathlib import Path

from .failure_models import FailureRecord, FailurePattern, OptimizationSuggestion

logger = logging.getLogger(__name__)


class FailureStore:
    """失败存储器（JSONL 格式）"""

    def __init__(self, store_path: str = "data/failures.jsonl"):
        self.store_path = store_path
        self.max_file_size_mb = 100
        self.compression_enabled = False

        # 确保目录存在
        Path(self.store_path).parent.mkdir(parents=True, exist_ok=True)

        # 内存缓存（用于快速查询）
        self.memory_cache: List[FailureRecord] = []
        self.cache_loaded = False

    def save_failure(self, failure_record: FailureRecord) -> bool:
        """保存单个失败记录"""
        try:
            with open(self.store_path, 'a', encoding='utf-8') as f:
                json_line = json.dumps(failure_record.to_dict(), ensure_ascii=False)
                f.write(json_line + '\n')

            # 更新内存缓存
            self.memory_cache.append(failure_record)

            logger.debug(f"Failure record saved: {failure_record.failure_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving failure record: {e}")
            return False

    def save_failures(self, failure_records: List[FailureRecord]) -> int:
        """批量保存失败记录"""
        saved_count = 0

        try:
            with open(self.store_path, 'a', encoding='utf-8') as f:
                for record in failure_records:
                    json_line = json.dumps(record.to_dict(), ensure_ascii=False)
                    f.write(json_line + '\n')
                    saved_count += 1

            # 更新内存缓存
            self.memory_cache.extend(failure_records)

            logger.debug(f"Batch saved {saved_count} failure records")
            return saved_count

        except Exception as e:
            logger.error(f"Error saving failure records: {e}")
            return saved_count

    def load_all_failures(self) -> List[FailureRecord]:
        """加载所有失败记录"""
        if self.cache_loaded and self.memory_cache:
            return self.memory_cache.copy()

        self.memory_cache.clear()

        if not os.path.exists(self.store_path):
            logger.warning(f"Store file not found: {self.store_path}")
            return []

        try:
            with open(self.store_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            record = FailureRecord.from_dict(data)
                            self.memory_cache.append(record)
                        except Exception as e:
                            logger.warning(f"Error parsing failure record: {e}")

            self.cache_loaded = True
            logger.info(f"Loaded {len(self.memory_cache)} failure records")
            return self.memory_cache.copy()

        except Exception as e:
            logger.error(f"Error loading failure records: {e}")
            return []

    def query_by_failure_type(self, failure_type: str) -> List[FailureRecord]:
        """按失败类型查询"""
        self._ensure_cache_loaded()

        return [
            record for record in self.memory_cache
            if record.failure_type.value == failure_type
        ]

    def query_by_tool(self, tool_name: str) -> List[FailureRecord]:
        """按工具名称查询"""
        self._ensure_cache_loaded()

        return [
            record for record in self.memory_cache
            if record.context.tool_name == tool_name
        ]

    def query_by_domain(self, domain: str) -> List[FailureRecord]:
        """按医疗领域查询"""
        self._ensure_cache_loaded()

        return [
            record for record in self.memory_cache
            if record.context.medical_domain == domain
        ]

    def query_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[FailureRecord]:
        """按时间范围查询"""
        self._ensure_cache_loaded()

        return [
            record for record in self.memory_cache
            if start_time <= record.timestamp <= end_time
        ]

    def query_recent(self, hours: int = 24) -> List[FailureRecord]:
        """查询最近 N 小时的失败"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        return self.query_by_time_range(start_time, end_time)

    def query_by_severity(self, severity_level: int) -> List[FailureRecord]:
        """按严重程度查询"""
        self._ensure_cache_loaded()

        return [
            record for record in self.memory_cache
            if record.severity.value == severity_level
        ]

    def query_systematic_failures(self) -> List[FailureRecord]:
        """查询系统性失败"""
        self._ensure_cache_loaded()

        return [
            record for record in self.memory_cache
            if record.details.is_systematic
        ]

    def query_repeatable_failures(self) -> List[FailureRecord]:
        """查询可重复的失败"""
        self._ensure_cache_loaded()

        return [
            record for record in self.memory_cache
            if record.details.is_repeatable
        ]

    def get_failure_statistics(self) -> Dict[str, Any]:
        """获取失败统计信息"""
        self._ensure_cache_loaded()

        if not self.memory_cache:
            return {
                'total_failures': 0,
                'by_type': {},
                'by_tool': {},
                'by_domain': {},
                'by_severity': {},
            }

        stats = {
            'total_failures': len(self.memory_cache),
            'by_type': {},
            'by_tool': {},
            'by_domain': {},
            'by_severity': {},
            'systematic_failures': 0,
            'repeatable_failures': 0,
        }

        for record in self.memory_cache:
            # 按类型统计
            failure_type = record.failure_type.value
            stats['by_type'][failure_type] = stats['by_type'].get(failure_type, 0) + 1

            # 按工具统计
            if record.context.tool_name:
                tool_name = record.context.tool_name
                stats['by_tool'][tool_name] = stats['by_tool'].get(tool_name, 0) + 1

            # 按领域统计
            domain = record.context.medical_domain
            stats['by_domain'][domain] = stats['by_domain'].get(domain, 0) + 1

            # 按严重程度统计
            severity = record.severity.value
            stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1

            # 系统性失败
            if record.details.is_systematic:
                stats['systematic_failures'] += 1

            # 可重复失败
            if record.details.is_repeatable:
                stats['repeatable_failures'] += 1

        return stats

    def stream_failures(self) -> Iterator[FailureRecord]:
        """流式读取失败记录"""
        if not os.path.exists(self.store_path):
            return

        try:
            with open(self.store_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            record = FailureRecord.from_dict(data)
                            yield record
                        except Exception as e:
                            logger.warning(f"Error parsing failure record: {e}")

        except Exception as e:
            logger.error(f"Error streaming failure records: {e}")

    def clear_old_failures(self, days: int = 30) -> int:
        """清除旧的失败记录"""
        cutoff_time = datetime.now() - timedelta(days=days)

        # 加载所有记录
        all_records = self.load_all_failures()

        # 过滤保留的记录
        kept_records = [
            record for record in all_records
            if record.timestamp > cutoff_time
        ]

        removed_count = len(all_records) - len(kept_records)

        # 重写文件
        try:
            with open(self.store_path, 'w', encoding='utf-8') as f:
                for record in kept_records:
                    json_line = json.dumps(record.to_dict(), ensure_ascii=False)
                    f.write(json_line + '\n')

            self.memory_cache = kept_records
            logger.info(f"Cleared {removed_count} old failure records")
            return removed_count

        except Exception as e:
            logger.error(f"Error clearing old failures: {e}")
            return 0

    def export_to_json(self, output_path: str) -> bool:
        """导出所有失败记录为 JSON"""
        try:
            all_records = self.load_all_failures()
            data = [record.to_dict() for record in all_records]

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"Exported {len(all_records)} records to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting failures: {e}")
            return False

    def get_store_size(self) -> Dict[str, Any]:
        """获取存储大小信息"""
        if not os.path.exists(self.store_path):
            return {'size_bytes': 0, 'size_mb': 0, 'record_count': 0}

        file_size = os.path.getsize(self.store_path)
        record_count = len(self.load_all_failures())

        return {
            'size_bytes': file_size,
            'size_mb': file_size / (1024 * 1024),
            'record_count': record_count,
        }

    # 私有方法

    def _ensure_cache_loaded(self) -> None:
        """确保缓存已加载"""
        if not self.cache_loaded:
            self.load_all_failures()
