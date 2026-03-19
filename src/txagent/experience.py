import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def similarity_score(text_a: str, text_b: str) -> float:
    return SequenceMatcher(None, normalize_text(text_a), normalize_text(text_b)).ratio()


@dataclass
class FailureRecord:
    state: str
    action: str
    tool: str
    error: str
    outcome: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "action": self.action,
            "tool": self.tool,
            "error": self.error,
            "outcome": self.outcome,
            "timestamp": self.timestamp,
            "metadata": self.metadata or {},
        }


@dataclass
class SkillRecord:
    pattern: str
    skill: str
    source_error: str
    source_tool: str
    created_at: str
    updated_at: str
    usage_count: int = 1
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern,
            "skill": self.skill,
            "source_error": self.source_error,
            "source_tool": self.source_tool,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "usage_count": self.usage_count,
            "metadata": self.metadata or {},
        }


class JsonlStore:
    def __init__(self, path: str):
        self.path = path

    def _ensure_parent(self) -> None:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def load_lines(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        payload = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    payload.append(json.loads(line))
        return payload

    def overwrite_lines(self, rows: List[Dict[str, Any]]) -> None:
        self._ensure_parent()
        with open(self.path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def append_line(self, row: Dict[str, Any]) -> None:
        self._ensure_parent()
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class FailureMemoryStore:
    def __init__(self, path: str):
        self.store = JsonlStore(path)
        self.records = [FailureRecord(**row) for row in self.store.load_lines()]

    def add(self, record: FailureRecord) -> None:
        self.records.append(record)
        self.store.append_line(record.to_dict())

    def retrieve(self, query: str, top_k: int = 3) -> List[FailureRecord]:
        scored = []
        for record in self.records:
            score = max(
                similarity_score(query, record.state),
                similarity_score(query, record.error),
                similarity_score(query, record.outcome),
            )
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:top_k]]


class SkillBank:
    def __init__(self, path: str):
        self.store = JsonlStore(path)
        self.skills = [SkillRecord(**row) for row in self.store.load_lines()]

    def add_or_update(self, skill_record: SkillRecord) -> None:
        for existing in self.skills:
            if existing.pattern == skill_record.pattern and existing.skill == skill_record.skill:
                existing.usage_count += 1
                existing.updated_at = skill_record.updated_at
                rows = [skill.to_dict() for skill in self.skills]
                self.store.overwrite_lines(rows)
                return

        self.skills.append(skill_record)
        self.store.append_line(skill_record.to_dict())

    def retrieve(self, query: str, top_k: int = 3) -> List[SkillRecord]:
        scored = []
        for skill in self.skills:
            score = max(
                similarity_score(query, skill.pattern),
                similarity_score(query, skill.skill),
                similarity_score(query, skill.source_error),
            )
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [skill for _, skill in scored[:top_k]]


class FailureToSkillGenerator:
    def __init__(self, generator_fn: Optional[Callable[[str], str]] = None):
        self.generator_fn = generator_fn

    def generate(self, record: FailureRecord) -> Optional[SkillRecord]:
        pattern = None
        skill_text = None

        if self.generator_fn is not None:
            prompt = self.build_prompt(record)
            try:
                output = self.generator_fn(prompt)
                parsed = self.parse_generation(output)
                pattern = parsed.get("pattern")
                skill_text = parsed.get("skill")
            except Exception:
                pattern = None
                skill_text = None

        if not pattern:
            pattern = self._build_pattern(record)
        if not skill_text:
            skill_text = self._build_skill(record)
        if skill_text is None:
            return None

        now = datetime.utcnow().isoformat() + "Z"
        return SkillRecord(
            pattern=pattern,
            skill=skill_text,
            source_error=record.error,
            source_tool=record.tool,
            created_at=now,
            updated_at=now,
            metadata={"source_state": record.state},
        )

    def build_prompt(self, record: FailureRecord) -> str:
        return (
            "You are a skill generator for an agent.\n"
            "Your job is to convert one concrete failure case into:\n"
            "1. A concise reusable failure pattern.\n"
            "2. A generalized skill that helps avoid the same mistake.\n\n"
            "Requirements:\n"
            "- The pattern should describe the recurring failure type, not the full case.\n"
            "- The skill should be actionable, general, and reusable.\n"
            "- Do not mention specific timestamps.\n"
            "- Keep both fields short.\n"
            "- Respond in JSON only.\n\n"
            "Output schema:\n"
            "{\"pattern\": \"...\", \"skill\": \"...\"}\n\n"
            "Failure case:\n"
            f"state: {record.state}\n"
            f"action: {record.action}\n"
            f"tool: {record.tool}\n"
            f"error: {record.error}\n"
            f"outcome: {record.outcome}\n"
        )

    def parse_generation(self, output: str) -> Dict[str, str]:
        output = output.strip()
        json_match = re.search(r"\{.*\}", output, re.DOTALL)
        if json_match:
            try:
                payload = json.loads(json_match.group(0))
                return {
                    "pattern": str(payload.get("pattern", "")).strip(),
                    "skill": str(payload.get("skill", "")).strip(),
                }
            except json.JSONDecodeError:
                pass

        pattern_match = re.search(r"pattern\s*:\s*(.+)", output, re.IGNORECASE)
        skill_match = re.search(r"skill\s*:\s*(.+)", output, re.IGNORECASE)
        return {
            "pattern": pattern_match.group(1).strip() if pattern_match else "",
            "skill": skill_match.group(1).strip() if skill_match else "",
        }

    def _build_pattern(self, record: FailureRecord) -> str:
        return f"{record.tool}: {record.error}".strip()

    def _build_skill(self, record: FailureRecord) -> Optional[str]:
        error_text = normalize_text(record.error + " " + record.outcome)
        tool_name = record.tool

        if "missing" in error_text and "parameter" in error_text:
            return f"Before calling {tool_name}, verify that all required parameters are present and well-formed."
        if "dosage" in error_text:
            return f"Before calling {tool_name}, retrieve dosage information and include it in the tool arguments."
        if "not a valid function call" in error_text or "invalid function call" in error_text:
            return "Before executing tools, validate the function-call JSON format and required arguments."
        if "disabled" in error_text:
            return f"Do not rely on {tool_name} when it is disabled; choose an alternative tool or continue reasoning without it."
        if "timeout" in error_text or "connection" in error_text:
            return f"When {tool_name} is unavailable, retry conservatively or switch to a fallback evidence source."
        if "error" in error_text or "failed" in error_text:
            return f"Use {tool_name} only after checking prerequisites and expected input fields."
        return None


def build_experience_context(
    query: str,
    failure_memory: Optional[FailureMemoryStore],
    skill_bank: Optional[SkillBank],
    top_k: int = 2,
) -> str:
    if failure_memory is None and skill_bank is None:
        return ""

    failures = failure_memory.retrieve(query, top_k=top_k) if failure_memory is not None else []
    skills = skill_bank.retrieve(query, top_k=top_k) if skill_bank is not None else []

    if not failures and not skills:
        return ""

    lines = [
        "Experience guidance:",
        "Use retrieved failure patterns and skills to avoid repeating earlier mistakes.",
    ]

    if skills:
        lines.append("Retrieved skills:")
        for idx, skill in enumerate(skills, start=1):
            lines.append(f"{idx}. Pattern: {skill.pattern}")
            lines.append(f"{idx}. Skill: {skill.skill}")

    if failures:
        lines.append("Retrieved failures:")
        for idx, failure in enumerate(failures, start=1):
            lines.append(f"{idx}. Tool: {failure.tool}")
            lines.append(f"{idx}. Error: {failure.error}")
            lines.append(f"{idx}. Outcome: {failure.outcome}")

    lines.append("Prefer actions that satisfy the retrieved skills before calling tools.")
    return "\n".join(lines)
