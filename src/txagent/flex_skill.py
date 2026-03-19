import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional

from .txagent import TxAgent


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def similarity_score(text_a: str, text_b: str) -> float:
    return SequenceMatcher(None, normalize_text(text_a), normalize_text(text_b)).ratio()


class JsonlRepository:
    def __init__(self, path: str):
        self.path = path

    def ensure_parent(self) -> None:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        rows = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def overwrite(self, rows: List[Dict[str, Any]]) -> None:
        self.ensure_parent()
        with open(self.path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def append(self, row: Dict[str, Any]) -> None:
        self.ensure_parent()
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class RawFailureExperience:
    experience_id: str
    query: str
    state: str
    thought: str
    action: str
    tool: str
    observation: str
    error: str
    outcome: str
    error_type: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "query": self.query,
            "state": self.state,
            "thought": self.thought,
            "action": self.action,
            "tool": self.tool,
            "observation": self.observation,
            "error": self.error,
            "outcome": self.outcome,
            "error_type": self.error_type,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class ExperienceCluster:
    cluster_id: str
    signature: str
    summary: str
    member_ids: List[str]
    tool: str
    error_type: str
    updated_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "signature": self.signature,
            "summary": self.summary,
            "member_ids": self.member_ids,
            "tool": self.tool,
            "error_type": self.error_type,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


@dataclass
class ExecutableSkill:
    skill_id: str
    name: str
    pattern: str
    instruction: str
    preconditions: List[str]
    action_hint: str
    source_cluster_id: str
    supporting_experience_ids: List[str]
    confidence: float
    status: str
    version: int
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "pattern": self.pattern,
            "instruction": self.instruction,
            "preconditions": self.preconditions,
            "action_hint": self.action_hint,
            "source_cluster_id": self.source_cluster_id,
            "supporting_experience_ids": self.supporting_experience_ids,
            "confidence": self.confidence,
            "status": self.status,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


@dataclass
class FlexSkillConfig:
    raw_experience_path: str = "data/flexskill_raw_experiences.jsonl"
    cluster_path: str = "data/flexskill_clusters.jsonl"
    skill_bank_path: str = "data/flexskill_skill_bank.jsonl"
    online_top_k_failures: int = 2
    online_top_k_skills: int = 3
    cluster_min_size: int = 2
    consolidation_interval: int = 3
    enable_skill_generation: bool = True
    enable_background_refine: bool = True


class RawExperienceStore:
    def __init__(self, path: str):
        self.repo = JsonlRepository(path)
        self.rows = [RawFailureExperience(**row) for row in self.repo.load()]

    def add(self, row: RawFailureExperience) -> None:
        self.rows.append(row)
        self.repo.append(row.to_dict())

    def retrieve(self, query: str, top_k: int) -> List[RawFailureExperience]:
        scored = []
        for row in self.rows:
            score = max(
                similarity_score(query, row.query),
                similarity_score(query, row.error),
                similarity_score(query, row.outcome),
            )
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[:top_k]]


class ClusterStore:
    def __init__(self, path: str):
        self.repo = JsonlRepository(path)
        self.rows = [ExperienceCluster(**row) for row in self.repo.load()]

    def upsert(self, cluster: ExperienceCluster) -> None:
        for idx, row in enumerate(self.rows):
            if row.cluster_id == cluster.cluster_id:
                self.rows[idx] = cluster
                self.repo.overwrite([item.to_dict() for item in self.rows])
                return
        self.rows.append(cluster)
        self.repo.append(cluster.to_dict())

    def get_by_signature(self, signature: str) -> Optional[ExperienceCluster]:
        for row in self.rows:
            if row.signature == signature:
                return row
        return None


class ExecutableSkillBank:
    def __init__(self, path: str):
        self.repo = JsonlRepository(path)
        self.rows = [ExecutableSkill(**row) for row in self.repo.load()]

    def active_skills(self) -> List[ExecutableSkill]:
        return [row for row in self.rows if row.status == "active"]

    def retrieve(self, query: str, top_k: int) -> List[ExecutableSkill]:
        scored = []
        for row in self.active_skills():
            score = max(
                similarity_score(query, row.pattern),
                similarity_score(query, row.instruction),
                similarity_score(query, " ".join(row.preconditions)),
            )
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[:top_k]]

    def add_or_update(self, skill: ExecutableSkill) -> None:
        for idx, row in enumerate(self.rows):
            if row.pattern == skill.pattern and row.instruction == skill.instruction:
                row.supporting_experience_ids = sorted(
                    set(row.supporting_experience_ids + skill.supporting_experience_ids)
                )
                row.updated_at = skill.updated_at
                row.confidence = max(row.confidence, skill.confidence)
                row.version += 1
                self.rows[idx] = row
                self.repo.overwrite([item.to_dict() for item in self.rows])
                return
        self.rows.append(skill)
        self.repo.append(skill.to_dict())

    def summary(self) -> Dict[str, Any]:
        active = self.active_skills()
        return {
            "total_skills": len(self.rows),
            "active_skills": len(active),
            "patterns": [skill.pattern for skill in active[:10]],
        }


class FailureAttributor:
    def attribute(self, tool: str, error: str, outcome: str) -> str:
        error_text = normalize_text(error + " " + outcome)
        if "missing" in error_text and "parameter" in error_text:
            return "missing_parameter"
        if "format" in error_text or "json" in error_text:
            return "format_error"
        if "dosage" in error_text:
            return "missing_prerequisite_information"
        if "timeout" in error_text or "connection" in error_text:
            return "tool_unavailable"
        if "not found" in error_text or "no result" in error_text:
            return "retrieval_failure"
        if "disabled" in error_text:
            return "tool_disabled"
        if "error" in error_text or "failed" in error_text:
            return f"{tool}_execution_failure"
        return "unknown_failure"


class ExperienceClusterer:
    def signature(self, experience: RawFailureExperience) -> str:
        return f"{experience.tool}|{experience.error_type}"

    def summarize(self, experiences: List[RawFailureExperience]) -> str:
        if not experiences:
            return ""
        tool = experiences[0].tool
        error_type = experiences[0].error_type
        return f"Repeated {error_type} failures when using {tool}."


class StableSkillSynthesizer:
    def __init__(self, generator_fn: Optional[Callable[[str], str]] = None):
        self.generator_fn = generator_fn

    def build_prompt(
        self,
        cluster: ExperienceCluster,
        experiences: List[RawFailureExperience],
    ) -> str:
        lines = [
            "You are generating a stable executable skill for a continual tool-using agent.",
            "Convert clustered failures into one reusable skill.",
            "The skill must be precise, action-oriented, and executable.",
            "Avoid vague advice, motivational wording, or unverifiable claims.",
            "Respond in JSON with keys: name, pattern, instruction, preconditions, action_hint.",
            "",
            f"Cluster summary: {cluster.summary}",
            f"Tool: {cluster.tool}",
            f"Error type: {cluster.error_type}",
            "Examples:",
        ]
        for idx, exp in enumerate(experiences[:5], start=1):
            lines.append(f"{idx}. state: {exp.state}")
            lines.append(f"{idx}. action: {exp.action}")
            lines.append(f"{idx}. error: {exp.error}")
            lines.append(f"{idx}. outcome: {exp.outcome}")
        return "\n".join(lines)

    def synthesize(
        self,
        cluster: ExperienceCluster,
        experiences: List[RawFailureExperience],
    ) -> Optional[Dict[str, Any]]:
        if self.generator_fn is not None:
            try:
                output = self.generator_fn(self.build_prompt(cluster, experiences))
                parsed = self.parse(output)
                if parsed.get("instruction"):
                    return parsed
            except Exception:
                pass
        return self.heuristic_synthesize(cluster)

    def parse(self, output: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return {
            "name": str(payload.get("name", "")).strip(),
            "pattern": str(payload.get("pattern", "")).strip(),
            "instruction": str(payload.get("instruction", "")).strip(),
            "preconditions": payload.get("preconditions", []),
            "action_hint": str(payload.get("action_hint", "")).strip(),
        }

    def heuristic_synthesize(self, cluster: ExperienceCluster) -> Dict[str, Any]:
        tool = cluster.tool
        error_type = cluster.error_type
        if error_type == "missing_parameter":
            return {
                "name": f"Validate {tool} arguments",
                "pattern": f"{tool} calls fail when required parameters are missing",
                "instruction": f"Before calling {tool}, verify all required parameters are present and normalized.",
                "preconditions": [f"{tool} selected", "tool arguments assembled"],
                "action_hint": "Run an argument completeness check before execution.",
            }
        if error_type == "missing_prerequisite_information":
            return {
                "name": f"Retrieve prerequisites for {tool}",
                "pattern": f"{tool} calls fail when prerequisite clinical information is absent",
                "instruction": f"Before calling {tool}, retrieve prerequisite information such as dosage, timing, or patient attributes.",
                "preconditions": [f"{tool} selected", "question lacks prerequisite details"],
                "action_hint": "Insert a prerequisite-retrieval step before the tool call.",
            }
        if error_type == "format_error":
            return {
                "name": "Validate function call format",
                "pattern": "tool calls fail because the function-call payload is malformed",
                "instruction": "Validate function-call JSON schema and required argument names before execution.",
                "preconditions": ["tool call prepared"],
                "action_hint": "Run schema validation before sending the tool call.",
            }
        return {
            "name": f"Stabilize {tool} usage",
            "pattern": cluster.summary,
            "instruction": f"Check prerequisites and expected inputs before using {tool}.",
            "preconditions": [f"{tool} selected"],
            "action_hint": "Add a pre-execution verification step.",
        }


class SkillStabilizer:
    def validate(
        self,
        candidate: Dict[str, Any],
        existing_skills: List[ExecutableSkill],
    ) -> Optional[Dict[str, Any]]:
        instruction = candidate.get("instruction", "").strip()
        pattern = candidate.get("pattern", "").strip()
        if not instruction or not pattern:
            return None
        if len(instruction.split()) < 5:
            return None
        banned_terms = ["maybe", "possibly", "try to", "could be", "something like"]
        lower_instruction = instruction.lower()
        if any(term in lower_instruction for term in banned_terms):
            return None
        for skill in existing_skills:
            if similarity_score(skill.instruction, instruction) > 0.92:
                return None
        preconditions = candidate.get("preconditions", [])
        if not isinstance(preconditions, list):
            preconditions = [str(preconditions)]
        return {
            "name": candidate.get("name", "Unnamed skill").strip() or "Unnamed skill",
            "pattern": pattern,
            "instruction": instruction,
            "preconditions": [str(item).strip() for item in preconditions if str(item).strip()],
            "action_hint": candidate.get("action_hint", "").strip(),
        }


class FlexSkillController:
    def __init__(
        self,
        config: FlexSkillConfig,
        generator_fn: Optional[Callable[[str], str]] = None,
    ):
        self.config = config
        self.raw_store = RawExperienceStore(config.raw_experience_path)
        self.cluster_store = ClusterStore(config.cluster_path)
        self.skill_bank = ExecutableSkillBank(config.skill_bank_path)
        self.attributor = FailureAttributor()
        self.clusterer = ExperienceClusterer()
        self.synthesizer = StableSkillSynthesizer(generator_fn=generator_fn)
        self.stabilizer = SkillStabilizer()

    def retrieve_guidance(self, query: str) -> str:
        failures = self.raw_store.retrieve(query, self.config.online_top_k_failures)
        skills = self.skill_bank.retrieve(query, self.config.online_top_k_skills)
        if not failures and not skills:
            return ""

        lines = [
            "FLEX-Skill guidance:",
            "Use validated skills first. Use raw failures only as backup context.",
        ]
        if skills:
            lines.append("Validated executable skills:")
            for idx, skill in enumerate(skills, start=1):
                lines.append(f"{idx}. Pattern: {skill.pattern}")
                lines.append(f"{idx}. Instruction: {skill.instruction}")
                if skill.preconditions:
                    lines.append(f"{idx}. Preconditions: {', '.join(skill.preconditions)}")
        if failures:
            lines.append("Related raw failures:")
            for idx, exp in enumerate(failures, start=1):
                lines.append(f"{idx}. Tool: {exp.tool}")
                lines.append(f"{idx}. Error: {exp.error}")
                lines.append(f"{idx}. Outcome: {exp.outcome}")
        lines.append("Prefer actions that satisfy the validated skills before tool execution.")
        return "\n".join(lines)

    def observe_failure(
        self,
        query: str,
        state: str,
        thought: str,
        action: str,
        tool: str,
        observation: str,
        error: str,
        outcome: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RawFailureExperience:
        error_type = self.attributor.attribute(tool, error, outcome)
        exp = RawFailureExperience(
            experience_id=str(uuid.uuid4()),
            query=query,
            state=state,
            thought=thought,
            action=action,
            tool=tool,
            observation=observation,
            error=error,
            outcome=outcome,
            error_type=error_type,
            timestamp=datetime.utcnow().isoformat() + "Z",
            metadata=metadata or {},
        )
        self.raw_store.add(exp)
        self._online_update_cluster(exp)
        if self.config.enable_background_refine and len(self.raw_store.rows) % self.config.consolidation_interval == 0:
            self.consolidate()
        return exp

    def _online_update_cluster(self, exp: RawFailureExperience) -> None:
        signature = self.clusterer.signature(exp)
        cluster = self.cluster_store.get_by_signature(signature)
        now = datetime.utcnow().isoformat() + "Z"
        if cluster is None:
            cluster = ExperienceCluster(
                cluster_id=str(uuid.uuid4()),
                signature=signature,
                summary=self.clusterer.summarize([exp]),
                member_ids=[exp.experience_id],
                tool=exp.tool,
                error_type=exp.error_type,
                updated_at=now,
            )
        else:
            cluster.member_ids = sorted(set(cluster.member_ids + [exp.experience_id]))
            member_rows = [row for row in self.raw_store.rows if row.experience_id in cluster.member_ids]
            cluster.summary = self.clusterer.summarize(member_rows)
            cluster.updated_at = now
        self.cluster_store.upsert(cluster)

    def consolidate(self) -> None:
        if not self.config.enable_skill_generation:
            return

        for cluster in self.cluster_store.rows:
            if len(cluster.member_ids) < self.config.cluster_min_size:
                continue
            member_rows = [row for row in self.raw_store.rows if row.experience_id in cluster.member_ids]
            candidate = self.synthesizer.synthesize(cluster, member_rows)
            if not candidate:
                continue
            stabilized = self.stabilizer.validate(candidate, self.skill_bank.active_skills())
            if stabilized is None:
                continue

            now = datetime.utcnow().isoformat() + "Z"
            skill = ExecutableSkill(
                skill_id=str(uuid.uuid4()),
                name=stabilized["name"],
                pattern=stabilized["pattern"],
                instruction=stabilized["instruction"],
                preconditions=stabilized["preconditions"],
                action_hint=stabilized["action_hint"],
                source_cluster_id=cluster.cluster_id,
                supporting_experience_ids=cluster.member_ids,
                confidence=min(1.0, 0.5 + 0.1 * len(cluster.member_ids)),
                status="active",
                version=1,
                created_at=now,
                updated_at=now,
                metadata={"error_type": cluster.error_type, "tool": cluster.tool},
            )
            self.skill_bank.add_or_update(skill)

    def build_status_snapshot(self) -> Dict[str, Any]:
        return {
            "raw_experience_count": len(self.raw_store.rows),
            "cluster_count": len(self.cluster_store.rows),
            "skill_bank": self.skill_bank.summary(),
            "config": {
                "online_top_k_failures": self.config.online_top_k_failures,
                "online_top_k_skills": self.config.online_top_k_skills,
                "cluster_min_size": self.config.cluster_min_size,
                "consolidation_interval": self.config.consolidation_interval,
            },
        }

    def export_status(self, path: str) -> None:
        repo = JsonlRepository(path)
        repo.ensure_parent()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.build_status_snapshot(), handle, indent=2, ensure_ascii=False)


class FlexSkillTxAgent(TxAgent):
    def __init__(
        self,
        model_name,
        rag_model_name,
        flexskill_config: Optional[FlexSkillConfig] = None,
        skill_generator_use_llm: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(model_name, rag_model_name, *args, **kwargs)
        self.flexskill_config = flexskill_config or FlexSkillConfig()
        generator_fn = self.run_skill_generator_prompt if skill_generator_use_llm else None
        self.flexskill = FlexSkillController(
            config=self.flexskill_config,
            generator_fn=generator_fn,
        )
        self.flexskill_skill_generator_use_llm = skill_generator_use_llm

    def run_skill_generator_prompt(self, prompt):
        return self.run_self_agent(
            message=prompt,
            temperature=0.1,
            max_new_tokens=256,
            max_token=8192,
        )

    def initialize_conversation(self, message, conversation=None, history=None):
        conversation = super().initialize_conversation(
            message,
            conversation=conversation,
            history=history,
        )
        guidance = self.flexskill.retrieve_guidance(message)
        if guidance:
            flex_message = {"role": "system", "content": guidance}
            if len(conversation) > 1 and conversation[1]["role"] == "system" and conversation[1]["content"].startswith("FLEX-Skill guidance:"):
                conversation[1] = flex_message
            else:
                conversation.insert(1, flex_message)
        return conversation

    def detect_failure_result(self, call_result):
        result_text = str(call_result).lower()
        failure_markers = [
            "error",
            "failed",
            "exception",
            "not a valid function call",
            "missing parameter",
            "timeout",
            "disabled",
        ]
        return any(marker in result_text for marker in failure_markers)

    def observe_flex_failure(
        self,
        query: str,
        thought: str,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]],
        call_result: str,
        source: str,
    ) -> None:
        self.flexskill.observe_failure(
            query=query,
            state=query,
            thought=thought,
            action=json.dumps(tool_args or {}, ensure_ascii=False),
            tool=tool_name,
            observation=str(call_result),
            error=str(call_result),
            outcome=str(call_result),
            metadata={"source": source},
        )

    def _extract_tool_sequence(self, revised_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not revised_messages:
            return []
        assistant_message = revised_messages[0]
        raw_tool_calls = assistant_message.get("tool_calls")
        if raw_tool_calls is None:
            return []
        if isinstance(raw_tool_calls, str):
            try:
                raw_tool_calls = json.loads(raw_tool_calls)
            except json.JSONDecodeError:
                return []
        if not isinstance(raw_tool_calls, list):
            return []
        return raw_tool_calls

    def run_function_call(
        self,
        fcall_str,
        return_message=False,
        existing_tools_prompt=None,
        message_for_call_agent=None,
        call_agent=False,
        call_agent_level=None,
        temperature=None,
    ):
        revised_messages, existing_tools_prompt, special_tool_call = super().run_function_call(
            fcall_str,
            return_message=return_message,
            existing_tools_prompt=existing_tools_prompt,
            message_for_call_agent=message_for_call_agent,
            call_agent=call_agent,
            call_agent_level=call_agent_level,
            temperature=temperature,
        )
        tool_sequence = self._extract_tool_sequence(revised_messages)
        tool_index = 0
        for message in revised_messages:
            if message["role"] != "tool":
                continue
            payload = json.loads(message["content"])
            if self.detect_failure_result(payload.get("content", "")):
                tool_info = tool_sequence[tool_index] if tool_index < len(tool_sequence) else {}
                self.observe_flex_failure(
                    query=message_for_call_agent or "",
                    thought=fcall_str,
                    tool_name=tool_info.get("name", "tool_call"),
                    tool_args=tool_info.get("arguments", {}),
                    call_result=payload.get("content", ""),
                    source="run_function_call",
                )
            tool_index += 1
        return revised_messages, existing_tools_prompt, special_tool_call

    def run_function_call_stream(
        self,
        fcall_str,
        return_message=False,
        existing_tools_prompt=None,
        message_for_call_agent=None,
        call_agent=False,
        call_agent_level=None,
        temperature=None,
        return_gradio_history=True,
    ):
        result = yield from super().run_function_call_stream(
            fcall_str,
            return_message=return_message,
            existing_tools_prompt=existing_tools_prompt,
            message_for_call_agent=message_for_call_agent,
            call_agent=call_agent,
            call_agent_level=call_agent_level,
            temperature=temperature,
            return_gradio_history=return_gradio_history,
        )
        revised_messages = result[0]
        tool_sequence = self._extract_tool_sequence(revised_messages)
        tool_index = 0
        for message in revised_messages:
            if message["role"] != "tool":
                continue
            payload = json.loads(message["content"])
            if self.detect_failure_result(payload.get("content", "")):
                tool_info = tool_sequence[tool_index] if tool_index < len(tool_sequence) else {}
                self.observe_flex_failure(
                    query=message_for_call_agent or "",
                    thought=fcall_str,
                    tool_name=tool_info.get("name", "tool_call"),
                    tool_args=tool_info.get("arguments", {}),
                    call_result=payload.get("content", ""),
                    source="run_function_call_stream",
                )
            tool_index += 1
        return result
