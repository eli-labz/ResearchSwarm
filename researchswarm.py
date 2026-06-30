"""ResearchSwarm CLI entrypoint.

Routes a task through the Digital Cognitive Labor classifier before deciding
whether the work can be completed digitally, must be handed off to a human, or
should be split into a hybrid workflow.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Protocol

from researchswarm_agent import DigitalCognitiveLaborAgent, TaskDomain
from researchswarm_memory import DEFAULT_MEMORY_DB_PATH, ResearchSwarmMemoryStore
from researchswarm_wiki import ResearchSwarmWiki


class ExecutionStatus(str, Enum):
    DIGITAL_EXECUTION = "digital-execution"
    HUMAN_HANDOFF_REQUIRED = "human-handoff-required"
    HYBRID_WORKFLOW = "hybrid-workflow"
    CLARIFICATION_REQUIRED = "clarification-required"


RESEARCHSWARM_COMMAND_HINTS = (
    ({"prepare", "tokenizer", "dataset", "download", "data"}, "uv run prepare.py"),
    ({"train", "experiment", "baseline", "benchmark", "eval", "training"}, "uv run train.py"),
)

LOOP_ENGINEERING_ROOT = Path(__file__).resolve().parent / "thirdparty" / "loop-engineering-main"
LOOP_PATTERN_REGISTRY = LOOP_ENGINEERING_ROOT / "patterns" / "registry.yaml"


@dataclass
class RoutedTask:
    instruction: str
    domain: str
    status: ExecutionStatus
    confidence: float
    cognitive_functions: list[str] = field(default_factory=list)
    digital_work: list[str] = field(default_factory=list)
    human_handoff: list[str] = field(default_factory=list)
    suggested_commands: list[str] = field(default_factory=list)
    verification_targets: list[str] = field(default_factory=list)
    next_action: str = ""
    executor_name: str = ""
    execution_summary: str = ""
    execution_artifact: str = ""
    memory_context: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    def to_text(self) -> str:
        lines = [
            f"instruction: {self.instruction}",
            f"domain: {self.domain}",
            f"status: {self.status.value}",
            f"confidence: {self.confidence}",
        ]
        if self.cognitive_functions:
            lines.append(f"cognitive_functions: {', '.join(self.cognitive_functions)}")
        if self.digital_work:
            lines.append("digital_work:")
            lines.extend(f"- {item}" for item in self.digital_work)
        if self.human_handoff:
            lines.append("human_handoff:")
            lines.extend(f"- {item}" for item in self.human_handoff)
        if self.suggested_commands:
            lines.append("suggested_commands:")
            lines.extend(f"- {item}" for item in self.suggested_commands)
        if self.verification_targets:
            lines.append("verification_targets:")
            lines.extend(f"- {item}" for item in self.verification_targets)
        if self.executor_name:
            lines.append(f"executor_name: {self.executor_name}")
        if self.execution_summary:
            lines.append(f"execution_summary: {self.execution_summary}")
        if self.execution_artifact:
            lines.append("execution_artifact:")
            lines.extend(f"  {item}" for item in self.execution_artifact.splitlines())
        if self.memory_context:
            lines.append("memory_context:")
            lines.extend(f"- {item}" for item in self.memory_context)
        lines.append(f"next_action: {self.next_action}")
        return "\n".join(lines)


@dataclass
class ExecutionContext:
    instruction: str
    input_text: str = ""
    file_path: str = ""
    run_prepare: bool = False
    run_train: bool = False

    def load_source_text(self) -> str:
        if self.input_text:
            return self.input_text
        if self.file_path:
            path = Path(self.file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {self.file_path}")
            if any(part.lower().startswith(".") and "env" in part.lower() for part in path.parts):
                raise PermissionError("Refusing to read a secret-like environment file.")
            return path.read_text(encoding="utf-8", errors="replace")
        return ""


@dataclass
class ExecutorResult:
    executor_name: str
    summary: str
    artifact: str


@dataclass
class LoopPattern:
    pattern_id: str
    name: str
    file_name: str
    goal: str = ""
    cadence: str = ""
    risk: str = ""


class TaskExecutor(Protocol):
    name: str

    def can_handle(self, task: RoutedTask, context: ExecutionContext) -> bool:
        ...

    def execute(self, task: RoutedTask, context: ExecutionContext) -> ExecutorResult:
        ...


class ResearchWorkflowExecutor:
    name = "research-workflow"
    KEYWORDS = {"train", "training", "experiment", "baseline", "prepare", "tokenizer", "dataset", "overnight", "research"}

    def can_handle(self, task: RoutedTask, context: ExecutionContext) -> bool:
        lowered = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", context.instruction.lower()))
        return bool(lowered & self.KEYWORDS) or bool(task.suggested_commands)

    def execute(self, task: RoutedTask, context: ExecutionContext) -> ExecutorResult:
        commands = task.suggested_commands or ["uv run prepare.py", "uv run train.py"]
        artifact_lines = [
            "# ResearchSwarm Overnight Experiment Plan",
            "",
            "## Primary Goal",
            "Give an AI agent a small but real LLM training setup and let it experiment autonomously overnight.",
            "",
            "## Command Sequence",
            *[f"- {command}" for command in commands],
            "",
            "## Control Surface",
            "- Use program.md for the dedicated autonomous research loop.",
            "- Use train.py as the only mutable training surface.",
            "- Use prepare.py only for fixed data preparation and evaluation setup.",
            "",
            "## Success Criteria",
            "- Baseline run completes within the fixed time budget.",
            "- val_bpb is captured and compared against future experiments.",
            "- Results are logged so the branch can advance only on improvement.",
        ]

        action_results: list[str] = []
        if context.run_prepare:
            action_results.append(self._run_script("prepare.py"))
        if context.run_train:
            action_results.append(self._run_script("train.py"))

        if action_results:
            artifact_lines.extend([
                "",
                "## Executed Actions",
                *action_results,
            ])
            summary = "Executed the requested ResearchSwarm workflow actions and captured their results."
        else:
            artifact_lines.extend([
                "",
                "## Action Mode",
                "- No scripts were launched because no explicit run flags were supplied.",
                "- Use --run-prepare and/or --run-train to execute the overnight workflow steps.",
            ])
            summary = "Generated a concrete overnight experiment plan centered on the training workflow."

        return ExecutorResult(
            executor_name=self.name,
            summary=summary,
            artifact="\n".join(artifact_lines),
        )

    def _run_script(self, script_name: str) -> str:
        command = self._resolve_command(script_name)
        repo_root = Path(__file__).resolve().parent
        result = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        stdout_tail = self._tail_text(result.stdout)
        stderr_tail = self._tail_text(result.stderr)
        lines = [
            f"- {' '.join(command)}",
            f"  exit_code: {result.returncode}",
        ]
        if stdout_tail:
            lines.append("  stdout_tail:")
            lines.extend(f"    {line}" for line in stdout_tail.splitlines())
        if stderr_tail:
            lines.append("  stderr_tail:")
            lines.extend(f"    {line}" for line in stderr_tail.splitlines())
        return "\n".join(lines)

    def _resolve_command(self, script_name: str) -> list[str]:
        if shutil.which("uv"):
            return ["uv", "run", script_name]
        if shutil.which("py"):
            return ["py", "-3", script_name]
        return [sys.executable, script_name]

    def _tail_text(self, text: str, max_lines: int = 20) -> str:
        if not text:
            return ""
        lines = text.strip().splitlines()
        return "\n".join(lines[-max_lines:])


class SummarizationExecutor:
    name = "summarizer"
    KEYWORDS = {"summarize", "summarise", "summary", "condense", "recap"}

    def can_handle(self, task: RoutedTask, context: ExecutionContext) -> bool:
        lowered = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", context.instruction.lower()))
        return bool(lowered & self.KEYWORDS)

    def execute(self, task: RoutedTask, context: ExecutionContext) -> ExecutorResult:
        source = context.load_source_text().strip() or context.instruction
        sentences = re.split(r"(?<=[.!?])\s+|\n+", source)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        summary = " ".join(sentences[:3]) if sentences else source
        artifact = "\n".join([
            "# Summary",
            "",
            summary,
        ])
        return ExecutorResult(
            executor_name=self.name,
            summary="Produced a concise text summary.",
            artifact=artifact,
        )


class ReportExecutor:
    name = "report-generator"
    KEYWORDS = {"report", "postmortem", "brief", "memo"}

    def can_handle(self, task: RoutedTask, context: ExecutionContext) -> bool:
        lowered = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", context.instruction.lower()))
        return bool(lowered & self.KEYWORDS)

    def execute(self, task: RoutedTask, context: ExecutionContext) -> ExecutorResult:
        source = context.load_source_text().strip()
        if not source:
            source = context.instruction
        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", source) if segment.strip()]
        highlights = sentences[:3] if sentences else [source]
        artifact_lines = [
            "# ResearchSwarm Report",
            "",
            "## Objective",
            context.instruction,
            "",
            "## Findings",
        ]
        artifact_lines.extend(f"- {item}" for item in highlights)
        artifact_lines.extend([
            "",
            "## Next Steps",
            "- Validate the artifact against the original request.",
            "- If this supports the overnight training workflow, feed the result back into program.md or results analysis.",
        ])
        return ExecutorResult(
            executor_name=self.name,
            summary="Generated a structured markdown report.",
            artifact="\n".join(artifact_lines),
        )


class WikiExecutor:
    """Maintains the markdown knowledge layer beside the SQLite memory ledger."""

    name = "wiki"

    def __init__(self, wiki: ResearchSwarmWiki) -> None:
        self.wiki = wiki

    def can_handle(self, task: RoutedTask, context: ExecutionContext) -> bool:
        lowered = context.instruction.lower()
        return bool(re.search(r"\bwiki\s+(ingest|query|lint|init)\b", lowered))

    def execute(self, task: RoutedTask, context: ExecutionContext) -> ExecutorResult:
        instruction = context.instruction.strip()

        if re.search(r"\bwiki\s+lint\b", instruction, flags=re.IGNORECASE):
            artifact = self.wiki.lint()
            return ExecutorResult(
                executor_name=self.name,
                summary="Completed LLM-Wiki lint health-check.",
                artifact=artifact,
            )

        query_match = re.search(r"\bwiki\s+query\s+(.+)$", instruction, flags=re.IGNORECASE)
        if query_match:
            question = query_match.group(1).strip()
            artifact = self.wiki.query(question)
            return ExecutorResult(
                executor_name=self.name,
                summary="Answered from LLM-Wiki synthesized pages with citations.",
                artifact=artifact,
            )

        ingest_match = re.search(r"\bwiki\s+ingest\s+(.+)$", instruction, flags=re.IGNORECASE)
        if ingest_match:
            source = ingest_match.group(1).strip().strip('"').strip("'")
            batch_match = re.match(r"--batch\s+(.+)$", source, flags=re.IGNORECASE)
            if batch_match:
                artifact = self.wiki.ingest_batch(batch_match.group(1).strip().strip('"').strip("'"))
            else:
                artifact = self.wiki.ingest(source)
            return ExecutorResult(
                executor_name=self.name,
                summary="Ingested a source into the LLM-Wiki knowledge layer.",
                artifact=artifact,
            )

        if re.search(r"\bwiki\s+init\b", instruction, flags=re.IGNORECASE):
            self.wiki.init()
            artifact = "\n".join(
                [
                    "# Wiki Init",
                    "",
                    "Initialized LLM-Wiki directory structure and schema files.",
                ]
            )
            return ExecutorResult(
                executor_name=self.name,
                summary="Initialized the LLM-Wiki scaffold.",
                artifact=artifact,
            )

        return ExecutorResult(
            executor_name=self.name,
            summary="No wiki sub-command matched. Use wiki ingest, wiki query, wiki lint, or wiki init.",
            artifact="No wiki command executed.",
        )


class LoopEngineeringExecutor:
    """Adapter that activates reusable loop-engineering patterns for digital labor."""

    name = "loop-engineering"
    KEYWORDS = {
        "loop",
        "triage",
        "sweeper",
        "babysitter",
        "changelog",
        "cleanup",
        "dependency",
        "issue",
        "daily",
        "ci",
        "pattern",
        "registry",
    }

    def can_handle(self, task: RoutedTask, context: ExecutionContext) -> bool:
        lowered = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", context.instruction.lower()))
        return bool(lowered & self.KEYWORDS) or "loop-engineering" in context.instruction.lower()

    def execute(self, task: RoutedTask, context: ExecutionContext) -> ExecutorResult:
        patterns = self._load_patterns()
        selected = self._select_pattern(context.instruction, patterns)

        if not selected:
            artifact = "\n".join(
                [
                    "# Loop Engineering Integration",
                    "",
                    f"- source_root: {LOOP_ENGINEERING_ROOT}",
                    "- status: no-matching-pattern",
                    "",
                    "## Guidance",
                    "- Use one of the known patterns from thirdparty/loop-engineering-main/patterns/.",
                    "- Keep overnight training as the default workflow for model experimentation.",
                ]
            )
            return ExecutorResult(
                executor_name=self.name,
                summary="Loop-engineering integration is available but no specific pattern matched the instruction.",
                artifact=artifact,
            )

        pattern_file = LOOP_ENGINEERING_ROOT / "patterns" / selected.file_name
        preview = self._render_pattern_preview(pattern_file)
        artifact_lines = [
            "# Loop Engineering Activation",
            "",
            f"- source_root: {LOOP_ENGINEERING_ROOT}",
            f"- selected_pattern_id: {selected.pattern_id}",
            f"- selected_pattern_name: {selected.name}",
            f"- selected_pattern_file: {pattern_file}",
        ]
        if selected.goal:
            artifact_lines.append(f"- goal: {selected.goal}")
        if selected.cadence:
            artifact_lines.append(f"- cadence: {selected.cadence}")
        if selected.risk:
            artifact_lines.append(f"- risk: {selected.risk}")

        artifact_lines.extend(
            [
                "",
                "## Cognitive-Control Alignment",
                "- Route text-based fragments to digital execution now.",
                "- Preserve human-action fragments as explicit handoff items.",
                "- Split mixed prompts into digital work, human handoff, and clarifications.",
                "",
                "## Pattern Preview",
                preview,
            ]
        )

        if task.human_handoff:
            artifact_lines.extend(
                [
                    "",
                    "## Human Handoff Boundary",
                    *[f"- {item}" for item in task.human_handoff],
                ]
            )

        return ExecutorResult(
            executor_name=self.name,
            summary=f"Activated loop-engineering pattern '{selected.name}' from thirdparty source.",
            artifact="\n".join(artifact_lines),
        )

    def _load_patterns(self) -> list[LoopPattern]:
        if not LOOP_PATTERN_REGISTRY.exists():
            return []

        lines = LOOP_PATTERN_REGISTRY.read_text(encoding="utf-8", errors="replace").splitlines()
        patterns: list[LoopPattern] = []
        current: dict[str, str] = {}

        for raw_line in lines:
            line = raw_line.strip()
            if line.startswith("- id:"):
                if current:
                    patterns.append(self._to_pattern(current))
                current = {"id": line.split(":", 1)[1].strip()}
                continue
            if not current:
                continue
            for key in ("name", "file", "goal", "cadence", "risk"):
                prefix = f"{key}:"
                if line.startswith(prefix):
                    current[key] = line.split(":", 1)[1].strip()
                    break

        if current:
            patterns.append(self._to_pattern(current))
        return patterns

    def _to_pattern(self, payload: dict[str, str]) -> LoopPattern:
        pattern_id = payload.get("id", "unknown").strip()
        name = payload.get("name", pattern_id.replace("-", " ").title()).strip()
        file_name = payload.get("file", f"{pattern_id}.md").strip()
        goal = payload.get("goal", "").strip()
        cadence = payload.get("cadence", "").strip()
        risk = payload.get("risk", "").strip()
        return LoopPattern(
            pattern_id=pattern_id,
            name=name,
            file_name=file_name,
            goal=goal,
            cadence=cadence,
            risk=risk,
        )

    def _select_pattern(self, instruction: str, patterns: list[LoopPattern]) -> LoopPattern | None:
        if not patterns:
            return None

        lowered = instruction.lower()
        tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", lowered))
        best_pattern: LoopPattern | None = None
        best_score = -1

        for pattern in patterns:
            score = 0
            name_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", pattern.name.lower()))
            id_tokens = set(pattern.pattern_id.lower().split("-"))
            goal_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", pattern.goal.lower()))

            score += len(tokens & name_tokens) * 3
            score += len(tokens & id_tokens) * 3
            score += len(tokens & goal_tokens)
            if pattern.pattern_id in lowered:
                score += 5
            if pattern.name.lower() in lowered:
                score += 5

            if score > best_score:
                best_score = score
                best_pattern = pattern

        return best_pattern if best_score > 0 else None

    def _render_pattern_preview(self, path: Path, max_lines: int = 12) -> str:
        if not path.exists():
            return "Pattern markdown file not found in thirdparty source."

        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        snippet = lines[:max_lines]
        if not snippet:
            return "Pattern markdown file is empty."
        return "\n".join(f"{line}" for line in snippet)


class FileAnalysisExecutor:
    name = "file-analyzer"
    KEYWORDS = {"analyze", "analyse", "inspect", "review", "explain"}

    def can_handle(self, task: RoutedTask, context: ExecutionContext) -> bool:
        lowered = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", context.instruction.lower()))
        return bool(context.file_path) and bool(lowered & self.KEYWORDS)

    def execute(self, task: RoutedTask, context: ExecutionContext) -> ExecutorResult:
        source = context.load_source_text()
        path = Path(context.file_path)
        lines = source.splitlines()
        preview = "\n".join(lines[:5])
        artifact = "\n".join([
            "# File Analysis",
            "",
            f"- path: {path}",
            f"- suffix: {path.suffix or '[none]'}",
            f"- line_count: {len(lines)}",
            f"- character_count: {len(source)}",
            "",
            "## Preview",
            preview,
        ])
        return ExecutorResult(
            executor_name=self.name,
            summary="Analyzed the requested file and produced a structural preview.",
            artifact=artifact,
        )


class ResearchSwarmEntrypoint:
    def __init__(self, memory_db_path: str | Path | None = None, wiki_repo_root: str | Path | None = None) -> None:
        self.classifier = DigitalCognitiveLaborAgent()
        self.memory_store = ResearchSwarmMemoryStore(memory_db_path or DEFAULT_MEMORY_DB_PATH)
        self.wiki = ResearchSwarmWiki(repo_root=Path(wiki_repo_root) if wiki_repo_root else None)
        self.executors: list[TaskExecutor] = [
            ResearchWorkflowExecutor(),
            WikiExecutor(self.wiki),
            LoopEngineeringExecutor(),
            ReportExecutor(),
            SummarizationExecutor(),
            FileAnalysisExecutor(),
        ]

    def route(self, instruction: str) -> RoutedTask:
        if self._is_wiki_command(instruction):
            report = RoutedTask(
                instruction=instruction,
                domain=TaskDomain.TEXT_BASED.value,
                status=ExecutionStatus.DIGITAL_EXECUTION,
                confidence=1.0,
                cognitive_functions=[
                    "perceive",
                    "understand",
                    "plan",
                    "execute",
                    "verify",
                ],
                digital_work=[f"Digital execution fragment: {instruction}"],
                human_handoff=[],
                suggested_commands=[],
                verification_targets=self._build_verification_targets(ExecutionStatus.DIGITAL_EXECUTION),
                next_action="Execute autonomously inside digital systems and return artifacts plus verification.",
                memory_context=self.memory_store.recent_context_lines(),
            )
            self.memory_store.record_route(report.to_dict() | {"summary": report.to_text()})
            return report

        profile = self.classifier.classify_task(instruction)

        if profile.domain is TaskDomain.TEXT_BASED:
            status = ExecutionStatus.DIGITAL_EXECUTION
        elif profile.domain is TaskDomain.HUMAN_ACTION:
            status = ExecutionStatus.HUMAN_HANDOFF_REQUIRED
        elif profile.domain is TaskDomain.HYBRID:
            status = ExecutionStatus.HYBRID_WORKFLOW
        else:
            status = ExecutionStatus.CLARIFICATION_REQUIRED

        digital_work = self._build_digital_work(profile.domain, profile.digital_segments or [instruction])
        human_handoff = self._build_human_handoff(profile.domain, profile.human_segments)
        if profile.open_questions:
            human_handoff.extend(profile.open_questions)
        suggested_commands = self._suggest_commands(instruction) if status is not ExecutionStatus.HUMAN_HANDOFF_REQUIRED else []
        verification_targets = self._build_verification_targets(status)

        report = RoutedTask(
            instruction=instruction,
            domain=profile.domain.value,
            status=status,
            confidence=profile.confidence,
            cognitive_functions=[item.value for item in profile.cognitive_functions],
            digital_work=digital_work,
            human_handoff=human_handoff,
            suggested_commands=suggested_commands,
            verification_targets=verification_targets,
            next_action=profile.recommended_action,
            memory_context=self.memory_store.recent_context_lines(),
        )

        self.memory_store.record_route(report.to_dict() | {"summary": report.to_text()})
        return report

    def execute(
        self,
        instruction: str,
        input_text: str = "",
        file_path: str = "",
        run_prepare: bool = False,
        run_train: bool = False,
    ) -> RoutedTask:
        routed_task = self.route(instruction)
        if self._is_wiki_command(instruction) and routed_task.status not in {ExecutionStatus.DIGITAL_EXECUTION, ExecutionStatus.HYBRID_WORKFLOW}:
            routed_task.status = ExecutionStatus.DIGITAL_EXECUTION
            routed_task.domain = TaskDomain.TEXT_BASED.value
            routed_task.human_handoff = []
            routed_task.digital_work = [f"Digital execution fragment: {instruction}"]
            routed_task.verification_targets = self._build_verification_targets(ExecutionStatus.DIGITAL_EXECUTION)
            routed_task.next_action = "Execute autonomously inside digital systems and return artifacts plus verification."

        if routed_task.status not in {ExecutionStatus.DIGITAL_EXECUTION, ExecutionStatus.HYBRID_WORKFLOW}:
            return routed_task

        context = ExecutionContext(
            instruction=instruction,
            input_text=input_text,
            file_path=file_path,
            run_prepare=run_prepare,
            run_train=run_train,
        )
        for executor in self.executors:
            if executor.can_handle(routed_task, context):
                result = executor.execute(routed_task, context)
                routed_task.executor_name = result.executor_name
                routed_task.execution_summary = result.summary
                routed_task.execution_artifact = result.artifact
                self.memory_store.record_execution(routed_task.to_dict() | {"summary": result.summary})
                return routed_task

        routed_task.execution_summary = "No executor matched; returning routing guidance only."
        self.memory_store.record_execution(routed_task.to_dict() | {"summary": routed_task.execution_summary})
        return routed_task

    def _segment_instruction(self, instruction: str) -> list[str]:
        parts = re.split(r"\b(?:and then|then|after that|afterwards|before|and)\b|[;,]", instruction, flags=re.IGNORECASE)
        segments = [part.strip(" .") for part in parts if part.strip(" .")]
        return segments or [instruction.strip()]

    def _build_digital_work(self, domain: TaskDomain, segments: list[str]) -> list[str]:
        if domain is TaskDomain.HUMAN_ACTION:
            return []
        if domain is TaskDomain.UNKNOWN:
            return ["Clarify the digital scope before attempting execution."]
        return [f"Digital execution fragment: {segment}" for segment in segments]

    def _build_human_handoff(self, domain: TaskDomain, segments: list[str]) -> list[str]:
        if domain is TaskDomain.TEXT_BASED:
            return []
        if domain is TaskDomain.UNKNOWN:
            return ["Ask whether any physical or human-authority step is required."]
        if not segments:
            return ["Identify the real-world step and assign it to a human operator."]
        return [f"Human action required: {segment}" for segment in segments]

    def _suggest_commands(self, instruction: str) -> list[str]:
        lowered_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", instruction.lower()))
        commands: list[str] = []
        for signals, command in RESEARCHSWARM_COMMAND_HINTS:
            if lowered_tokens & signals and command not in commands:
                commands.append(command)
        return commands

    def _build_verification_targets(self, status: ExecutionStatus) -> list[str]:
        if status is ExecutionStatus.DIGITAL_EXECUTION:
            return [
                "Confirm the requested digital artifact was produced.",
                "Check the output against the original task constraints.",
            ]
        if status is ExecutionStatus.HUMAN_HANDOFF_REQUIRED:
            return [
                "Wait for evidence from the human operator.",
                "Do not mark the task complete until the physical step is confirmed.",
            ]
        if status is ExecutionStatus.HYBRID_WORKFLOW:
            return [
                "Verify the digital portion immediately.",
                "Hold final completion until the human-action evidence arrives.",
            ]
        return ["Clarify the execution medium before proceeding."]

    def _is_wiki_command(self, instruction: str) -> bool:
        return bool(re.search(r"\bwiki\s+(ingest|query|lint|init)\b", instruction, flags=re.IGNORECASE))


def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchSwarm task router")
    parser.add_argument("task", help="Task instruction to route")
    parser.add_argument("--input-text", default="", help="Optional source text for text-based executors")
    parser.add_argument("--file", default="", help="Optional file path for text-based executors")
    parser.add_argument("--run-prepare", action="store_true", help="Explicitly run prepare.py when the research workflow executor is selected")
    parser.add_argument("--run-train", action="store_true", help="Explicitly run train.py when the research workflow executor is selected")
    parser.add_argument("--route-only", action="store_true", help="Only classify and route the task without executing a digital executor")
    parser.add_argument("--format", choices=("json", "text"), default="json", help="Output format")
    args = parser.parse_args()

    entrypoint = ResearchSwarmEntrypoint()
    report = entrypoint.route(args.task) if args.route_only else entrypoint.execute(
        args.task,
        input_text=args.input_text,
        file_path=args.file,
        run_prepare=args.run_prepare,
        run_train=args.run_train,
    )
    if args.format == "text":
        print(report.to_text())
        return
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()