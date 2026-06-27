"""ResearchSwarm cognitive control layer.

This module leaves the existing training stack untouched and adds a separate
decision surface for Digital Cognitive Labor. The key capability is routing a
task into either:

1. text-based work the agent can execute directly in software, or
2. human-action work that requires a person, a physical actuator, or explicit
   real-world intervention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import re
from typing import Iterable


class TaskDomain(str, Enum):
    TEXT_BASED = "text-based"
    HUMAN_ACTION = "human-action"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class CognitiveFunction(str, Enum):
    PERCEIVE = "perceive"
    UNDERSTAND = "understand"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    ESCALATE = "escalate"


TEXT_SIGNALS = {
    "analyze",
    "baseline",
    "classify",
    "code",
    "compare",
    "compute",
    "debug",
    "design",
    "document",
    "draft",
    "evaluate",
    "explain",
    "extract",
    "generate",
    "implement",
    "optimize",
    "plan",
    "prepare",
    "run",
    "review",
    "refactor",
    "research",
    "route",
    "summarize",
    "train",
    "training",
    "translate",
    "write",
}

HUMAN_ACTION_SIGNALS = {
    "assemble",
    "attend",
    "bring",
    "call",
    "carry",
    "clean",
    "click",
    "deliver",
    "drive",
    "email",
    "fax",
    "file",
    "go",
    "install",
    "interview",
    "lift",
    "meet",
    "move",
    "negotiate",
    "operate",
    "phone",
    "photograph",
    "pick",
    "press",
    "purchase",
    "repair",
    "scan",
    "ship",
    "sign",
    "speak",
    "travel",
    "visit",
    "walk",
}

HUMAN_ONLY_PATTERNS = (
    r"\bin person\b",
    r"\bphysically\b",
    r"\bon[- ]site\b",
    r"\breal world\b",
    r"\bwet signature\b",
    r"\bmanual approval\b",
)


@dataclass
class TaskProfile:
    instruction: str
    domain: TaskDomain
    confidence: float
    text_signals: list[str] = field(default_factory=list)
    human_action_signals: list[str] = field(default_factory=list)
    cognitive_functions: list[CognitiveFunction] = field(default_factory=list)
    recommended_action: str = ""
    execution_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["domain"] = self.domain.value
        payload["cognitive_functions"] = [item.value for item in self.cognitive_functions]
        return payload


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.lower())


def _ordered_unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


class DigitalCognitiveLaborAgent:
    """Routes tasks based on whether software can complete them autonomously."""

    def classify_task(self, instruction: str) -> TaskProfile:
        tokens = _tokenize(instruction)
        text_hits = [token for token in tokens if token in TEXT_SIGNALS]
        human_hits = [token for token in tokens if token in HUMAN_ACTION_SIGNALS]
        human_pattern_hits = [pattern for pattern in HUMAN_ONLY_PATTERNS if re.search(pattern, instruction, re.IGNORECASE)]

        text_score = len(text_hits)
        human_score = len(human_hits) + len(human_pattern_hits) * 2

        if text_score and human_score:
            domain = TaskDomain.HYBRID
        elif human_score:
            domain = TaskDomain.HUMAN_ACTION
        elif text_score:
            domain = TaskDomain.TEXT_BASED
        else:
            domain = TaskDomain.UNKNOWN

        total_score = text_score + human_score
        confidence = 0.5 if total_score == 0 else round(max(text_score, human_score) / total_score, 2)

        cognitive_functions = self._infer_cognitive_functions(domain)
        execution_steps = self._build_execution_steps(domain, instruction)
        recommended_action = self._recommend_action(domain)

        return TaskProfile(
            instruction=instruction,
            domain=domain,
            confidence=confidence,
            text_signals=_ordered_unique(text_hits),
            human_action_signals=_ordered_unique(human_hits + human_pattern_hits),
            cognitive_functions=cognitive_functions,
            recommended_action=recommended_action,
            execution_steps=execution_steps,
        )

    def _infer_cognitive_functions(self, domain: TaskDomain) -> list[CognitiveFunction]:
        if domain is TaskDomain.TEXT_BASED:
            return [
                CognitiveFunction.PERCEIVE,
                CognitiveFunction.UNDERSTAND,
                CognitiveFunction.PLAN,
                CognitiveFunction.EXECUTE,
                CognitiveFunction.VERIFY,
            ]
        if domain is TaskDomain.HUMAN_ACTION:
            return [
                CognitiveFunction.PERCEIVE,
                CognitiveFunction.UNDERSTAND,
                CognitiveFunction.PLAN,
                CognitiveFunction.ESCALATE,
                CognitiveFunction.VERIFY,
            ]
        if domain is TaskDomain.HYBRID:
            return [
                CognitiveFunction.PERCEIVE,
                CognitiveFunction.UNDERSTAND,
                CognitiveFunction.PLAN,
                CognitiveFunction.EXECUTE,
                CognitiveFunction.ESCALATE,
                CognitiveFunction.VERIFY,
            ]
        return [
            CognitiveFunction.PERCEIVE,
            CognitiveFunction.UNDERSTAND,
            CognitiveFunction.PLAN,
        ]

    def _recommend_action(self, domain: TaskDomain) -> str:
        if domain is TaskDomain.TEXT_BASED:
            return "Execute autonomously inside digital systems and return artifacts plus verification."
        if domain is TaskDomain.HUMAN_ACTION:
            return "Do not pretend to complete the task; produce a human handoff checklist and wait for real-world completion."
        if domain is TaskDomain.HYBRID:
            return "Complete the digital portion now, isolate the physical/manual portion, and create an explicit handoff boundary."
        return "Clarify the task before execution because the domain is not yet distinguishable."

    def _build_execution_steps(self, domain: TaskDomain, instruction: str) -> list[str]:
        if domain is TaskDomain.TEXT_BASED:
            return [
                "Parse the request into inputs, constraints, and outputs.",
                "Execute the required digital work product.",
                "Verify the result against the stated objective.",
                "Return the artifact, evidence, and any residual risks.",
            ]
        if domain is TaskDomain.HUMAN_ACTION:
            return [
                "Identify the exact real-world action that cannot be completed by software.",
                "Generate a checklist, prerequisites, and safety constraints for the human operator.",
                "Mark the task as pending human execution rather than falsely reporting completion.",
                "Request confirmation or evidence after the human step is done.",
            ]
        if domain is TaskDomain.HYBRID:
            return [
                "Separate the task into digital sub-work and human-action sub-work.",
                "Execute the digital sub-work immediately.",
                "Create a minimal handoff package for the human-action sub-work.",
                "Resume verification once the human-action evidence is supplied.",
            ]
        return [
            "Restate the task.",
            "Ask a clarifying question focused on the execution medium.",
            f"Clarify whether this request is meant to stay inside software systems: {instruction}",
        ]

    def process(self, instruction: str) -> dict:
        profile = self.classify_task(instruction)
        return profile.to_dict()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Classify Digital Cognitive Labor tasks for ResearchSwarm.")
    parser.add_argument("instruction", help="Natural-language task to classify")
    args = parser.parse_args()

    agent = DigitalCognitiveLaborAgent()
    report = agent.process(args.instruction)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()