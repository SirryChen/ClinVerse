from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class EpisodeContext:
    case_id: str
    visible_info: dict
    tools_schema: list[dict]
    max_turns: int
    memory_context: str = ""


@dataclass(frozen=True)
class Action:
    name: str
    arguments: dict = field(default_factory=dict)


class DoctorAgent(Protocol):
    def reset(self, context: EpisodeContext) -> None: ...
    def act(self) -> Action: ...
    def observe(self, observation: str) -> None: ...


class AgentTrainer(Protocol):
    def context(self, context: EpisodeContext) -> str: ...
    def update(self, record: dict) -> None: ...
    def save(self, path: str | Path) -> None: ...
