from abc import ABC, abstractmethod


class BaseDoctorAgent(ABC):
    @abstractmethod
    def step(self, messages: list[dict]) -> tuple[str, dict]:
        """Given conversation messages, return (function_name, function_args)."""

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state for a new case."""
