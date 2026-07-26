from abc import ABC, abstractmethod


class BaseProvider(ABC):
    @abstractmethod
    def respond(self, action_args: dict, case: dict) -> tuple[str, dict]:
        """Return (natural_language_response, hidden_metadata)."""
