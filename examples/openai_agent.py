from __future__ import annotations

import json
from openai import OpenAI
from agents.gpt_agent import GPTDoctorAgent
from agents.interfaces import Action, EpisodeContext


class OpenAIAgent:
    def __init__(self, provider: dict, language: str = "en"):
        self.provider, self.language = provider, language

    def reset(self, context: EpisodeContext) -> None:
        self.context, self.turn, self.call_id = context, 0, None
        cfg = self.provider
        self.agent = GPTDoctorAgent(
            OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=cfg.get("timeout", 180)),
            cfg["doctor_model"], language=self.language, tools_schema=context.tools_schema,
            extra_body=cfg.get("extra_body"), memory_context=context.memory_context or None,
            message_style=cfg.get("doctor_message_style", "openai"),
        )

    def act(self) -> Action:
        messages = [{"role": "user", "content": json.dumps(self.context.visible_info)}] if self.turn == 0 else []
        name, arguments, self.call_id = self.agent.step(
            messages, force_finalize=self.turn >= self.context.max_turns - 1,
        )
        self.turn += 1
        return Action(name, arguments)

    def observe(self, observation: str) -> None:
        self.agent.add_tool_result(self.call_id, observation)
