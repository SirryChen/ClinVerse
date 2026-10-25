from __future__ import annotations

from copy import deepcopy
from agents.interfaces import Action, DoctorAgent, EpisodeContext


def run_episode(env, agent: DoctorAgent, *, memory_context: str = "", context_provider=None) -> dict:
    initial = env.reset()
    context = EpisodeContext(initial.case_id, deepcopy(initial.visible_info),
                             deepcopy(initial.tools_schema), env.max_turns, memory_context)
    if context_provider:
        context = EpisodeContext(context.case_id, context.visible_info, context.tools_schema,
                                 context.max_turns, context_provider(deepcopy(context)))
    allowed = {tool["function"]["name"] for tool in context.tools_schema}
    agent.reset(context)
    for _ in range(env.max_turns):
        action = agent.act()
        if not isinstance(action, Action) or action.name not in allowed:
            raise ValueError("Agent must return an Action from the environment tool schema")
        if not isinstance(action.arguments, dict):
            raise ValueError("Action arguments must be an object")
        step = env.step(action.name, action.arguments)
        if step.done:
            break
        agent.observe(step.observation)
    if not env.done:
        raise RuntimeError("Episode did not terminate within the configured turn budget")
    return {"case_id": initial.case_id, "transcript": env.transcript.to_dict(), "scores": env.score()}
