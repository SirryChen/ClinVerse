"""HTTP layer over ClinicalEnv/TaskStream, for agent frameworks that live in a
different process or language and can't import this repo directly. Wraps the
exact same reset/step/score contract as the in-process SDK — nothing here
re-implements clinical logic, it only does session bookkeeping and JSON
(de)serialization.

Session state is a naive in-memory dict: fine for a single-process research
eval harness, not meant to survive a restart or run behind multiple workers.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.config import load_yaml_config
from env.clinical_env import ClinicalEnv
from env.task_stream import TaskStream

REPO_ROOT = Path(__file__).parent.parent


def _datasets_config(cfg: dict, repo_root: Path) -> dict[str, dict]:
    return {
        name: {"data_dir": str((repo_root / entry["data_dir"]).resolve())}
        for name, entry in cfg.get("datasets", {}).items()
        if entry.get("data_dir")
    }


class CaseSummary(BaseModel):
    case_id: str
    category: Optional[str]


class CreateSessionRequest(BaseModel):
    case_id: str


class CreateSessionResponse(BaseModel):
    session_id: str
    case_id: str
    visible_info: dict
    tools_schema: list


class StepRequest(BaseModel):
    tool: str
    args: dict = {}


class StepResponse(BaseModel):
    observation: str
    done: bool
    info: dict


def create_app(cfg: dict, repo_root: Path = REPO_ROOT) -> FastAPI:
    datasets_config = _datasets_config(cfg, repo_root)
    cases_dir = repo_root / cfg["paths"]["cases"]
    all_cases = TaskStream(cases_dir)
    train_cases, test_cases = all_cases.split()
    case_index = {ref.case_id: ref for ref in all_cases}
    sessions: dict[str, ClinicalEnv] = {}

    llm_config = cfg["llm_providers"][cfg["default_llm_provider"]]
    max_turns = cfg["simulation"]["max_turns"]
    language = cfg["simulation"].get("language", "en")

    app = FastAPI(title="ClinVerse Clinical Env API")

    @app.get("/cases", response_model=list[CaseSummary])
    def list_cases(split: str = "all"):
        streams = {"train": train_cases, "test": test_cases, "all": all_cases}
        if split not in streams:
            raise HTTPException(400, f"Unknown split '{split}', expected train|test|all")
        return [CaseSummary(case_id=r.case_id, category=r.category) for r in streams[split]]

    @app.post("/sessions", response_model=CreateSessionResponse)
    def create_session(req: CreateSessionRequest):
        ref = case_index.get(req.case_id)
        if ref is None:
            raise HTTPException(404, f"Unknown case_id '{req.case_id}'")
        env = ClinicalEnv.from_case_file(
            case_path=ref.path,
            patient_openai_config=llm_config,
            datasets_config=datasets_config,
            max_turns=max_turns,
            language=language,
        )
        obs = env.reset()
        session_id = uuid.uuid4().hex
        sessions[session_id] = env
        return CreateSessionResponse(
            session_id=session_id,
            case_id=obs.case_id,
            visible_info=obs.visible_info,
            tools_schema=obs.tools_schema,
        )

    def _get_session(session_id: str) -> ClinicalEnv:
        env = sessions.get(session_id)
        if env is None:
            raise HTTPException(404, f"Unknown session_id '{session_id}'")
        return env

    @app.post("/sessions/{session_id}/step", response_model=StepResponse)
    def step_session(session_id: str, req: StepRequest):
        env = _get_session(session_id)
        try:
            result = env.step(req.tool, req.args)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return StepResponse(observation=result.observation, done=result.done, info=result.info)

    @app.get("/sessions/{session_id}/score")
    def score_session(session_id: str):
        env = _get_session(session_id)
        if not env.done:
            raise HTTPException(409, "Episode not finished yet.")
        return env.score()

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str):
        sessions.pop(session_id, None)
        return {"ok": True}

    return app


def _load_default_config() -> dict:
    return load_yaml_config(REPO_ROOT / "config" / "default.yaml")


app = create_app(_load_default_config())
