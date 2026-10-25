from __future__ import annotations
import os
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from core.config import load_yaml_config
from env.clinical_env import ClinicalEnv
from env.task_stream import TaskStream

ROOT = Path(__file__).resolve().parents[1]


class CreateSession(BaseModel):
    case_id: str


class ActionRequest(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)


def create_app(cfg: dict, repo_root: Path = ROOT) -> FastAPI:
    app = FastAPI(title="ClinVerse Agent API", version="0.3.0")
    refs = {ref.case_id: ref for ref in TaskStream(repo_root / cfg["paths"]["cases"])}
    sessions = {}
    datasets = {name: {**entry, "data_dir": str((repo_root / entry["data_dir"]).resolve())}
                for name, entry in cfg["datasets"].items()}

    @app.get("/cases")
    def cases():
        return [{"case_id": case_id} for case_id in refs]

    @app.post("/sessions")
    def create_session(request: CreateSession):
        if request.case_id not in refs:
            raise HTTPException(404, "Case not found")
        env = ClinicalEnv.from_case_file(
            refs[request.case_id].path, cfg["llm_providers"][cfg["default_patient_provider"]],
            lab_openai_config=cfg["llm_providers"][cfg["default_lab_provider"]], datasets_config=datasets,
            language=cfg["simulation"].get("language", "en"), max_turns=cfg["simulation"]["max_turns"],
        )
        initial = env.reset()
        session_id = str(uuid4())
        sessions[session_id] = (env, {t["function"]["name"] for t in initial.tools_schema})
        return {"session_id": session_id, "case_id": initial.case_id,
                "visible_info": initial.visible_info, "tools_schema": initial.tools_schema}

    def get_session(session_id):
        if session_id not in sessions:
            raise HTTPException(404, "Session not found")
        return sessions[session_id]

    @app.post("/sessions/{session_id}/step")
    def step(session_id: str, request: ActionRequest):
        env, allowed = get_session(session_id)
        if env.done:
            raise HTTPException(409, "Episode has finished")
        if request.tool not in allowed:
            raise HTTPException(400, "Select a tool from the session schema")
        result = env.step(request.tool, request.args)
        return {"observation": result.observation, "done": result.done}

    @app.get("/sessions/{session_id}/score")
    def score(session_id: str):
        env, _ = get_session(session_id)
        if not env.done:
            raise HTTPException(409, "Complete the episode before requesting scores")
        return env.score()

    @app.delete("/sessions/{session_id}")
    def close_session(session_id: str):
        get_session(session_id)
        del sessions[session_id]
        return {"ok": True}

    return app


app = create_app(load_yaml_config(os.environ.get("CLINVERSE_CONFIG", ROOT / "config/default.yaml")))
