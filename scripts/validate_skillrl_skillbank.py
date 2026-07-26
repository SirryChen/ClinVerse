#!/usr/bin/env python3
"""Validate and exercise SkillRL's skills-only memory without GPU training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and query a SkillRL skill bank.")
    parser.add_argument("--skills", required=True, help="SkillRL-compatible JSON skill bank.")
    parser.add_argument("--task", required=True, help="Current task description.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--task-specific-top-k", type=int, default=3)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    skillrl_root = repo_root / "related-work" / "SkillRL"
    sys.path.insert(0, str(skillrl_root))
    from agent_system.memory.skills_only_memory import SkillsOnlyMemory

    memory = SkillsOnlyMemory(
        skills_json_path=args.skills,
        retrieval_mode="template",
        task_specific_top_k=args.task_specific_top_k,
    )
    retrieved = memory.retrieve(args.task, top_k=args.top_k)
    print(json.dumps({
        "skill_count": memory.get_skill_count(),
        "task_type": retrieved.get("task_type"),
        "general_skills": len(retrieved.get("general_skills", [])),
        "task_specific_skills": len(retrieved.get("task_specific_skills", [])),
        "prompt": memory.format_for_prompt(retrieved),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
