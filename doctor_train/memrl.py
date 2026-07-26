from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


MEMORY_FORMAT = "hospital-memrl-v1"


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def reward_from_scores(scores: dict) -> float:
    """Map benchmark scores to one bounded reward.

    ClinicalEnv uses the named clinical metrics below, while AIHospital-like
    environments often expose a single ``reward`` or ``success`` field. The
    fallback keeps the memory layer benchmark-agnostic.
    """
    if "reward" in scores:
        return round(max(-1.0, min(1.0, float(scores["reward"]))), 4)
    if "success" in scores and not any(k.endswith("_score") for k in scores):
        success = scores["success"]
        if isinstance(success, str):
            success = success.strip().lower() in {"1", "true", "yes", "success"}
        return 1.0 if bool(success) else 0.0

    components = [
        ("diagnosis_score", 0.5),
        # The unified information-collection score replaces the former
        # duplicate history/information components while preserving their
        # combined reward weight.
        ("information_collection_score", 0.3),
        ("abnormal_test_coverage", 0.1),
        ("test_order_coverage", 0.1),
    ]
    present = [(float(scores[name]), weight) for name, weight in components if name in scores]
    if not present:
        generic = [float(value) for key, value in scores.items() if key.endswith("_score")]
        return round(sum(generic) / len(generic), 4) if generic else 0.0
    total_weight = sum(weight for _, weight in present)
    return round(sum(value * weight for value, weight in present) / total_weight, 4)


def _tokenize(text: str) -> set[str]:
    """Create stable lexical features for both English and Chinese cases."""
    text = str(text or "").lower()
    tokens = {tok for tok in re.findall(r"[\w.]+", text) if len(tok) > 1}
    # ``\w`` treats a complete Chinese phrase as one token. Character bigrams
    # preserve useful overlap without requiring an embedding API at retrieval.
    cjk = "".join(re.findall(r"[\u3400-\u9fff]", text))
    tokens.update(cjk[idx : idx + 2] for idx in range(max(0, len(cjk) - 1)))
    return tokens


def _jaccard(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _initial_visible_from_record(record: dict) -> dict:
    return record.get("transcript", {}).get("initial_visible_info", {}) or {}


def intent_from_visible(visible: dict, *, source: str = "") -> str:
    def _clip(value: Any, limit: int = 240) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + " ..."

    parts = [
        f"age={visible.get('age', '')}",
        f"gender={visible.get('gender', '')}",
        f"admission={visible.get('admission_type', '')}",
        f"vitals={_clip(visible.get('vital_signs', ''), 240)}",
        f"chief_complaint={_clip(visible.get('chief_complaint', ''), 240)}",
    ]
    return " | ".join(part for part in parts if part and not part.endswith("="))


def intent_from_case(case: dict) -> str:
    view = case.get("simulation_views", {}).get("patient_actor_view", {})
    init = view.get("initial_presentation", {})
    source = case.get("core_admission", {}).get("source", "")
    visible = {
        "age": init.get("age"),
        "gender": init.get("gender"),
        "admission_type": init.get("admission_type"),
        "vital_signs": init.get("vital_signs"),
        "chief_complaint": _chief_complaint(view.get("symptom_context", "")),
    }
    return intent_from_visible(visible, source=source)


def _chief_complaint(hpi: str) -> str:
    if not hpi:
        return ""
    for sep in (".", "\n"):
        idx = hpi.find(sep)
        if 10 < idx < 250:
            return hpi[: idx + 1].strip()
    return hpi[:200].strip()


def intent_from_record(record: dict) -> str:
    return intent_from_visible(_initial_visible_from_record(record), source=record.get("source", ""))


def _compact_turn(turn: dict, max_response_chars: int = 220) -> str:
    fn_name = turn.get("function", "")
    args = {k: v for k, v in (turn.get("args", {}) or {}).items() if not str(k).startswith("_")}
    response = (turn.get("response") or "").strip()
    if len(response) > max_response_chars:
        response = response[:max_response_chars] + " ..."
    return f"{fn_name} args={json.dumps(args, ensure_ascii=False)} -> {response}"


def experience_from_record(record: dict, max_turns: int = 8) -> str:
    scores = record.get("scores", {})
    reward = reward_from_scores(scores)
    outcome = "successful" if reward >= 0.75 else "partially successful" if reward > 0 else "unsuccessful"
    lines = [
        f"Outcome: {outcome}; reward={reward:.3f}",
        "Transferable trajectory (do not copy patient-specific facts):",
        "Trajectory:",
    ]
    turns = record.get("transcript", {}).get("turns", [])[:max_turns]
    lines.extend(f"- {_compact_turn(turn)}" for turn in turns)
    tags = ", ".join(record.get("tags", []))
    if tags:
        lines.append(f"Tags: {tags}")
    return "\n".join(lines)


def memory_from_record(record: dict) -> dict:
    reward = reward_from_scores(record.get("scores", {}))
    success = reward >= 0.75 or float(record.get("scores", {}).get("diagnosis_score", 0.0)) >= 1.0
    memory_id = f"mem_{record.get('case_id', 'unknown')}"
    return {
        "memory_id": memory_id,
        "intent": intent_from_record(record),
        "experience": experience_from_record(record),
        "utility": {
            "q_value": reward,
            "reward_ma": reward,
            "visits": 0,
            "last_reward": reward,
        },
        "metadata": {
            "case_id": record.get("case_id", ""),
            "source": record.get("source", ""),
            "task_type": record.get("task_type", "clinical_diagnosis"),
            "success": success,
            "tags": record.get("tags", []),
            "created_at": datetime.now().isoformat(),
            "ground_truth_icd10": record.get("ground_truth", {}).get("icd10_code", ""),
            "predicted_icd10": record.get("prediction", {}).get("icd10_code", ""),
        },
    }


def build_memory_bank(records: list[dict]) -> dict:
    seen: set[str] = set()
    items = []
    for record in records:
        item = memory_from_record(record)
        if item["memory_id"] in seen:
            continue
        seen.add(item["memory_id"])
        items.append(item)
    return {"format": MEMORY_FORMAT, "items": items}


def load_memory_bank(path: str | Path | None) -> dict:
    if not path:
        return {"format": MEMORY_FORMAT, "items": []}
    with open(path) as f:
        bank = json.load(f)
    if isinstance(bank, list):
        return {"format": MEMORY_FORMAT, "items": bank}
    if bank.get("format") != MEMORY_FORMAT:
        bank.setdefault("format", MEMORY_FORMAT)
    bank.setdefault("items", [])
    return bank


def save_memory_bank(bank: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")


def retrieve_memories(
    bank: dict,
    intent: str,
    *,
    top_k: int = 3,
    min_similarity: float = 0.0,
    weight_similarity: float = 0.5,
    weight_q: float = 0.5,
    q_min_threshold: float | None = None,
) -> list[dict]:
    ranked = []
    total_weight = weight_similarity + weight_q
    if total_weight <= 0:
        raise ValueError("At least one MemRL retrieval weight must be positive.")
    sim_weight = weight_similarity / total_weight
    q_weight = weight_q / total_weight
    for item in bank.get("items", []):
        sim = _jaccard(intent, item.get("intent", ""))
        q_value = float(item.get("utility", {}).get("q_value", 0.0))
        score = (sim_weight * sim) + (q_weight * q_value)
        if sim >= min_similarity and (q_min_threshold is None or q_value >= q_min_threshold):
            enriched = dict(item)
            enriched["retrieval"] = {
                "similarity": round(sim, 4),
                "q_value": round(q_value, 4),
                "score": round(score, 4),
                "weight_similarity": round(sim_weight, 4),
                "weight_q": round(q_weight, 4),
            }
            ranked.append(enriched)
    ranked.sort(key=lambda item: item["retrieval"]["score"], reverse=True)
    return ranked[: max(0, top_k)]


def format_memory_context(memories: list[dict], *, language: str = "en") -> str:
    if not memories:
        return ""
    if language == "zh":
        header = (
            "== 可参考的既往诊疗记忆 ==\n"
            "以下是相似病例中的经验。它们可能有帮助，也可能不完全适用。"
            "请只提取可迁移的诊疗策略，不要把其中的诊断或检查结果当成本病例事实。\n"
        )
    else:
        header = (
            "== Retrieved Episodic Memories ==\n"
            "These are experiences from similar prior cases. They may be useful but are not facts about the current patient. "
            "Use only transferable diagnostic strategy, not copied diagnoses or results.\n"
        )
    blocks = [header]
    for idx, item in enumerate(memories, start=1):
        meta = item.get("metadata", {})
        utility = item.get("utility", {})
        retrieval = item.get("retrieval", {})
        blocks.append(
            "\n".join(
                [
                    f"[Memory {idx}] id={item.get('memory_id', '')} success={meta.get('success')} q={float(utility.get('q_value', 0.0)):.3f} sim={float(retrieval.get('similarity', 0.0)):.3f}",
                    f"Intent: {item.get('intent', '')}",
                    item.get("experience", ""),
                ]
            )
        )
    return "\n\n".join(blocks)


def update_memory_values(
    bank: dict,
    selected_memory_ids: list[str],
    reward: float,
    *,
    alpha: float = 0.1,
) -> list[dict[str, Any]]:
    selected = set(selected_memory_ids)
    updates: list[dict[str, Any]] = []
    for item in bank.get("items", []):
        if item.get("memory_id") not in selected:
            continue
        utility = item.setdefault("utility", {})
        old_q = float(utility.get("q_value", 0.0))
        new_q = ((1.0 - alpha) * old_q) + (alpha * reward)
        old_ma = float(utility.get("reward_ma", old_q))
        reward_ma = ((1.0 - alpha) * old_ma) + (alpha * reward)
        utility["q_value"] = round(new_q, 6)
        utility["reward_ma"] = round(reward_ma, 6)
        utility["last_reward"] = reward
        utility["visits"] = int(utility.get("visits", 0)) + 1
        utility["updated_at"] = datetime.now().isoformat()
        updates.append({"memory_id": item.get("memory_id"), "old_q": old_q, "new_q": utility["q_value"]})
    return updates


def add_or_replace_memory(bank: dict, record: dict) -> dict:
    item = memory_from_record(record)
    items = bank.setdefault("items", [])
    for idx, existing in enumerate(items):
        if existing.get("memory_id") == item["memory_id"]:
            items[idx] = item
            return item
    items.append(item)
    return item
