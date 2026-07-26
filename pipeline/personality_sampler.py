"""Sample Big Five (OCEAN) personality traits for patient simulation.

Each trait is scored 1-10 from an independent normal distribution (μ=5.5, σ=1.8)
clipped to [1,10], which approximates population survey distributions.

Scores are stored in the case JSON alongside behavioral descriptions that are
injected directly into the PatientActor system prompt.
"""

from __future__ import annotations

import random


def _clamp(x: float, lo: float = 1.0, hi: float = 10.0) -> int:
    return int(max(lo, min(hi, round(x))))


def sample_personality(rng: random.Random | None = None) -> dict:
    """Sample OCEAN scores and build behavioral descriptions for one patient."""
    if rng is None:
        rng = random.Random()

    scores = {
        "O": _clamp(rng.gauss(5.5, 1.8)),  # Openness
        "C": _clamp(rng.gauss(5.5, 1.8)),  # Conscientiousness
        "E": _clamp(rng.gauss(5.5, 1.8)),  # Extraversion
        "A": _clamp(rng.gauss(5.5, 1.8)),  # Agreeableness
        "N": _clamp(rng.gauss(5.5, 1.8)),  # Neuroticism
    }
    return {
        "scores": scores,
        "behavioral_descriptions": _build_descriptions(scores),
    }


def _level(score: int) -> str:
    if score >= 8:
        return "high"
    if score <= 3:
        return "low"
    return "mid"


# Behavioral descriptions per trait × level, written as patient instructions.
# Focus on clinical-encounter behaviors: how much the patient volunteers,
# how they remember history, how they respond emotionally, etc.
_EN: dict[str, dict[str, str]] = {
    "O": {
        "high": "You are curious about your condition and ask follow-up questions about possible causes and what tests might reveal.",
        "mid": "You answer the doctor's questions without much elaboration or curiosity.",
        "low": "You focus only on your immediate discomfort and do not ask about diagnoses or explanations — you just want to feel better.",
    },
    "C": {
        "high": "You are well-organized about your health: you know exact medication names and doses, and can describe when your symptoms started with precision.",
        "mid": "You recall most of your medical history but may be vague on exact dates or dosages.",
        "low": "You are vague about your medical history — you often cannot recall exact medication names, when symptoms started, or past diagnoses.",
    },
    "E": {
        "high": "You are talkative and proactively volunteer information — you elaborate on your symptoms without being prompted.",
        "mid": "You answer questions directly but do not elaborate unless the doctor asks a specific follow-up.",
        "low": "You give brief, minimal answers and require the doctor to ask specific questions to draw out information.",
    },
    "A": {
        "high": "You are cooperative and trusting — you agree with the doctor's suggestions and are eager to follow their guidance.",
        "mid": "You are generally cooperative but occasionally ask for clarification before agreeing.",
        "low": "You are somewhat skeptical and may push back gently on questions or proposed directions.",
    },
    "N": {
        "high": "You are visibly anxious and emotionally distressed — you emphasize how bad you feel and worry aloud about serious illness.",
        "mid": "You express some concern about your condition but remain relatively composed.",
        "low": "You are calm and tend to downplay your symptoms — you minimize discomfort even when asked about severity.",
    },
}

_ZH: dict[str, dict[str, str]] = {
    "O": {
        "high": "你对自己的病情很好奇，会主动问医生症状的可能原因以及各项检查的意义。",
        "mid": "你正常回答医生的问题，不会主动延伸或追问。",
        "low": "你只关注当下的不适，不想了解病因或诊断，只希望尽快好起来。",
    },
    "C": {
        "high": "你对健康管理很细心，能准确说出药物名称、剂量，以及症状开始的具体时间。",
        "mid": "你大致记得自己的病史，但对具体日期或剂量可能有些模糊。",
        "low": "你对自己的病史比较模糊——经常记不清药名、症状什么时候开始，或者以前得过什么病。",
    },
    "E": {
        "high": "你话比较多，会主动提供很多信息，不等医生追问就开始描述症状细节。",
        "mid": "你直接回答医生的问题，但不会主动多说。",
        "low": "你的回答很简短，需要医生主动追问才能获取更多信息。",
    },
    "A": {
        "high": "你很配合，对医生的建议表示认同，愿意按医生说的做。",
        "mid": "你总体配合，但有时会对建议提出问题或要求解释。",
        "low": "你有些怀疑，对医生的某些问题或建议会提出质疑。",
    },
    "N": {
        "high": "你明显焦虑，情绪有些紧张，会强调自己有多不舒服，担心是严重疾病。",
        "mid": "你对自己的状况有些担心，但总体还比较冷静。",
        "low": "你比较淡定，倾向于淡化症状，就算医生追问严重程度，你也觉得还好。",
    },
}


def _build_descriptions(scores: dict[str, int]) -> dict[str, str]:
    result: dict[str, str] = {}
    for trait, score in scores.items():
        lv = _level(score)
        result[f"{trait}_en"] = _EN[trait][lv]
        result[f"{trait}_zh"] = _ZH[trait][lv]
    return result


def format_for_prompt(personality: dict, language: str = "en") -> str:
    """Return a formatted string ready for insertion into a system prompt."""
    scores = personality["scores"]
    descs = personality["behavioral_descriptions"]
    suffix = "_zh" if language == "zh" else "_en"

    score_line = (
        f"O:{scores['O']} C:{scores['C']} E:{scores['E']} "
        f"A:{scores['A']} N:{scores['N']} （满分10）"
        if language == "zh"
        else f"O:{scores['O']} C:{scores['C']} E:{scores['E']} "
             f"A:{scores['A']} N:{scores['N']} (out of 10)"
    )
    lines = [score_line] + [descs[f"{t}{suffix}"] for t in ("O", "C", "E", "A", "N")]
    return "\n".join(f"- {ln}" for ln in lines)
