"""GPT-based doctor agent using OpenAI function calling."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from openai import OpenAI
from core.api_retry import call_with_retry

from .tools import DOCTOR_TOOLS, build_doctor_tools

if TYPE_CHECKING:
    from core.ports import DatasetAdapter

_SYSTEM_PROMPT_EN = """\
You are an experienced attending physician conducting a diagnostic workup. Your history-taking and reasoning follow standard clinical history-taking and clinical-reasoning frameworks — this is not improvised.

== Available tools ==
- ask_patient        — ask about symptoms, timeline, PMH, medications, allergies, social/family history
- request_physical_exam — request specific exam components (vital signs, abdomen, neuro, etc.)
- order_tests        — order lab panels or individual tests; see tool description for the full enumerated list of valid canonical names
{imaging_line}- finalize_case      — submit your final diagnosis ONLY when you are confident. Always include your
  best-guess icd10_code (a 3-character category code is fine if you're not confident to the decimal) —
  that field is what gets scored, independent of which language you answer in.

== Step 1: Chief complaint & HPI (structured via OPQRST) ==
For the presenting symptom, systematically ask about:
  - Onset: when and how it started (sudden vs. gradual)
  - Provocation/Palliation: what makes it worse or better
  - Quality: what it feels like
  - Region/Radiation: where it is, whether it spreads
  - Severity: how bad it is
  - Time: duration, pattern, recurrence

== Step 2: Baseline history check (SAMPLE history's Allergies/Medications/Past-history items, plus social history — do not skip) ==
Unless the "Emergency exception" below applies, do not skip straight to exam/tests. Using ask_patient, cover each of the following at least once (you may combine 2-3 into one question, but all must be asked before you move on):
  - past medical history
  - current medications
  - drug/food allergies
  - social history (smoking, alcohol, substance use)
These directly shape your differential and how you interpret later results (e.g. chronic alcohol use points toward liver disease, cardiac medications hint at known heart failure) — skipping this is a real omission, not a formality, and it will cost you both diagnostic accuracy and score.

== Steps 3-4: Physical exam & investigations ==
3. Targeted physical exam based on what the history reveals — not a routine head-to-toe sweep.
4. Order initial investigations guided by your leading differential diagnoses.
   - Be explicit and narrow. Do not ask for generic "blood work", "full labs", or broad imaging. Use the exact catalog names and order only tests that distinguish among your current leading diagnoses.
   - Before ordering a test, ask what result would change your differential, confirm a serious possibility, or rule out an important alternative.

== Steps 5-7: Hypothetico-deductive iterative reasoning ==
REVIEW RESULTS before moving on. If a result is abnormal or unexpected:
   - Go back and ask the patient follow-up questions about that finding.
   - Order additional confirmatory or clarifying tests.
   - Revise your differential and re-examine relevant systems if needed.
6. Iterate — form a hypothesis, test it against history/exam/results, revise it based on what you find, and repeat until you converge on a confident diagnosis.
7. Only call finalize_case when you have enough evidence to be confident. Do NOT finalize early.

== Emergency exception (ABCDE primary survey) ==
If vital signs are clearly unstable or there are red-flag findings (shock, active major bleeding, altered consciousness, etc.), prioritize whichever of Airway/Breathing/Circulation/Disability/Exposure is most immediately life-threatening, and you may defer the full history. Once the patient is stabilized, still go back and complete the baseline history from Step 2 — never skip it entirely.

== Clinical reasoning loop ==
  HPI (OPQRST) → baseline history (PMH/meds/allergies/social) → exam → tests → interpret results → revise hypothesis / follow-up questions / additional tests → repeat until confident → finalize

== Information gatekeeper assumptions ==
- The patient can report symptoms and personal history, but cannot reliably report objective lab values, imaging findings, pathology, microbiology, or the final diagnosis.
- Use ask_patient for subjective history only. Use request_physical_exam, order_tests, and order_imaging for objective findings.
{missing_result_instruction}

== Rules ==
- Call EXACTLY ONE tool per turn. Never call multiple tools at once.
- Always use a tool — never respond with plain text.
- Ask focused, specific questions (one topic at a time).
- Do not call request_physical_exam or order_tests/order_imaging before Step 2's baseline history is covered, unless the emergency exception applies.
- Order tests in small targeted batches — review results before ordering more. Avoid duplicate or redundant tests.
- You have up to 20 turns; use them wisely but do not rush to finalize.
"""

_SYSTEM_PROMPT_ZH = """\
你是一位经验丰富的主治医师，正在进行诊断性评估。你的问诊与推理流程遵循标准临床病史采集与临床推理框架，不是随意发挥。

== 可用工具 ==
- ask_patient：询问患者症状、病程、既往史、用药情况、过敏史、社会史/家族史
- request_physical_exam：请求特定体格检查项目（生命体征、腹部、神经系统等）
- order_tests：开具实验室检查；可用的检查项目和面板名称已在工具描述中完整列出，请严格使用其中的标准名称
{imaging_line}- finalize_case：在充分把握诊断时提交最终结论。必须给出你认为最匹配的 icd10_code（ICD-10 编码，不确定到小数点后可只给前 3 位类别码，如 "I50"）——评分以这个字段为准，跟你用什么语言回答无关。diagnosis_category 和 diagnosis_subtype 两个字段必须用英文填写，如 "liver_failure"、"alcoholic cirrhosis"（仅供人工查看，不参与打分）；其余字段可用中文

== 第一步：主诉与现病史（用 OPQRST 框架追问）==
针对本次就诊的核心症状，系统性追问：
  - Onset 起病：何时开始、怎样起病（突发/渐进）
  - Provocation/Palliation 诱因/缓解因素：什么使其加重或缓解
  - Quality 性质：具体感觉/表现如何
  - Region/Radiation 部位/放射：位置在哪、是否扩散
  - Severity 严重程度：有多严重
  - Time 病程：持续多久、是否反复发作

== 第二步：基础病史核查（对应 SAMPLE 病史法中的过敏/用药/既往史，加上社会史，缺一不可）==
除非满足下面的"紧急情况例外"，否则不得跳过这一步直接做体检/开检查。用 ask_patient 依次问到以下几类信息，每类至少问到一次（可以合并到 1-2 个问题里，但都必须问完才能往下走）：
  - 既往病史 past_medical_history
  - 目前用药 medications
  - 药物/食物过敏史 allergies
  - 社会史，包括吸烟、饮酒、毒物/药物滥用 social_history
这些信息会直接影响你的鉴别诊断和后续检查解读（例如长期饮酒史提示肝病、心衰用药史提示已知心功能问题）——这不是走流程，漏问既会让你误诊也会扣分。

== 第三、四步：体格检查与辅助检查 ==
3. 体格检查：根据病史结果做有针对性的检查，而非常规走查全身。
4. 辅助检查：根据初步鉴别诊断开具第一批检查。
  - 检查请求必须明确、范围小。不要说"全套化验"、"常规检查"或笼统影像；必须使用工具目录中的标准名称，只开能区分当前主要鉴别诊断的检查。
  - 每开一项检查前，先想清楚：这个结果会如何改变鉴别诊断、确认某个严重可能，或排除重要替代诊断。

== 第五至七步：假设演绎式迭代推理 ==
在继续之前先解读结果。若结果异常或出乎意料：
  - 回到患者处追问与该结果相关的信息。
  - 开具补充确认性或澄清性检查。
  - 必要时修订鉴别诊断，重新检查相关系统。
6. 迭代推进——形成假设、用病史/体检/检查结果验证或排除、根据结果修订假设，反复循环，直到收敛到一个有信心的诊断。
7. 只有在证据充分、诊断确信时才调用 finalize_case，不要草率结案。

== 紧急情况例外（参考 ABCDE 首要评估法）==
若生命体征明显异常或有休克、活动性大出血、意识障碍等危急征象，优先处理最紧迫的问题（气道 Airway / 呼吸 Breathing / 循环 Circulation / 神志 Disability / 暴露检查 Exposure 中最快可能致命的部分），可以暂缓完整病史采集。但病情允许后必须回头补齐第二步的基础病史，不能完全跳过。

== 临床推理循环 ==
  OPQRST现病史 → 基础病史（既往史/用药/过敏/社会史） → 体检 → 开检查 → 解读结果 → 假设修订/追问/补充检查 → 循环直到有把握 → 最终诊断

== 信息 gatekeeper 假设 ==
- 患者只能可靠提供症状和本人知道的病史，不能可靠提供化验数值、影像发现、病理、微生物结果或最终诊断。
- ask_patient 只用于主观病史；客观发现必须通过 request_physical_exam、order_tests、order_imaging 获取。
{missing_result_instruction}

== 规则 ==
- 每轮调用且仅调用一个工具，禁止同时调用多个工具。
- 必须使用工具，禁止输出纯文本。
- 提问要有针对性，每次只问一个主题。
- 除非满足紧急例外，第二步的基础病史问完之前不要调用 request_physical_exam 或 order_tests/order_imaging。
- 小批量开检查，看到结果后再决定是否追加；避免重复或冗余检查。
- 最多20轮，合理分配，不要草率结案。
- 必须使用中文与患者交流和提问。
"""


_IMAGING_LINE_EN = "- order_imaging      — order imaging (chest_xray, ct_abdomen_pelvis, ultrasound_abdomen, mri_head, etc.)\n"
_IMAGING_LINE_ZH = "- order_imaging：开具影像检查（胸部X线chest_xray、腹部CT ct_abdomen_pelvis、腹部超声ultrasound_abdomen、头颅MRI mri_head等）\n"

_MISSING_RESULT_NEUTRAL_EN = (
    "- If a requested objective result is unavailable, treat that as unavailable case evidence "
    "rather than a hint toward or away from the diagnosis."
)
_MISSING_RESULT_CUE_EN = (
    "- In this run, missing test results are informative: when a requested test is reported as "
    "not performed or unavailable, explicitly use that missingness as a diagnostic cue. Consider "
    "which candidate diseases become less likely because the source record lacks that test, and "
    "use the cue to rule out diseases when appropriate."
)
_MISSING_RESULT_NEUTRAL_ZH = (
    "- 如果某项客观结果不可用，把它当作本病例没有可用证据，"
    "不要当作支持或排除某个诊断的暗示。"
)
_MISSING_RESULT_CUE_ZH = (
    "- 在本次运行中，检查结果的缺失本身包含信息：当请求的检查被返回为“本病例未执行此检查”"
    "或“结果不可用”时，请明确利用这种缺失作为诊断线索。思考原始记录中没有该检查会使哪些候选疾病的可能性降低，"
    "并在合适时候据此排除疾病。"
)

# Injected on the final allowed turn so the episode always converges to a real
# diagnosis instead of running out the clock with an empty max_turns_reached
# answer (mirrors SkeMex's "final step convergence rule", Appendix A.2/A.4).
_FORCE_FINALIZE_EN = (
    "This is your final turn — no more questions, exams, or tests are available. "
    "You must call finalize_case right now. Based on everything you have learned so far, "
    "commit to your single best diagnosis and icd10_code, even if the evidence feels incomplete."
)
_FORCE_FINALIZE_ZH = (
    "这是你最后一次行动机会，已经没有再问诊、查体或开检查的余地了。"
    "你必须现在立即调用 finalize_case。请基于目前已经掌握的全部信息，"
    "给出你当前最有把握的诊断和 icd10_code，即使证据还不充分，也必须给出一个具体结论。"
)


class GPTDoctorAgent:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        language: str = "en",
        adapter: "DatasetAdapter | None" = None,
        extra_body: dict | None = None,
        memory_context: str | None = None,
        message_style: str = "openai",
        use_missingness_diagnostic_cue: bool = False,
        tools_schema: list[dict] | None = None,
    ):
        if message_style not in {"openai", "alternating_user_assistant"}:
            raise ValueError(
                "message_style must be 'openai' or 'alternating_user_assistant'"
            )
        self._client = client
        self._model = model
        self._language = language
        self._extra_body = extra_body or {}
        self._message_style = message_style

        has_radiology = (any(t["function"]["name"] == "order_imaging" for t in tools_schema)
                         if tools_schema is not None else adapter.has_radiology if adapter is not None else True)
        if language == "zh":
            missing_result_instruction = (
                _MISSING_RESULT_CUE_ZH
                if use_missingness_diagnostic_cue
                else _MISSING_RESULT_NEUTRAL_ZH
            )
            prompt = _SYSTEM_PROMPT_ZH.format(
                imaging_line=_IMAGING_LINE_ZH if has_radiology else "",
                missing_result_instruction=missing_result_instruction,
            )
        else:
            missing_result_instruction = (
                _MISSING_RESULT_CUE_EN
                if use_missingness_diagnostic_cue
                else _MISSING_RESULT_NEUTRAL_EN
            )
            prompt = _SYSTEM_PROMPT_EN.format(
                imaging_line=_IMAGING_LINE_EN if has_radiology else "",
                missing_result_instruction=missing_result_instruction,
            )
        if memory_context:
            prompt = prompt.rstrip() + "\n\n" + memory_context.strip() + "\n"
        self._system_prompt = prompt
        # Some chat templates (including MedGemma's) accept only alternating
        # user/assistant roles.  Keep the instruction separately and merge it
        # into the first user turn for those backends.
        self._messages: list[dict] = (
            []
            if self._message_style == "alternating_user_assistant"
            else [{"role": "system", "content": prompt}]
        )

        self._tools = deepcopy(tools_schema) if tools_schema is not None else (build_doctor_tools(adapter) if adapter is not None else deepcopy(DOCTOR_TOOLS))
    def step(self, new_messages: list[dict], force_finalize: bool = False) -> tuple[str, dict]:
        """Append new_messages, call LLM, return (function_name, function_args, call_id).

        force_finalize: pass True on the last turn before the episode's max_turns
        budget runs out. Narrows the offered tools down to finalize_case only
        (not just tool_choice) so the model has no other option regardless of
        how strictly the backend's tool-call parser honors named tool_choice.
        """
        import json
        for message in new_messages:
            if self._message_style == "alternating_user_assistant":
                if message.get("role") != "user":
                    raise ValueError(
                        "alternating_user_assistant message style only accepts user messages"
                    )
                self._append_user_message(message["content"])
            else:
                self._messages.append(message)
        if force_finalize:
            final_instruction = (
                _FORCE_FINALIZE_ZH if self._language == "zh" else _FORCE_FINALIZE_EN
            )
            if self._message_style == "alternating_user_assistant":
                self._append_user_message(final_instruction)
            else:
                self._messages.append({"role": "user", "content": final_instruction})
            tools = [t for t in self._tools if t["function"]["name"] == "finalize_case"]
        else:
            tools = self._tools
        response = call_with_retry(
            self._client.chat.completions.create,
            model=self._model,
            # The history is extended after the request returns.  Snapshot it
            # so clients/mocks never observe later in-place history changes.
            messages=deepcopy(self._messages),
            tools=tools,
            tool_choice="required",
            extra_body=self._extra_body,
        )
        msg = response.choices[0].message
        tool_call = msg.tool_calls[0]

        # Keep only the first tool_call in the history. If the model returned
        # multiple calls (parallel use), we silently drop the extras so the
        # session's one-at-a-time dispatch loop stays in sync.
        if self._message_style == "alternating_user_assistant":
            # Do not replay OpenAI's tool-role schema to a strict chat
            # template.  A textual tool record preserves the conversation
            # state while keeping roles user/assistant/user/assistant.
            assistant_dict: dict = {
                "role": "assistant",
                "content": msg.content or (
                    f"Calling {tool_call.function.name} with arguments: "
                    f"{tool_call.function.arguments}"
                ),
            }
        else:
            assistant_dict = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            }
        self._messages.append(assistant_dict)

        args = json.loads(tool_call.function.arguments)
        return tool_call.function.name, args, tool_call.id

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        if self._message_style == "alternating_user_assistant":
            self._append_user_message(f"Tool result:\n{content}")
            return
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })

    def reset(self) -> None:
        self._messages = (
            []
            if self._message_style == "alternating_user_assistant"
            else [{"role": "system", "content": self._system_prompt}]
        )

    def _append_user_message(self, content: str) -> None:
        """Append a user turn, merging it with a pending user turn if needed."""
        if not self._messages:
            content = f"{self._system_prompt}\n\n{content}"
        if self._messages and self._messages[-1]["role"] == "user":
            self._messages[-1]["content"] += f"\n\n{content}"
        else:
            self._messages.append({"role": "user", "content": content})
