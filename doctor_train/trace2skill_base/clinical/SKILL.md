# Clinical Diagnosis Agent Skill

Use this skill when acting as a doctor in ClinVerse diagnostic episodes.

## Workflow

1. Start with focused history-taking around the chief complaint and timeline.
2. Ask baseline history before broad testing: past medical history, medications, allergies, and social history.
3. Request targeted physical exam findings that can separate the leading differential diagnoses.
4. Order tests or imaging in small, diagnosis-discriminative batches.
5. Interpret abnormal findings before adding more workup.
6. Revise the differential diagnosis after each new piece of evidence.
7. Finalize only after subjective history and objective findings support one best diagnosis.

## Guardrails

- Do not copy facts from prior cases into the current case.
- Do not treat unavailable objective results as evidence for or against a diagnosis.
- Do not finalize immediately unless the case is already clinically decisive.
- Keep the final diagnosis aligned with the strongest evidence and provide the closest ICD-10 code.
