"""Episodic memory with semantic recall and utility-based selection."""
from __future__ import annotations
import json
from pathlib import Path
from .embeddings import normalize
from core.presentation import visible_from_case

MEMORY_FORMAT = 'clinverse-memrl-v2'
REWARD_WEIGHTS = {'diagnosis_score': 0.5, 'information_collection_score': 0.3,
                  'abnormal_test_coverage': 0.1, 'test_order_coverage': 0.1}


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def reward_from_scores(scores: dict) -> float:
    missing = set(REWARD_WEIGHTS) - scores.keys()
    if missing:
        raise ValueError(f'Missing reward components: {sorted(missing)}')
    if any(not 0 <= float(scores[key]) <= 1 for key in REWARD_WEIGHTS):
        raise ValueError('Clinical scores must be within [0, 1]')
    return round(sum(float(scores[key]) * weight for key, weight in REWARD_WEIGHTS.items()), 4)


def intent_from_visible(visible: dict) -> str:
    return ' | '.join(f'{key}={visible[key]}' for key in ('age', 'gender', 'admission_type', 'vital_signs', 'chief_complaint', 'initial_information') if visible.get(key) is not None)


def intent_from_case(case: dict) -> str:
    return intent_from_visible(visible_from_case(case))


def intent_from_record(record: dict) -> str:
    return intent_from_visible(record['transcript']['initial_visible_info'])


def experience_from_record(record: dict) -> str:
    turns = [{'action': turn['function'], 'arguments': {k: v for k, v in turn['args'].items() if not k.startswith('_')},
              'observation': turn['response']} for turn in record['transcript']['turns']]
    return json.dumps({'trajectory': turns, 'diagnosis': record['transcript'].get('final_answer'),
                       'reward': reward_from_scores(record['scores'])}, ensure_ascii=False)


def memory_from_record(record: dict, embedding: list[float]) -> dict:
    reward = reward_from_scores(record['scores'])
    return {'memory_id': f"mem_{record['source']}_{record['case_id']}",
            'intent': intent_from_record(record), 'embedding': normalize(embedding),
            'experience': experience_from_record(record),
            'utility': {'q_value': reward, 'visits': 0},
            'metadata': {'case_id': record['case_id'], 'source': record['source'],
                         'success': record['scores']['diagnosis_score'] == 1.0}}


def empty_bank(model: str) -> dict:
    return {'format': MEMORY_FORMAT, 'embedding_model': model, 'items': []}


def build_memory_bank(records: list[dict], encoder) -> dict:
    bank = empty_bank(encoder.model)
    vectors = encoder.encode([intent_from_record(record) for record in records])
    if len(vectors) != len(records):
        raise ValueError('Embedding count does not match the training records')
    for record, vector in zip(records, vectors):
        add_or_replace_memory(bank, record, vector)
    return bank


def load_memory_bank(path: str | Path | None, *, model: str) -> dict:
    if path is None:
        return empty_bank(model)
    bank = json.loads(Path(path).read_text())
    if bank.get('format') != MEMORY_FORMAT or bank.get('embedding_model') != model:
        raise ValueError('Memory format or embedding model does not match the configuration')
    dimensions = set()
    ids = set()
    for item in bank['items']:
        item['embedding'] = normalize(item['embedding'])
        dimensions.add(len(item['embedding']))
        if item['memory_id'] in ids:
            raise ValueError('Memory IDs must be unique')
        ids.add(item['memory_id'])
        if not 0 <= item['utility']['q_value'] <= 1:
            raise ValueError('Memory utility must be within [0, 1]')
    if len(dimensions) > 1:
        raise ValueError('Memory embedding dimensions must agree')
    return bank


def save_memory_bank(bank: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(bank, ensure_ascii=False, indent=2))
    temporary.replace(path)


def retrieve_memories(bank: dict, query: list[float], *, top_k: int = 3,
                      recall_k: int = 20, min_similarity: float = 0.0) -> list[dict]:
    if not 1 <= top_k <= recall_k:
        raise ValueError('Recall size must be at least the positive selection size')
    query = normalize(query)
    candidates = []
    for item in bank['items']:
        vector = item['embedding']
        if len(vector) != len(query):
            raise ValueError('Query and memory embedding dimensions differ')
        similarity = sum(a * b for a, b in zip(query, vector))
        if similarity >= min_similarity:
            candidates.append({**item, 'retrieval': {'similarity': similarity, 'q_value': item['utility']['q_value']}})
    recalled = sorted(candidates, key=lambda item: item['retrieval']['similarity'], reverse=True)[:recall_k]
    return sorted(recalled, key=lambda item: (item['utility']['q_value'], item['retrieval']['similarity']), reverse=True)[:top_k]


def format_memory_context(memories: list[dict], *, language: str = 'en') -> str:
    if not memories:
        return ''
    header = 'Use these prior experiences as diagnostic strategies, not as facts about the current patient.'
    return header + '\n\n' + '\n\n'.join(f"Intent: {item['intent']}\n{item['experience']}" for item in memories)


def update_memory_values(bank: dict, selected_ids: list[str], reward: float, *, alpha: float = 0.1) -> list[dict]:
    if not 0 < alpha <= 1 or not 0 <= reward <= 1:
        raise ValueError('Learning rate and reward are outside their supported ranges')
    updates = []
    for item in bank['items']:
        if item['memory_id'] in selected_ids:
            utility = item['utility']
            old = utility['q_value']
            utility['q_value'] = (1 - alpha) * old + alpha * reward
            utility['visits'] += 1
            updates.append({'memory_id': item['memory_id'], 'old_q': old, 'new_q': utility['q_value']})
    return updates


def add_or_replace_memory(bank: dict, record: dict, embedding: list[float]) -> None:
    item = memory_from_record(record, embedding)
    if bank['items'] and len(item['embedding']) != len(bank['items'][0]['embedding']):
        raise ValueError('Memory embedding dimensions differ')
    for existing in bank['items']:
        if existing['memory_id'] == item['memory_id']:
            item['utility'] = dict(existing['utility'])
    bank['items'] = [existing for existing in bank['items'] if existing['memory_id'] != item['memory_id']]
    bank['items'].append(item)
