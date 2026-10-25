"""Trajectory coverage and diagnostic agreement metrics."""
from core import icd

INFO_WEIGHTS = {'key': 3., 'supporting': 2., 'background': 1.}


def history_collection_score(collected_ids: list[str], units: list[dict]) -> float:
    hidden = [unit for unit in units if unit['patient_visible'] and not unit['initially_visible']]
    total = sum(INFO_WEIGHTS[unit['importance']] for unit in hidden)
    collected = set(collected_ids)
    return round(sum(INFO_WEIGHTS[unit['importance']] for unit in hidden if unit['info_id'] in collected) / total, 4) if total else 1.


def test_order_coverage(returned_ids: list[str], inventory: list[dict]) -> float:
    available = {item['evidence_id'] for item in inventory if not item['initially_visible'] and item['type'] in {'lab', 'microbiology', 'radiology'}}
    return round(len(set(returned_ids) & available) / len(available), 4) if available else 1.


def abnormal_test_coverage(returned_ids: list[str], inventory: list[dict]) -> float:
    abnormal = {item['evidence_id'] for item in inventory if not item['initially_visible'] and item.get('is_abnormal') and item['type'] in {'lab', 'microbiology'}}
    return round(len(set(returned_ids) & abnormal) / len(abnormal), 4) if abnormal else 1.


def diagnosis_category_accuracy(predicted: str, reference: str) -> int:
    return int(icd.same_block(predicted, reference))


def diagnosis_subtype_accuracy(predicted: str, reference: str) -> int:
    return int(icd.same_category3(predicted, reference))


def information_collection_score(history_score: float) -> float:
    return round(history_score, 4)
