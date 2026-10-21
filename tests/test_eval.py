import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from evaluate import evaluate  # noqa: E402


def test_eval_set_meets_baseline(assistant):
    lines = (ROOT / "eval" / "questions.jsonl").read_text().splitlines()
    report = evaluate(assistant, [json.loads(line) for line in lines], k=3)
    assert report.total == len(lines)
    assert report.hit_at_k >= 0.9
    assert report.mrr >= 0.8
