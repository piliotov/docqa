import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from docqa import Assistant

HERE = Path(__file__).parent


@dataclass
class Report:
    total: int
    hit_at_1: float
    hit_at_k: float
    mrr: float
    keyword_recall: float

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in self.__dict__.items()}


def evaluate(assistant: Assistant, questions: list[dict[str, object]], k: int) -> Report:
    hits1 = hitsk = rr = recall = 0.0
    for item in questions:
        expected = str(item["doc"])
        answer = assistant.ask(str(item["question"]), k=k)
        ranked = [s.doc_id for s in answer.sources]
        if ranked[:1] == [expected]:
            hits1 += 1
        if expected in ranked:
            hitsk += 1
            rr += 1 / (ranked.index(expected) + 1)
        keywords = [str(w).lower() for w in item["keywords"]]  # type: ignore[attr-defined]
        text = answer.text.lower()
        recall += sum(w in text for w in keywords) / len(keywords)
    n = len(questions)
    return Report(n, hits1 / n, hitsk / n, rr / n, recall / n)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answer quality")
    parser.add_argument("--docs", type=Path, default=HERE / "docs")
    parser.add_argument("--questions", type=Path, default=HERE / "questions.jsonl")
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--min-hit-at-k", type=float, default=0.0)
    args = parser.parse_args()

    assistant = Assistant(alpha=args.alpha, chunk_words=60)
    assistant.add_folder(args.docs)
    lines = args.questions.read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]
    report = evaluate(assistant, questions, args.k)
    print(json.dumps(report.as_dict(), indent=2))

    if report.hit_at_k < args.min_hit_at_k:
        print(f"hit@{args.k} {report.hit_at_k:.3f} below {args.min_hit_at_k}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
