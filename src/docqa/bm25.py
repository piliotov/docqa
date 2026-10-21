import math
from collections import Counter

from docqa.tokenize import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._df: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, texts: list[str]) -> None:
        for text in texts:
            counts = Counter(tokenize(text))
            self._docs.append(counts)
            self._lengths.append(sum(counts.values()))
            self._df.update(counts.keys())

    def remove(self, indices: set[int]) -> None:
        keep = [i for i in range(len(self._docs)) if i not in indices]
        docs = [self._docs[i] for i in keep]
        self._docs, self._lengths, self._df = [], [], Counter()
        for counts in docs:
            self._docs.append(counts)
            self._lengths.append(sum(counts.values()))
            self._df.update(counts.keys())

    def scores(self, query: str) -> list[float]:
        n = len(self._docs)
        if n == 0:
            return []
        avg_len = sum(self._lengths) / n or 1.0
        terms = tokenize(query)
        result = []
        for counts, length in zip(self._docs, self._lengths, strict=True):
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * length / avg_len)
                score += idf * tf * (self.k1 + 1) / denom
            result.append(score)
        return result
