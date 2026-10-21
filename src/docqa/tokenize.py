import re

_WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it its of on or that the
    this to was were what when where which who why will with does do did can""".split()
)


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _WORD.findall(text.lower())
    if drop_stopwords:
        return [t for t in tokens if t not in STOPWORDS]
    return tokens
