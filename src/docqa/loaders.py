from pathlib import Path

from pypdf import PdfReader

SUPPORTED = {".txt", ".md", ".pdf"}


class UnsupportedFileError(ValueError):
    pass


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise UnsupportedFileError(f"unsupported file type: {suffix or path.name}")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def iter_documents(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
