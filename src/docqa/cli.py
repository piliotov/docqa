import argparse
import sys
from pathlib import Path

from docqa.assistant import Assistant
from docqa.config import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docqa", description="Ask questions about documents")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="index a folder of documents")
    ingest.add_argument("folder", type=Path)

    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=None)

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    settings = Settings.from_env()

    if args.command == "serve":
        import uvicorn

        from docqa.api import create_app

        uvicorn.run(create_app(settings=settings), host=args.host, port=args.port)
        return 0

    assistant = Assistant.from_settings(settings)
    if args.command == "ingest":
        if not args.folder.is_dir():
            print(f"not a folder: {args.folder}", file=sys.stderr)
            return 1
        added = assistant.add_folder(args.folder)
        assistant.save(settings.index_dir)
        for name, chunks in added.items():
            print(f"{name}: {chunks} chunks")
        print(f"indexed {len(added)} documents into {settings.index_dir}")
        return 0

    answer = assistant.ask(args.question, args.k)
    print(answer.text)
    if answer.sources:
        print("\nSources:")
        for n, source in enumerate(answer.sources, start=1):
            print(f"  [{n}] {source.doc_id} #{source.chunk} ({source.score:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
