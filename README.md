# docqa

[![CI](https://github.com/piliotov/docqa/actions/workflows/ci.yml/badge.svg)](https://github.com/piliotov/docqa/actions/workflows/ci.yml)

Ask questions about your own documents and get answers with cited sources.

docqa is a small retrieval augmented generation (RAG) service. It splits documents into
chunks, indexes them for both semantic and keyword search, and answers questions using
only the retrieved context. It runs fully offline by default and can be pointed at any
OpenAI compatible model server, including a local Ollama instance.

## Features

- Ingest `.txt`, `.md` and `.pdf` files through the CLI or HTTP API
- Sentence aware chunking with overlap
- Hybrid retrieval: dense vectors and BM25 combined with reciprocal rank fusion
- Pluggable embeddings: dependency free hashing embedder or sentence-transformers
- Pluggable generation: offline extractive answers or any OpenAI compatible chat API
- Answers cite the chunks they came from
- Index persisted to disk
- Evaluation script reporting hit@k, MRR and keyword recall
- Typed, linted and tested, with CI on every push

## How it works

```
documents -> loader -> chunker -> embedder -----> vector store
                                        \-------> BM25 index
question  -> embedder + BM25 -> rank fusion -> top k chunks -> LLM -> answer + sources
```

Dense search catches paraphrases, BM25 catches exact terms like error codes or version
numbers. Rank fusion combines both without having to calibrate their scores against each
other. `DOCQA_ALPHA` controls the weight between them.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

docqa ingest eval/docs
docqa ask "What happens when a liveness probe fails?"
```

```
A liveness probe detects containers that are stuck. [1] When the liveness probe fails,
the kubelet restarts the container. [1]

Sources:
  [1] kubernetes-probes.md #0 (0.0164)
  [2] coffee-brewing.md #0 (0.0081)
  ...
```

## HTTP API

```bash
docqa serve
```

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/health` | Status and index size |
| GET | `/documents` | List indexed documents |
| POST | `/documents` | Upload a file (multipart) |
| POST | `/documents/text` | Add raw text as JSON |
| DELETE | `/documents/{doc_id}` | Remove a document |
| POST | `/ask` | Ask a question |

```bash
curl -F file=@notes.pdf localhost:8000/documents
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "How long do refunds take?", "k": 3}'
```

Interactive docs are available at `http://localhost:8000/docs`.

## Using a real model

The default extractive mode needs no model and no API key. To generate answers with an
LLM, point docqa at any OpenAI compatible endpoint.

Local and free with [Ollama](https://ollama.com):

```bash
ollama pull llama3.2
export DOCQA_LLM=openai
export DOCQA_LLM_MODEL=llama3.2
export DOCQA_LLM_BASE_URL=http://localhost:11434/v1
```

Hosted:

```bash
export DOCQA_LLM=openai
export DOCQA_LLM_MODEL=gpt-4o-mini
export DOCQA_LLM_BASE_URL=https://api.openai.com/v1
export DOCQA_LLM_API_KEY=...
```

For better semantic search install the optional embeddings and rebuild the index:

```bash
pip install -e ".[local]"
export DOCQA_EMBEDDER=sentence-transformers
docqa ingest path/to/docs
```

## Configuration

| Variable | Default | |
| -------- | ------- | - |
| `DOCQA_INDEX_DIR` | `data/index` | Where the index is stored |
| `DOCQA_EMBEDDER` | `hashing` | `hashing` or `sentence-transformers` |
| `DOCQA_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | |
| `DOCQA_LLM` | `extractive` | `extractive` or `openai` |
| `DOCQA_LLM_MODEL` | `llama3.2` | |
| `DOCQA_LLM_BASE_URL` | `http://localhost:11434/v1` | |
| `DOCQA_LLM_API_KEY` | | |
| `DOCQA_CHUNK_WORDS` | `120` | Max words per chunk |
| `DOCQA_TOP_K` | `4` | Chunks passed to the model |
| `DOCQA_ALPHA` | `0.5` | 1.0 dense only, 0.0 keyword only |

## Docker

```bash
docker build -t docqa .
docker run -p 8000:8000 -v docqa-data:/data docqa
```

## Evaluation

```bash
python eval/evaluate.py -k 3
```

```json
{
  "total": 12,
  "hit_at_1": 1.0,
  "hit_at_k": 1.0,
  "mrr": 1.0,
  "keyword_recall": 0.792
}
```

The questions live in `eval/questions.jsonl`. CI fails if hit@3 drops below 0.9.

## Development

```bash
make lint
make test
```

## Project layout

```
src/docqa/
  api.py         FastAPI app
  assistant.py   ingest and ask pipeline
  bm25.py        keyword index
  chunking.py    sentence aware chunker
  cli.py         command line interface
  config.py      settings from environment
  embeddings.py  hashing and sentence-transformers embedders
  llm.py         extractive and OpenAI compatible generators
  loaders.py     txt, md and pdf loading
  retriever.py   hybrid retrieval with rank fusion
  store.py       vector store with persistence
eval/            sample documents, questions and evaluation script
tests/           unit and API tests
```

## License

MIT
