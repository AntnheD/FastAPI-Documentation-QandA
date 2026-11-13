# FastAPI Documentation Q&A

A full-stack application for asking questions about FastAPI documentation.
Answers are generated from retrieved documentation passages and cite the
sections they come from; when the documentation does not support an answer,
the system abstains instead of guessing.

## 🚀 Tech Stack
- **Frontend:** static HTML/JavaScript page (`client/index.html`), served by the backend
- **Backend:** FastAPI / Python (`server/app.py`)
- **Q&A pipeline:** `src/docqa/` — BM25, dense (sentence-transformers) and hybrid
  retrieval, Hugging Face Transformer or extractive answer generation, and
  sentence-level grounding with citations

## 🗂️ Project Structure
```
client/              Demo web page (calls POST /ask)
server/app.py        FastAPI app: /health, /ask, serves client/
src/docqa/
  corpus.py          Chunk model and JSONL loader
  preprocess.py      Markdown docs -> chunks.jsonl
  retrieval/         BM25, dense, hybrid retrievers
  generation/        Prompting, generators, grounding/abstention
  pipeline.py        Question -> retrieval -> generation -> grounded answer
  config.py          JSON configuration with command-line overrides
  errors.py          Error types used by the CLI and the API
  cli.py             `python -m docqa <command>`
  questions.py, metrics.py, evaluate_*.py   Evaluation
configs/             default.json (dense + flan-t5-base), final.json (hybrid + Qwen2.5-0.5B)
data/processed/      Preprocessed documentation chunks
data/evaluation/     58 labelled evaluation questions (dev/val/test)
tests/               Unit and integration tests
```

## 📖 How to Run

### 1. Install
Python 3.10+ is required. A CPU is enough.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch   # optional: smaller CPU-only torch
pip install -e ".[server,test]"
```

### 2. Ask a question from the command line
```bash
python -m docqa ask "How do I make a query parameter optional?"
```
The default configuration downloads two small models from Hugging Face on
first use. For a quick run with no downloads, use BM25 retrieval and the
extractive generator:

```bash
python -m docqa ask "How do I make a query parameter optional?" \
  --set retrieval.method=bm25 --set generation.generator=extractive \
  --set generation.min_retrieval_score=0
```
Add `--debug` to see the retrieved passages and sentence support, or `--json`
for the full record.

### 3. Run the web app
```bash
uvicorn server.app:app --port 8000
```
Open http://127.0.0.1:8000/ for the demo page, or call the API directly:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How do I make a query parameter optional?"}'
```
The server reads its configuration from `DOCQA_CONFIG` (default
`configs/default.json`). To use the tuned final configuration:
`DOCQA_CONFIG=configs/final.json uvicorn server.app:app --port 8000`.

### 4. Other commands
```bash
python -m docqa info                         # resolved configuration and corpus status
python -m docqa check-questions              # validate the evaluation set against the corpus
python -m docqa eval-retrieval --split val --methods bm25,dense
python -m docqa eval-answers --split test --config configs/final.json --tag test_final
```

## ✅ Testing
```bash
python -m pytest
ruff check . && ruff format --check .
```
The tests need no model downloads; `tests/test_api_integration.py` runs the
full API -> pipeline -> retrieval -> generation path on the committed corpus.

## 👥 Contributors
- [AntnheD](https://github.com/AntnheD) (Antnhe Debebe)
- [Yabu920](https://github.com/Yabu920) (Yeabsra Abera)
-  [Rekik Alemayehu](https://github.com/rekikalemayehu) (Rekik Alemayehu)
- [tesamuel130](https://github.com/tesamuel130) (Samuel Tesfaye) — pipeline integration, configuration, evaluation and testing
