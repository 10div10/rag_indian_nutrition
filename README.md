# RAG Min — Indian Nutrition Q&A

Minimal, production-shaped RAG pipeline. Free tools only.

**Stack:** HuggingFace embeddings (bge-small) → Chroma (vector DB) → Groq (llama-3.1-8b-instant) → FastAPI → Docker → GitHub Actions CI → RAGAS eval.

## Run

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key
python3 ingest.py          # build index from data/
python3 query.py           # test a query
uvicorn app:app --reload   # serve API on :8000
```

Then open `frontend.html` in your browser to ask questions through a simple UI.

## Docker

```bash
docker build -t rag-min .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key rag-min
```

## Endpoints

- `GET /health`
- `POST /ask {"question": "...", "k": 4}`

## Evaluation (RAGAS)

RAGAS scores the pipeline against a held-out Q&A set, using Groq (`llama-3.1-8b-instant`) as the judge LLM and the same embeddings used for retrieval.

```bash
python3 eval.py
```

Results on the sample nutrition dataset:

| Metric | Score |
|---|---|
| Faithfulness | 0.875 |
| Answer Relevancy | 0.679 |
| Context Precision | 1.000 |
| Context Recall | 1.000 |

- **Faithfulness** — how much of the generated answer is actually grounded in retrieved context (low score = hallucination).
- **Answer Relevancy** — how directly the answer addresses the question asked.
- **Context Precision / Recall** — how well the right chunks get retrieved for a given question.

### Using your own dataset

This isn't locked to nutrition data. To point it at a different domain:

1. Drop your own `.txt` files into `data/` (delete or keep the sample nutrition files).
2. Rebuild the index: `python3 ingest.py`
3. Replace the `test_set` list in `eval.py` with real question/ground-truth pairs from your new domain, then run `python3 eval.py` to get fresh RAGAS scores.

No code changes needed elsewhere — ingestion, retrieval, and the API are domain-agnostic.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every push: installs dependencies, compiles all Python files, and builds the Docker image — so a green check confirms the whole pipeline actually builds, not just that it looks right.
