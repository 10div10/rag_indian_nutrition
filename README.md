# RAG Min

Minimal, production-shaped RAG pipeline. Free tools only.

**Stack:** HuggingFace embeddings (bge-small) → Chroma (vector DB) → Groq (llama-3.1-8b-instant) → FastAPI → Docker → GitHub Actions CI → RAGAS eval.

## Run
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key
python ingest.py          # build index from data/
python query.py           # test a query
python eval.py            # RAGAS metrics
uvicorn app:app --reload  # serve API
```

## Docker
```bash
docker build -t rag-min .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key rag-min
```

## Endpoints
- `GET /health`
- `POST /ask {"question": "...", "k": 4}`
