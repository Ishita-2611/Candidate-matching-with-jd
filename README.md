# Candidate Semantic Matching Pipeline

This project converts candidate JSONL records into compact semantic candidate objects, creates BGE-M3 embeddings, and stores vectors in Qdrant.

## What Gets Stored

- Semantic JSON batches are written to `batches/`.
- BGE-M3 vectors are stored in Qdrant collection `candidate_multivectors_bge_m3`.
- Optional local vector snapshots are written to `vector_db/`.

The folders `batches/`, `vector_db/`, `qdrant_storage/`, and `model_cache/` are generated runtime data and are intentionally ignored by Git.

## Setup

Install dependencies:

```bash
npm install
```

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Set `GROQ_API_KEY` in `.env`.

## Start Qdrant

```bash
npm run qdrant:up
```

Qdrant dashboard:

```text
http://localhost:6333/dashboard
```

Stop Qdrant:

```bash
npm run qdrant:down
```

## Run The Pipeline

Process the first 5 candidates:

```bash
BATCH_LIMIT=5 CONCURRENCY=1 OUTPUT_BATCH_SIZE=5 npm run process
```

On PowerShell:

```powershell
$env:BATCH_LIMIT="5"; $env:CONCURRENCY="1"; $env:OUTPUT_BATCH_SIZE="5"; npm run process
```

The flow is:

```text
candidates.jsonl -> semantic candidate objects -> BGE-M3 embeddings -> Qdrant
```

Semantic objects stay in the generated batch files as normal JSON. Embeddings are stored separately as one Qdrant point per candidate with named vectors:

```json
{
  "candidate_id": "CAND_0000001",
  "metadata": {
    "yoe": 6.9,
    "location": "Toronto",
    "salary_min": 18.7,
    "salary_max": 36.1
  },
  "vectors": {
    "skills_vector": [0.01],
    "role_vector": [0.01],
    "experience_vector": [0.01],
    "impact_vector": [0.01],
    "education_vector": [0.01]
  }
}
```

## Embed Existing Semantic Output

If semantic objects already exist in a JSON file, embed them directly:

```bash
INPUT_JSON=batches/output_batch0001.json EMBED_LIMIT=100 npm run embed:output
```

On PowerShell:

```powershell
$env:INPUT_JSON="batches/output_batch0001.json"; $env:EMBED_LIMIT="100"; npm run embed:output
```

## Build Final Candidate Index Format

Use this when you want the final object shape with `metadata`, `semantic_docs`, `vectors`, and `trust_features`:

```powershell
$env:INPUT_JSON="batches/output_batch0001.json"; $env:FINAL_INDEX_LIMIT="1"; npm run build:final-index
```

This writes `final_candidate_index.json` locally and upserts vectors into separate Qdrant collections:

```text
candidate_identity
candidate_skills
candidate_experience
candidate_chunks
candidate_domain
candidate_execution
```

## Configuration

Important environment variables:

- `GROQ_API_KEY`: Groq API key.
- `GROQ_MODEL`: Chat model for semantic conversion.
- `GROQ_BASE_URL`: OpenAI-compatible Groq endpoint.
- `INPUT_JSONL`: Source candidate JSONL file.
- `BATCH_LIMIT`: Number of source records to process.
- `CONCURRENCY`: Semantic conversion concurrency. Use `1` when rate limits are tight.
- `OUTPUT_BATCH_SIZE`: Number of semantic objects per output batch.
- `EMBEDDING_MODEL`: Defaults to `Xenova/bge-m3`.
- `EMBEDDING_DIM`: Defaults to `1024`.
- `QDRANT_URL`: Defaults to `http://localhost:6333`.
- `QDRANT_COLLECTION`: Defaults to `candidate_multivectors_bge_m3`.
- `QDRANT_UPSERT_BATCH_SIZE`: Defaults to `4`.

## Notes

BGE-M3 is memory-heavy in Node, so scripts run with a larger Node heap. Qdrant upserts are chunked to avoid request resets.

Do not commit `.env`, `candidates.jsonl`, generated batches, local vector snapshots, model cache, or Qdrant storage.
