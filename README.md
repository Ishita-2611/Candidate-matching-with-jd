# Candidate Semantic Matching Pipeline

This project converts candidate profiles into semantic JSON objects, then embeds those semantic objects with BGE-M3 and stores them in Qdrant as named multivectors.

## Flow

```text
candidates.jsonl
  -> semantic batch JSON files in batches/
  -> BGE-M3 named vectors
  -> Qdrant collection candidate_semantic_multivectors_bge_m3
```

`index.js` is intentionally the LLM demo file. It shows the Groq/OpenAI-compatible chat call that converts raw candidate JSON into the semantic object format.

## Setup

```bash
npm install
```

Copy `.env.example` to `.env` and set:

```text
GROQ_API_KEY=...
```

Do not commit `.env`, `candidates.jsonl`, model cache, local vector snapshots, or Qdrant storage.

## Generate Semantic Batches With LLM

This is the project demo path in `index.js`:

```powershell
$env:BATCH_LIMIT="5"
$env:CONCURRENCY="1"
$env:OUTPUT_BATCH_SIZE="5"
npm run process
```

Output is written to `batches/output_batch0001.json`.

## Generate Semantic Batches Without API Calls

For the full 1 lakh dataset, use the local deterministic generator so API limits do not block the run:

```powershell
$env:BATCH_LIMIT="100000"
$env:START_OFFSET="0"
$env:OUTPUT_BATCH_SIZE="1000"
$env:OUTPUT_DIR="batches"
npm run generate:local-batches
```

The current `batches/` folder is kept available for embeddings and is not ignored by Git.

## Start Qdrant

```powershell
npm run qdrant:up
```

Dashboard:

```text
http://localhost:6333/dashboard
```

## Store BGE-M3 Multivector Embeddings In Qdrant

Embed existing semantic batches and upsert them into one Qdrant collection:

```powershell
$env:BATCHES_DIR="batches"
$env:EMBED_LIMIT="5"
$env:QDRANT_COLLECTION="candidate_semantic_multivectors_bge_m3"
npm run embed:batches
```

Remove `EMBED_LIMIT` or set it to a large value when you want to process all batch objects. BGE-M3 is local and memory-heavy, so full 1 lakh processing will take a long time.

## Qdrant Shape

Each candidate is stored as one Qdrant point. The payload stays readable, and vectors are stored as Qdrant named vectors so the dashboard shows separate vector rows like the reference screenshot.

Payload example:

```json
{
  "candidate_id": "CAND_0000001",
  "metadata": {
    "years_of_experience": 6.9,
    "location": "Toronto",
    "country": "Canada",
    "open_to_work": true,
    "preferred_work_mode": "onsite",
    "notice_period_days": 60,
    "salary_range_lpa": {
      "min": 18.7,
      "max": 36.1
    }
  },
  "semantic_axes": {}
}
```

Named vectors stored on the same point:

```text
identity             length 1024
skills               length 1024
experience_summary   length 1024
domain               length 1024
execution_style      length 1024
trust_signals        length 1024
default              length 1024
```

## Useful Scripts

- `npm run process`: LLM semantic generation using `index.js`.
- `npm run generate:local-batches`: no-API semantic batch generation for large runs.
- `npm run embed:batches`: BGE-M3 multivector embedding storage in Qdrant.
- `npm run qdrant:up`: start local Qdrant with Docker.
- `npm test`: syntax check `index.js`.
