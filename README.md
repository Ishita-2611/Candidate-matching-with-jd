# Candidate Matching With JD

Pipeline for the India Runs Data & AI / Redrob candidate ranking challenge.

The repo is organized around two phases:

```text
OFFLINE, run before submission
candidates.jsonl -> semantic batches -> BGE-M3 embeddings in Qdrant -> RAG analysis / FAISS snapshot + signal cache

ONLINE, submission window
faiss_index.bin + id_map.json + jd_vector.npy + signals_cache.msgpack -> rank.py -> submission.csv + submission.xlsx
```

The online ranking step is CPU-only, no-network, and does not call Qdrant or any LLM API.

## Setup

```bash
npm install
pip install -r requirements.txt
```

Copy `.env.example` to `.env`. Set `QDRANT_URL` and `QDRANT_API_KEY` for Qdrant Cloud before embedding or searching.

Local Qdrant is optional for development:

```powershell
npm run qdrant:up
```

Local Qdrant dashboard:

```text
http://localhost:6333/dashboard
```

## Offline Pipeline

### 1. Parse JD

```powershell
$env:JD_INPUT_FILE="job_description.md"
$env:JD_OUTPUT_FILE="jd-semantic.json"
npm run parse:jd
```

### 2. Generate Semantic Candidate Batches

`generate-semantic-batches-v3.js` creates natural-language semantic axes. Most axes are local deterministic prose. The `execution_style` axis can use Groq offline with concurrency and cache.

```powershell
$env:INPUT_JSONL="candidates.jsonl"
$env:OUTPUT_DIR="batches"
$env:BATCH_LIMIT="100000"
$env:OUTPUT_BATCH_SIZE="1000"
$env:CONCURRENCY="50"
npm run generate:local-batches
```

Local-only fallback:

```powershell
$env:EXECUTION_STYLE_PROVIDER="local"
npm run generate:local-batches
```

Use the fine-tuned batch folders for embeddings. The old local deterministic `batches/` folder is no longer part of the repo.

### 2B. Fine-Tune Semantic Generator

If you have LLM-generated semantic examples, export them as chat fine-tuning JSONL:

```powershell
$env:FINETUNE_CANDIDATES_FILE="candidates.jsonl"
$env:FINETUNE_SEMANTIC_DIR="batches"
$env:FINETUNE_LIMIT="1000"
npm run finetune:prepare
```

Outputs:

```text
fine_tuning/semantic_train.jsonl
fine_tuning/semantic_validation.jsonl
fine_tuning/manifest.json
```

Use those files in OpenPipe, Fireworks, or another OpenAI-compatible fine-tuning provider. After the provider deploys a model id, generate semantic batches with:

```powershell
$env:FINETUNED_API_KEY="..."
$env:FINETUNED_BASE_URL="https://app.openpipe.ai/api/v1"
$env:FINETUNED_MODEL="openpipe:cruel-cooks-search"
$env:FINETUNED_LIMIT="1000"
npm run generate:finetuned
```

This writes to `batches_finetuned/` by default. Point `BATCHES_DIR` to the fine-tuned folder you want to embed.

### 3. Embed Candidates In Qdrant

```powershell
$env:BATCHES_DIR="batches_finetuned_1-20k"
$env:QDRANT_URL="https://your-cluster-id.region.cloud.qdrant.io"
$env:QDRANT_API_KEY="..."
$env:QDRANT_COLLECTION="candidate_semantic_multivectors_bge_m3"
npm run embed:batches
```

Named vectors:

```text
identity
skills
experience_summary
domain
execution_style
trust_signals
default
```

For Qdrant Cloud, the API key is sent using the `api-key` header by the Node scripts. Do not commit `.env`.

### 4. Embed JD

```powershell
$env:JD_SEMANTIC_FILE="jd-semantic.json"
$env:JD_EMBEDDINGS_FILE="jd-embeddings.json"
npm run embed:jd
```

### 5. Precompute Redrob/Honeypot Signals

```powershell
python precompute_signals.py --candidates candidates.jsonl --out signals_cache.msgpack
```

This produces behavioral scores, profile scores, honeypot/impossible-profile penalties, and deterministic reasoning.

Optional honeypot inspection:

```powershell
npm run detect:honeypots
```

### 6. Run Local RAG Analysis

The RAG path uses the Qdrant multivector collection as the retriever, applies hard filters and Redrob/honeypot penalties, then writes grounded candidate explanations.

```powershell
$env:JD_EMBEDDINGS_FILE="jd-embeddings.json"
$env:RAG_LIMIT="100"
$env:RAG_RECALL_LIMIT="500"
$env:RERANKER_MODEL="Xenova/ms-marco-MiniLM-L-6-v2"
$env:RAG_EXPLANATION_PROVIDER="local"
npm run rag:rank
```

Outputs:

```text
rag-results.json
rag-submission.csv
```

For offline-only LLM explanations on the top candidates:

```powershell
$env:RAG_EXPLANATION_PROVIDER="groq"
$env:RAG_EXPLAIN_LIMIT="100"
$env:RAG_EXPLANATION_CONCURRENCY="3"
npm run rag:rank
```

Do not use the LLM explanation mode inside the final no-network ranking run.

Ranking stages:

```text
1. Hard metadata filter from the semantic JD
2. BGE-M3 multivector retrieval in Qdrant
3. Cross-encoder reranking with RERANKER_MODEL
4. Skill overlap + Redrob business/profile signals
5. Disqualifier and honeypot penalties
6. Grounded explanation generation
```

Default reranker:

```text
Xenova/ms-marco-MiniLM-L-6-v2
```

The reranker reads the JD and each retrieved candidate together, so it is a real pairwise ranking model rather than just vector similarity.

### 7. Export FAISS Snapshot

```powershell
python export_faiss_snapshot.py --jd-embeddings jd-embeddings.json
```

Outputs:

```text
faiss_index.bin
id_map.json
jd_vector.npy
```

## Online Submission Command

```powershell
python rank.py --candidates candidates.jsonl --out submission.csv --xlsx-out submission.xlsx
```

`rank.py` excludes candidates with honeypot severity `>= 3` from the top-100 output, limits each justification to 15-18 words, and writes both CSV and XLSX.

`rank.py` loads:

```text
faiss_index.bin
id_map.json
signals_cache.msgpack
jd_vector.npy
```

It writes exactly:

```text
candidate_id,rank,score,reasoning
```

No network, no Qdrant, no LLM calls.

## Useful Commands

```powershell
npm test
npm run parse:jd
npm run finetune:prepare
npm run generate:finetuned
npm run generate:local-batches
npm run embed:batches
npm run embed:jd
npm run search:hybrid
npm run rag:rank
npm run detect:honeypots
python precompute_signals.py --candidates candidates.jsonl --out signals_cache.msgpack
python export_faiss_snapshot.py --jd-embeddings jd-embeddings.json
python rank.py --candidates candidates.jsonl --out submission.csv --xlsx-out submission.xlsx
```

## Generated Files

Ignored by default unless intentionally force-added:

```text
.env
candidates.jsonl
jd-semantic.json
jd-embeddings.json
qdrant_storage/
model_cache/
execution_style_cache/
faiss_index.bin
id_map.json
jd_vector.npy
signals_cache.msgpack
rag-results.json
rag-submission.csv
submission.csv
honeypot_candidates.json
```
