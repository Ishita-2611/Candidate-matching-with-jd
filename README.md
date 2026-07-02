# Redrob Candidate Ranking Submission

This repository produces the official top-100 CSV for the India Runs Data & AI / Redrob candidate ranking challenge.

The submitted file is:

```text
submission.csv
```

It contains exactly:

```text
candidate_id,rank,score,reasoning
```

## Approach

The expensive semantic retrieval step is precomputed. Candidate semantic text was embedded into Qdrant Cloud with BGE-M3 multivectors, then the released JD embedding was queried once to create:

```text
retrieval_candidates.json
```

The reproducible ranking step is offline. `rank.py` reads `candidates.jsonl` plus `retrieval_candidates.json`, recomputes profile, behavioral, location, experience-band, disqualifier, and honeypot features, then writes the final top 100.

The ranker is JD-specific for the Senior AI Engineer role:

- prioritizes embeddings/retrieval/ranking/vector-search evidence
- prefers India-based candidates, especially relevant Indian tech hubs
- prefers the JD's 5-9 year experience band
- downranks long notice periods, weak behavioral signals, and non-JD domains
- excludes severe honeypot/impossible-profile candidates
- writes concise reasoning using only candidate facts

## Reproduce Submission

Install Python 3.11+ or 3.12+. The final ranking step uses only the Python standard library.

Run:

```powershell
python rank.py --candidates candidates.jsonl --retrieval retrieval_candidates.json --out submission.csv
```

Expected runtime on this machine: about 13 seconds CPU-only.

No network, GPU, hosted LLM, Qdrant call, or external API is used during this ranking step.

## Optional Precompute

Only needed if regenerating the retrieval artifact from Qdrant Cloud:

```powershell
$env:QDRANT_URL="https://your-cluster.region.cloud.qdrant.io"
$env:QDRANT_API_KEY="..."
$env:QDRANT_COLLECTION="candidate_semantic_multivectors_bge_m3"
$env:JD_EMBEDDINGS_FILE="jd-embeddings.json"
$env:RETRIEVAL_ALL="true"
$env:RETRIEVAL_SCROLL_BATCH_SIZE="512"
npm run precompute:retrieval
```

This writes `retrieval_candidates.json`. The final submitted CSV does not require this step if the artifact is already present.

## Validation

Use:

```powershell
npm test
python rank.py --candidates candidates.jsonl --retrieval retrieval_candidates.json --out submission.csv
```

The current output validates as:

- 100 rows plus header
- ranks exactly 1 through 100
- no duplicate candidate IDs
- all candidate IDs exist in `candidates.jsonl`
- scores are monotonically non-increasing
- all candidates are India-based
- severe honeypot count in top 100 is 0

## Files

Core submission files:

```text
rank.py
retrieval_candidates.json
submission.csv
README.md
requirements.txt
package.json
precompute-retrieval.js
job_description.md
```

Large/generated files such as semantic text batches, local Qdrant storage, caches, and XLSX scratch outputs are intentionally ignored.
