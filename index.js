import OpenAI from "openai";
import { env, pipeline } from "@xenova/transformers";
import { createHash } from "node:crypto";
import { appendFileSync, createReadStream, mkdirSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import readline from "node:readline";

env.allowLocalModels = true;
env.cacheDir = process.env.TRANSFORMERS_CACHE || "./model_cache";

try {
  process.loadEnvFile?.();
} catch (error) {
  if (error.code !== "ENOENT") {
    throw error;
  }
}

const apiKey = process.env.GROQ_API_KEY;
const model = process.env.GROQ_MODEL || "meta-llama/llama-4-scout-17b-16e-instruct";
const groqBaseUrl = process.env.GROQ_BASE_URL || "https://api.groq.com/openai/v1";

if (!apiKey) {
  throw new Error("GROQ_API_KEY is required");
}

const openai = new OpenAI({
  apiKey,
  baseURL: groqBaseUrl,
});

// Prompt builder

function buildPrompt(candidate) {
  return `
You are a technical recruiter assistant. Convert the candidate profile below into a compact semantic candidate object.
Use factual, information-dense values inferred only from the candidate JSON.
Return valid JSON only. Do not include markdown fences, comments, trailing commas, or explanatory text.

Candidate JSON:
${JSON.stringify(candidate, null, 2)}

Produce ONLY a JSON object in exactly this shape:
{
  "candidate_id": "<same candidate_id from input>",
  "metadata": {
    "years_of_experience": <number>,
    "location": "<string>",
    "country": "<string>",
    "open_to_work": <boolean>,
    "preferred_work_mode": "<string>",
    "notice_period_days": <number>,
    "salary_range_lpa": {
      "min": <number>,
      "max": <number>
    }
  },
  "semantic_axes": {
    "identity": {
      "role_family": "<primary role family>",
      "secondary_roles": ["<role>", "<role>", "<role>"],
      "seniority": "<seniority>",
      "career_transition": {
        "from": "<source career area>",
        "to": "<target career area>"
      }
    },
    "skills": {
      "core_production_skills": ["<skill>"],
      "ml_skills": ["<skill>"],
      "ml_infra_skills": ["<skill>"],
      "weak_skills": ["<skill>"],
      "noisy_non_relevant_skills": ["<skill>"]
    },
    "experience_summary": {
      "system_types": ["<system type>"],
      "scale": {
        "daily_data_processed": "<string or unknown>",
        "source_systems": <number or 0>,
        "realtime_systems": <boolean>
      },
      "production_maturity": "<low|moderate|high>",
      "ml_maturity": "<low|moderate-low|moderate|high>"
    },
    "experience_chunks": [
      {
        "id": "chunk_1",
        "description": "<dense factual description>",
        "tags": ["<tag>"]
      }
    ],
    "domain": {
      "primary_domains": ["<domain>"],
      "secondary_domains": ["<domain>"],
      "missing_domains": ["<domain>"]
    },
    "execution_style": {
      "shipping_bias": "<low|medium|high>",
      "product_mindset": "<low|medium|high>",
      "research_bias": "<low|medium|high>",
      "ambiguity_tolerance": "<low|medium|high>",
      "ownership": "<low|medium|high>",
      "system_design_depth": "<low|medium|high>"
    },
    "trust_signals": {
      "github_score": <number>,
      "profile_completeness": <number>,
      "recruiter_response_rate": <number>,
      "interview_completion_rate": <number>,
      "offer_acceptance_rate": <number>
    }
  }
}
`.trim();
}

// Main conversion function

async function convertCandidateToSemanticObject(candidate) {
  const prompt = buildPrompt(candidate);

  const response = await openai.chat.completions.create({
    model,
    messages: [
      {
        role: "user",
        content: prompt,
      }
    ],
    temperature: 0.2,
    max_completion_tokens: 4096,
  });
  
  const raw = (response.choices[0]?.message?.content || "").trim();

  // Strip markdown fences if model adds them despite instructions
  const cleaned = raw.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "").trim();

  let parsed;
  try {
    parsed = JSON.parse(cleaned);
  } catch (err) {
    throw new Error(`Model returned non-JSON output:\n${raw}`);
  }

  const requiredKeys = ["candidate_id", "metadata", "semantic_axes"];

  for (const key of requiredKeys) {
    if (!parsed[key]) {
      throw new Error(`Missing or invalid key in response: "${key}"`);
    }
  }

  return parsed;
}

const convertCandidateToSemanticText = convertCandidateToSemanticObject;

// Batch processing with concurrency control

async function convertBatch(candidates, { concurrency = 5, onProgress } = {}) {
  const results = [];
  const errors = [];

  // Process in chunks to respect rate limits
  for (let i = 0; i < candidates.length; i += concurrency) {
    const chunk = candidates.slice(i, i + concurrency);

    const settled = await Promise.allSettled(
      chunk.map((c) => convertCandidateToSemanticText(c))
    );

    settled.forEach((outcome, idx) => {
      const candidate = chunk[idx];
      if (outcome.status === "fulfilled") {
        results.push(outcome.value);
      } else {
        errors.push({ candidate_id: candidate.candidate_id, error: outcome.reason.message });
        console.error(`[ERROR] ${candidate.candidate_id}: ${outcome.reason.message}`);
      }
    });

    if (onProgress) {
      onProgress({ done: Math.min(i + concurrency, candidates.length), total: candidates.length });
    }
  }

  return { results, errors };
}

async function readFirstJsonlObjects(filePath, limit) {
  const objects = [];
  const rl = readline.createInterface({
    input: createReadStream(filePath, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }

    objects.push(JSON.parse(trimmed));

    if (objects.length >= limit) {
      rl.close();
      break;
    }
  }

  return objects;
}

function parsePositiveInt(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryable(error) {
  const status = error?.status || error?.code;
  return status === 408 || status === 409 || status === 429 || (status >= 500 && status < 600);
}

async function convertWithRetry(candidate, attempts = 4) {
  let lastError;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await convertCandidateToSemanticObject(candidate);
    } catch (error) {
      lastError = error;

      if (attempt >= attempts || !isRetryable(error)) {
        break;
      }

      const delay = 1000 * 2 ** (attempt - 1) + Math.floor(Math.random() * 500);
      await sleep(delay);
    }
  }

  throw lastError;
}

function batchFileName(outputDir, batchNo) {
  return `${outputDir}/output_batch${String(batchNo).padStart(4, "0")}.json`;
}

function vectorFileName(vectorDbDir, batchNo) {
  return `${vectorDbDir}/vectors_batch${String(batchNo).padStart(4, "0")}.json`;
}

const embeddingProvider = process.env.EMBEDDING_PROVIDER || "transformers";
const embeddingModel = process.env.EMBEDDING_MODEL || "Xenova/bge-m3";
let embeddingPipeline;
let embeddingPipelinePromise;

function embeddingModelName() {
  return embeddingModel;
}

async function getEmbeddingPipeline() {
  if (!embeddingPipeline) {
    embeddingPipelinePromise ||= pipeline("feature-extraction", embeddingModel);
    embeddingPipeline = await embeddingPipelinePromise;
  }

  return embeddingPipeline;
}

function qdrantPointId(value) {
  const hex = createHash("sha256").update(value).digest("hex").slice(0, 32);
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function hashToUInt32(value) {
  return createHash("sha256").update(value).digest().readUInt32BE(0);
}

function tokenize(text) {
  return String(text).toLowerCase().match(/[a-z0-9+#.-]+/g) || [];
}

function embedTextWithHashing(text, dimension) {
  const vector = Array.from({ length: dimension }, () => 0);

  for (const token of tokenize(text)) {
    const hash = hashToUInt32(token);
    const index = hash % dimension;
    const sign = hashToUInt32(`${token}:sign`) % 2 === 0 ? 1 : -1;
    vector[index] += sign;
  }

  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
  return vector.map((value) => Number((value / norm).toFixed(6)));
}

async function embedText(text, dimension) {
  if (embeddingProvider === "local-hashing") {
    return embedTextWithHashing(text, dimension);
  }

  const extractor = await getEmbeddingPipeline();
  const output = await extractor(text, { pooling: "mean", normalize: true });
  const embedding = Array.from(output.data, (value) => Number(value.toFixed(6)));

  if (embedding.length !== dimension) {
    throw new Error(`Embedding dimension mismatch for ${embeddingModel}: expected ${dimension}, got ${embedding.length}`);
  }

  return embedding;
}

function axisText(semanticObject, axis) {
  const value = semanticObject.semantic_axes?.[axis];
  return JSON.stringify(value ?? {});
}

const semanticAxes = [
  "identity",
  "skills",
  "experience_summary",
  "experience_chunks",
  "domain",
  "execution_style",
  "trust_signals",
];

function semanticText(semanticObject) {
  return [
    `candidate_id: ${semanticObject.candidate_id}`,
    `metadata: ${JSON.stringify(semanticObject.metadata ?? {})}`,
    ...semanticAxes.map((axis) => `${axis}: ${axisText(semanticObject, axis)}`),
  ].join("\n");
}

async function buildVectorRecords(semanticObject, { dimension, batchNo }) {
  const modelName = embeddingModelName();
  const fullText = semanticText(semanticObject);
  const axisRecords = [];

  for (const axis of semanticAxes) {
    const text = axisText(semanticObject, axis);

    axisRecords.push({
      id: `${semanticObject.candidate_id}:${axis}`,
      candidate_id: semanticObject.candidate_id,
      axis,
      text,
      embedding_model: modelName,
      embedding_dimension: dimension,
      embedding: await embedText(text, dimension),
      metadata: semanticObject.metadata,
      source_batch_no: batchNo,
    });
  }

  return [
    {
      id: `${semanticObject.candidate_id}:full`,
      candidate_id: semanticObject.candidate_id,
      axis: "full",
      text: fullText,
      embedding_model: modelName,
      embedding_dimension: dimension,
      embedding: await embedText(fullText, dimension),
      metadata: semanticObject.metadata,
      source_batch_no: batchNo,
    },
    ...axisRecords,
  ];
}

function vectorRecordToQdrantPoint(record) {
  return {
    id: qdrantPointId(record.id),
    vector: record.embedding,
    payload: {
      record_id: record.id,
      candidate_id: record.candidate_id,
      axis: record.axis,
      text: record.text,
      embedding_model: record.embedding_model,
      embedding_dimension: record.embedding_dimension,
      metadata: record.metadata,
      source_batch_no: record.source_batch_no,
    },
  };
}

async function qdrantRequest(qdrantUrl, path, { method = "GET", body } = {}) {
  const response = await fetch(`${qdrantUrl}${path}`, {
    method,
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Qdrant ${method} ${path} failed: ${response.status} ${text}`);
  }

  return response.json();
}

async function ensureQdrantCollection({ qdrantUrl, collectionName, dimension }) {
  const existing = await fetch(`${qdrantUrl}/collections/${collectionName}`);

  if (existing.ok) {
    const body = await existing.json();
    const existingSize = body.result?.config?.params?.vectors?.size;

    if (existingSize && existingSize !== dimension) {
      throw new Error(
        `Qdrant collection ${collectionName} has dimension ${existingSize}, expected ${dimension}. Use a new QDRANT_COLLECTION or delete the old collection.`
      );
    }

    return;
  }

  await qdrantRequest(qdrantUrl, `/collections/${collectionName}`, {
    method: "PUT",
    body: {
      vectors: {
        size: dimension,
        distance: "Cosine",
      },
    },
  });
}

async function upsertQdrantVectors({ qdrantUrl, collectionName, vectorRecords }) {
  if (vectorRecords.length === 0) {
    return;
  }

  const chunkSize = parsePositiveInt(process.env.QDRANT_UPSERT_BATCH_SIZE, 4);

  for (let i = 0; i < vectorRecords.length; i += chunkSize) {
    await qdrantRequest(qdrantUrl, `/collections/${collectionName}/points?wait=true`, {
      method: "PUT",
      body: {
        points: vectorRecords.slice(i, i + chunkSize).map(vectorRecordToQdrantPoint),
      },
    });
  }
}

async function convertJsonlFile(
  filePath,
  {
    limit,
    concurrency,
    batchSize,
    outputDir,
    manifestJson,
    errorJsonl,
    vectorDbDir,
    vectorIndexJsonl,
    embeddingDimension,
    useQdrant,
    qdrantUrl,
    qdrantCollection,
  }
) {
  const manifest = [];
  let processed = 0;
  let succeeded = 0;
  let failed = 0;
  let vectorCount = 0;
  let currentBatchNo = 1;
  let currentBatch = [];
  let pending = [];

  mkdirSync(outputDir, { recursive: true });
  mkdirSync(vectorDbDir, { recursive: true });
  writeFileSync(errorJsonl, "");
  writeFileSync(vectorIndexJsonl, "");

  if (useQdrant) {
    await ensureQdrantCollection({
      qdrantUrl,
      collectionName: qdrantCollection,
      dimension: embeddingDimension,
    });
  }

  writeFileSync(
    `${vectorDbDir}/config.json`,
    `${JSON.stringify(
      {
        embedding_model: embeddingModelName(),
        embedding_dimension: embeddingDimension,
        distance: "cosine",
        qdrant: useQdrant
          ? {
              url: qdrantUrl,
              collection: qdrantCollection,
            }
          : undefined,
        record_shape: {
          id: "candidate_id:axis",
          candidate_id: "string",
          axis: "full|identity|skills|experience_summary|experience_chunks|domain|execution_style|trust_signals",
          text: "semantic text used for embedding",
          embedding: "normalized numeric vector",
          metadata: "candidate metadata",
        },
      },
      null,
      2
    )}\n`
  );

  async function writeBatchIfReady(force = false) {
    if (currentBatch.length === 0 || (!force && currentBatch.length < batchSize)) {
      return;
    }

    const file = batchFileName(outputDir, currentBatchNo);
    const vectorFile = vectorFileName(vectorDbDir, currentBatchNo);
    const vectorRecords = [];

    for (const semanticObject of currentBatch) {
      vectorRecords.push(
        ...(await buildVectorRecords(semanticObject, {
          dimension: embeddingDimension,
          batchNo: currentBatchNo,
        }))
      );
    }

    writeFileSync(file, `${JSON.stringify(currentBatch, null, 2)}\n`);
    writeFileSync(vectorFile, `${JSON.stringify(vectorRecords, null, 2)}\n`);
    appendFileSync(vectorIndexJsonl, vectorRecords.map((record) => JSON.stringify(record)).join("\n") + "\n");

    if (useQdrant) {
      await upsertQdrantVectors({
        qdrantUrl,
        collectionName: qdrantCollection,
        vectorRecords,
      });
    }

    vectorCount += vectorRecords.length;

    manifest.push({
      batch_no: currentBatchNo,
      file,
      count: currentBatch.length,
      vector_file: vectorFile,
      vector_count: vectorRecords.length,
      first_candidate_id: currentBatch[0]?.candidate_id,
      last_candidate_id: currentBatch.at(-1)?.candidate_id,
    });

    console.log(`Wrote ${file} (${currentBatch.length} objects, ${vectorRecords.length} vectors)`);
    currentBatchNo += 1;
    currentBatch = [];
  }

  async function flush() {
    const chunk = pending;
    pending = [];

    const settled = await Promise.allSettled(chunk.map((candidate) => convertWithRetry(candidate)));

    for (const [index, outcome] of settled.entries()) {
      const candidate = chunk[index];
      processed += 1;

      if (outcome.status === "fulfilled") {
        succeeded += 1;
        currentBatch.push(outcome.value);
        await writeBatchIfReady();
      } else {
        failed += 1;
        appendFileSync(
          errorJsonl,
          `${JSON.stringify({
            candidate_id: candidate.candidate_id,
            error: outcome.reason?.message || String(outcome.reason),
          })}\n`
        );
      }

      if (processed % 25 === 0 || processed === limit) {
        console.log(`Progress ${processed}/${limit} | ok=${succeeded} failed=${failed}`);
      }
    }
  }

  const rl = readline.createInterface({
    input: createReadStream(filePath, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }

    pending.push(JSON.parse(trimmed));

    if (pending.length >= concurrency) {
      await flush();
    }

    if (processed + pending.length >= limit) {
      rl.close();
      break;
    }
  }

  if (pending.length > 0) {
    await flush();
  }

  await writeBatchIfReady(true);
  writeFileSync(manifestJson, `${JSON.stringify(manifest, null, 2)}\n`);

  return { processed, succeeded, failed, batches: manifest.length, vectorCount };
}

async function main() {
  try {
    const inputJsonl = process.env.INPUT_JSONL || "candidates.jsonl";
    const limit = parsePositiveInt(process.env.BATCH_LIMIT, 5000);
    const concurrency = parsePositiveInt(process.env.CONCURRENCY, 5);
    const batchSize = parsePositiveInt(process.env.OUTPUT_BATCH_SIZE, 100);
    const outputDir = process.env.OUTPUT_DIR || "batches";
    const manifestJson = process.env.MANIFEST_JSON || `${outputDir}/manifest.json`;
    const errorJsonl = process.env.ERROR_JSONL || `${outputDir}/output.errors.jsonl`;
    const vectorDbDir = process.env.VECTOR_DB_DIR || "vector_db";
    const vectorIndexJsonl = process.env.VECTOR_INDEX_JSONL || `${vectorDbDir}/index.jsonl`;
    const embeddingDimension = parsePositiveInt(process.env.EMBEDDING_DIM, 1024);
    const useQdrant = process.env.USE_QDRANT !== "false";
    const qdrantUrl = process.env.QDRANT_URL || "http://localhost:6333";
    const qdrantCollection = process.env.QDRANT_COLLECTION || "candidate_embeddings_bge_m3";

    console.log("Starting semantic object generation...\n");
    console.log("Provider: groq");
    console.log(`Input JSONL: ${inputJsonl}`);
    console.log(`Model: ${model}`);
    console.log(`Groq base URL configured: ${groqBaseUrl ? "Yes" : "No"}`);
    console.log(`Limit: ${limit}`);
    console.log(`Concurrency: ${concurrency}`);
    console.log(`Output batch size: ${batchSize}`);
    console.log(`Output dir: ${outputDir}`);
    console.log(`Vector DB dir: ${vectorDbDir}`);
    console.log(`Embedding provider: ${embeddingProvider}`);
    console.log(`Embedding model: ${embeddingModelName()}`);
    console.log(`Embedding dimension: ${embeddingDimension}`);
    console.log(`Qdrant enabled: ${useQdrant ? "Yes" : "No"}`);
    if (useQdrant) {
      console.log(`Qdrant URL: ${qdrantUrl}`);
      console.log(`Qdrant collection: ${qdrantCollection}`);
    }
    console.log(`API Key configured: ${apiKey ? "Yes" : "No"}`);

    const summary = await convertJsonlFile(inputJsonl, {
      limit,
      concurrency,
      batchSize,
      outputDir,
      manifestJson,
      errorJsonl,
      vectorDbDir,
      vectorIndexJsonl,
      embeddingDimension,
      useQdrant,
      qdrantUrl,
      qdrantCollection,
    });

    console.log(`\nSaved ${summary.succeeded} semantic objects across ${summary.batches} batch files`);
    console.log(`Saved ${summary.vectorCount} vector records to ${vectorDbDir}`);
    console.log(`Manifest: ${manifestJson}`);
    console.log(`Errors JSONL: ${errorJsonl}`);
  } catch (error) {
    console.error("Error during conversion:", error.message);
    console.error("Full error:", error);
    process.exit(1);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}

export {
  convertCandidateToSemanticText,
  convertCandidateToSemanticObject,
  convertBatch,
  convertJsonlFile,
  buildVectorRecords,
  embedText,
  embeddingModelName,
  ensureQdrantCollection,
  qdrantPointId,
  upsertQdrantVectors,
  vectorRecordToQdrantPoint,
  semanticText,
  readFirstJsonlObjects,
};
