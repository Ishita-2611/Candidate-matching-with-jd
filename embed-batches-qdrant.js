import { env, pipeline } from "@xenova/transformers";
import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

try {
  process.loadEnvFile?.();
} catch (error) {
  if (error.code !== "ENOENT") {
    throw error;
  }
}

env.allowLocalModels = true;
env.cacheDir = process.env.TRANSFORMERS_CACHE || "./model_cache";

const vectorNames = [
  "identity",
  "skills",
  "experience_summary",
  "domain",
  "execution_style",
  "trust_signals",
  "default",
];

const embeddingModel = process.env.EMBEDDING_MODEL || "Xenova/bge-m3";
const embeddingDimension = parsePositiveInt(process.env.EMBEDDING_DIM, 1024);
const qdrantUrl = (process.env.QDRANT_URL || "http://localhost:6333").replace(/\/+$/, "");
const qdrantApiKey = process.env.QDRANT_API_KEY || "";
const qdrantCollection = process.env.QDRANT_COLLECTION || "candidate_semantic_multivectors_bge_m3";
const batchesDir = process.env.BATCHES_DIR || "batches_finetuned_1-20k";

let extractorPromise;

function parsePositiveInt(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function qdrantPointId(value) {
  const hex = createHash("sha256").update(String(value)).digest("hex").slice(0, 32);
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

async function getExtractor() {
  extractorPromise ||= pipeline("feature-extraction", embeddingModel);
  return extractorPromise;
}

async function embedText(text) {
  const extractor = await getExtractor();
  const output = await extractor(String(text || ""), { pooling: "mean", normalize: true });
  const vector = Array.from(output.data, (value) => Number(value.toFixed(6)));

  if (vector.length !== embeddingDimension) {
    throw new Error(
      `Embedding dimension mismatch for ${embeddingModel}: expected ${embeddingDimension}, got ${vector.length}`
    );
  }

  return vector;
}

function stableJson(value) {
  return JSON.stringify(value ?? {});
}

function embeddingText(value) {
  if (typeof value === "string") {
    return value;
  }

  return stableJson(value);
}

function semanticDocuments(semanticObject) {
  const axes = semanticObject.semantic_axes ?? {};

  return {
    identity: embeddingText(axes.identity),
    skills: embeddingText(axes.skills),
    experience_summary: [
      embeddingText(axes.experience_summary ?? {}),
      ...(axes.experience_chunks ?? []).map((chunk) => chunk.description || embeddingText(chunk)),
    ]
      .filter(Boolean)
      .join("\n"),
    domain: embeddingText(axes.domain),
    execution_style: embeddingText(axes.execution_style),
    trust_signals: embeddingText(axes.trust_signals),
    default:
      typeof axes.default === "string"
        ? axes.default
        : [
            `candidate_id: ${semanticObject.candidate_id}`,
            `metadata: ${stableJson(semanticObject.metadata)}`,
            `identity: ${embeddingText(axes.identity)}`,
            `skills: ${embeddingText(axes.skills)}`,
            `experience_summary: ${embeddingText(axes.experience_summary)}`,
            `experience_chunks: ${(axes.experience_chunks ?? []).map((chunk) => chunk.description || stableJson(chunk)).join(" ")}`,
            `domain: ${embeddingText(axes.domain)}`,
            `execution_style: ${embeddingText(axes.execution_style)}`,
            `trust_signals: ${embeddingText(axes.trust_signals)}`,
          ].join("\n"),
  };
}

async function buildPoint(semanticObject, sourceBatch) {
  const docs = semanticDocuments(semanticObject);
  const vector = {};

  for (const vectorName of vectorNames) {
    vector[vectorName] = await embedText(docs[vectorName]);
  }

  return {
    id: qdrantPointId(semanticObject.candidate_id),
    vector,
    payload: {
      candidate_id: semanticObject.candidate_id,
      metadata: semanticObject.metadata ?? {},
      semantic_axes: semanticObject.semantic_axes ?? {},
      semantic_documents: docs,
      embedding_model: embeddingModel,
      embedding_dimension: embeddingDimension,
      source_batch: sourceBatch,
    },
  };
}

async function qdrantRequest(path, { method = "GET", body } = {}) {
  const response = await fetch(`${qdrantUrl}${path}`, {
    method,
    headers: {
      connection: "close",
      ...(qdrantApiKey ? { "api-key": qdrantApiKey } : {}),
      ...(body ? { "content-type": "application/json" } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Qdrant ${method} ${path} failed: ${response.status} ${text}`);
  }

  return response.json();
}

async function ensureCollection() {
  try {
    const body = await qdrantRequest(`/collections/${qdrantCollection}`);
    const vectors = body.result?.config?.params?.vectors ?? {};
    const existingNames = Object.keys(vectors).sort();
    const expectedNames = [...vectorNames].sort();
    const namesMatch = JSON.stringify(existingNames) === JSON.stringify(expectedNames);
    const sizesMatch = expectedNames.every((name) => vectors[name]?.size === embeddingDimension);

    if (!namesMatch || !sizesMatch) {
      throw new Error(
        `Collection ${qdrantCollection} already exists with a different named-vector schema. Use a new QDRANT_COLLECTION or delete it from Qdrant.`
      );
    }

    return;
  } catch (error) {
    if (!String(error.message || "").includes("failed: 404")) {
      throw error;
    }
  }

  await qdrantRequest(`/collections/${qdrantCollection}`, {
    method: "PUT",
    body: {
      vectors: Object.fromEntries(
        vectorNames.map((name) => [
          name,
          {
            size: embeddingDimension,
            distance: "Cosine",
          },
        ])
      ),
    },
  });
}

async function upsertPoints(points) {
  if (points.length === 0) {
    return;
  }

  await qdrantRequest(`/collections/${qdrantCollection}/points?wait=true`, {
    method: "PUT",
    body: { points },
  });
}

async function existingPointIds(ids) {
  if (ids.length === 0) {
    return new Set();
  }

  const body = await qdrantRequest(`/collections/${qdrantCollection}/points`, {
    method: "POST",
    body: {
      ids,
      with_payload: false,
      with_vector: false,
    },
  });

  return new Set((body.result ?? []).map((point) => point.id));
}

function batchFiles() {
  return readdirSync(batchesDir)
    .filter((name) => /^output_batch\d{4}\.json$/i.test(name))
    .sort()
    .map((name) => `${batchesDir}/${name}`);
}

async function embedBatches() {
  const limit = parsePositiveInt(process.env.EMBED_LIMIT, Number.MAX_SAFE_INTEGER);
  const startOffset = parsePositiveInt(process.env.EMBED_START_OFFSET, 0);
  const upsertBatchSize = parsePositiveInt(process.env.QDRANT_UPSERT_BATCH_SIZE, 4);
  const concurrency = parsePositiveInt(process.env.EMBED_CONCURRENCY, 1);
  const skipExisting = process.env.EMBED_SKIP_EXISTING !== "false";
  const files = batchFiles();
  let seen = 0;
  let processed = 0;
  let skipped = 0;
  let points = [];
  let collectionReady = false;
  let pending = [];

  console.log(`Embedding model: ${embeddingModel}`);
  console.log(`Qdrant collection: ${qdrantCollection}`);
  console.log(`Named vectors: ${vectorNames.join(", ")}`);
  console.log(`Batch files found: ${files.length}`);
  console.log(`Start offset: ${startOffset}`);
  console.log(`Limit: ${limit === Number.MAX_SAFE_INTEGER ? "all" : limit}`);
  console.log(`Embedding concurrency: ${concurrency}`);
  console.log(`Skip existing Qdrant points: ${skipExisting}`);

  async function flushPending() {
    if (pending.length === 0) {
      return;
    }

    const chunk = pending;
    pending = [];

    if (!collectionReady) {
      await ensureCollection();
      collectionReady = true;
    }

    const missingChunk = skipExisting
      ? chunk.filter(({ semanticObject }) => true)
      : chunk;

    let workChunk = missingChunk;
    if (skipExisting) {
      const existingIds = await existingPointIds(
        chunk.map(({ semanticObject }) => qdrantPointId(semanticObject.candidate_id))
      );
      workChunk = chunk.filter(({ semanticObject }) => {
        const exists = existingIds.has(qdrantPointId(semanticObject.candidate_id));
        if (exists) {
          skipped += 1;
        }
        return !exists;
      });
    }

    if (workChunk.length === 0) {
      console.log(`Skipped ${skipped} existing candidates`);
      return;
    }

    const builtPoints = await Promise.all(workChunk.map(({ semanticObject, file }) => buildPoint(semanticObject, file)));
    points.push(...builtPoints);
    processed += builtPoints.length;

    if (points.length >= upsertBatchSize) {
      await upsertPoints(points);
      console.log(`Upserted ${processed} candidates${skipped ? `, skipped ${skipped} existing` : ""}`);
      points = [];
    }
  }

  for (const file of files) {
    const semanticObjects = JSON.parse(readFileSync(file, "utf8"));

    for (const semanticObject of semanticObjects) {
      if (processed + pending.length >= limit) {
        break;
      }

      if (seen < startOffset) {
        seen += 1;
        continue;
      }

      pending.push({ semanticObject, file });
      seen += 1;

      if (pending.length >= concurrency) {
        await flushPending();
      }
    }

    if (processed + pending.length >= limit) {
      break;
    }
  }

  await flushPending();

  if (points.length > 0 && !collectionReady) {
    await ensureCollection();
    collectionReady = true;
  }

  await upsertPoints(points);
  console.log(
    `Done. Upserted ${processed} candidate points into ${qdrantCollection}.${skipped ? ` Skipped ${skipped} existing points.` : ""}`
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  embedBatches().catch((error) => {
    console.error("Embedding failed:", error.message);
    process.exit(1);
  });
}

export {
  buildPoint,
  embedText,
  embedBatches,
  qdrantPointId,
  qdrantRequest,
  semanticDocuments,
  vectorNames,
};
