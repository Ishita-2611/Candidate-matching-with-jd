import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import {
  buildVectorRecords,
  embeddingModelName,
  ensureQdrantCollection,
  upsertQdrantVectors,
} from "./index.js";

function parsePositiveInt(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

const inputJson = process.env.INPUT_JSON || "output.json";
const vectorDbDir = process.env.VECTOR_DB_DIR || "vector_db";
const vectorIndexJsonl = process.env.VECTOR_INDEX_JSONL || `${vectorDbDir}/index.jsonl`;
const vectorBatchJson = process.env.VECTOR_BATCH_JSON || `${vectorDbDir}/vectors_batch0001.json`;
const qdrantUrl = process.env.QDRANT_URL || "http://localhost:6333";
const qdrantCollection = process.env.QDRANT_COLLECTION || "candidate_embeddings_bge_m3";
const embeddingDimension = parsePositiveInt(process.env.EMBEDDING_DIM, 1024);
const limit = parsePositiveInt(process.env.EMBED_LIMIT, 100);
const useQdrant = process.env.USE_QDRANT !== "false";

const semanticObjects = JSON.parse(readFileSync(inputJson, "utf8"))
  .slice(0, limit)
  .map((record) => record.semantic_object || record);

mkdirSync(vectorDbDir, { recursive: true });

if (useQdrant) {
  await ensureQdrantCollection({
    qdrantUrl,
    collectionName: qdrantCollection,
    dimension: embeddingDimension,
  });
}

const vectorRecords = [];

for (const semanticObject of semanticObjects) {
  vectorRecords.push(
    ...(await buildVectorRecords(semanticObject, {
      dimension: embeddingDimension,
      batchNo: 1,
    }))
  );
}

writeFileSync(vectorBatchJson, `${JSON.stringify(vectorRecords, null, 2)}\n`);
writeFileSync(vectorIndexJsonl, "");
appendFileSync(vectorIndexJsonl, `${vectorRecords.map((record) => JSON.stringify(record)).join("\n")}\n`);
writeFileSync(
  `${vectorDbDir}/config.json`,
  `${JSON.stringify(
    {
      embedding_model: embeddingModelName(),
      embedding_dimension: embeddingDimension,
      distance: "cosine",
      source_file: inputJson,
      source_object_count: semanticObjects.length,
      vector_count: vectorRecords.length,
      qdrant: useQdrant
        ? {
            url: qdrantUrl,
            collection: qdrantCollection,
          }
        : undefined,
    },
    null,
    2
  )}\n`
);

if (useQdrant) {
  await upsertQdrantVectors({
    qdrantUrl,
    collectionName: qdrantCollection,
    vectorRecords,
  });
}

console.log(
  JSON.stringify(
    {
      input_file: inputJson,
      objects: semanticObjects.length,
      vectors: vectorRecords.length,
      embedding_model: embeddingModelName(),
      embedding_dimension: embeddingDimension,
      qdrant_collection: useQdrant ? qdrantCollection : null,
    },
    null,
    2
  )
);
