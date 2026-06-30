import { readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

import { embedText, vectorNames } from "./embed-batches-qdrant.js";
import { queryDocumentForVector } from "./search-qdrant.js";

try {
  process.loadEnvFile?.();
} catch (error) {
  if (error.code !== "ENOENT") {
    throw error;
  }
}

const embeddingModel = process.env.EMBEDDING_MODEL || "Xenova/bge-m3";
const embeddingDimension = Number.parseInt(process.env.EMBEDDING_DIM || "1024", 10);

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, "utf8"));
}

async function buildJdEmbeddings(jdSemantic) {
  const vectors = {};
  const documents = {};

  for (const vectorName of vectorNames) {
    const text = queryDocumentForVector(jdSemantic, vectorName);
    documents[vectorName] = text;
    vectors[vectorName] = await embedText(text);
  }

  return {
    embedding_model: embeddingModel,
    embedding_dimension: embeddingDimension,
    metadata: jdSemantic.metadata ?? {},
    semantic_axes: jdSemantic.semantic_axes ?? {},
    semantic_documents: documents,
    vectors,
  };
}

async function main() {
  try {
    const inputFile = process.env.JD_SEMANTIC_FILE || "jd-semantic.json";
    const outputFile = process.env.JD_EMBEDDINGS_FILE || "jd-embeddings.json";
    const includeDocuments = process.env.JD_INCLUDE_DOCUMENTS !== "false";
    const includeFullVectors = process.env.JD_INCLUDE_FULL_VECTORS !== "false";

    console.log("Starting semantic JD embedding generation...\n");
    console.log(`Input semantic JD: ${inputFile}`);
    console.log(`Embedding model: ${embeddingModel}`);
    console.log(`Output file: ${outputFile}`);

    const jdSemantic = readJson(inputFile);
    const embeddings = await buildJdEmbeddings(jdSemantic);

    const output = {
      embedding_model: embeddings.embedding_model,
      embedding_dimension: embeddings.embedding_dimension,
      metadata: embeddings.metadata,
      semantic_axes: embeddings.semantic_axes,
      ...(includeDocuments ? { semantic_documents: embeddings.semantic_documents } : {}),
      vectors: includeFullVectors
        ? embeddings.vectors
        : Object.fromEntries(Object.entries(embeddings.vectors).map(([name, vector]) => [name, vector.slice(0, 12)])),
    };

    writeFileSync(outputFile, `${JSON.stringify(output, null, 2)}\n`);
    console.log(`\nSaved JD embeddings to ${outputFile}`);
  } catch (error) {
    console.error("JD embedding failed:", error.message);
    process.exit(1);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}

export {
  buildJdEmbeddings,
};
