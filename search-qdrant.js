import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

import { embedText, qdrantRequest, vectorNames } from "./embed-batches-qdrant.js";
import { buildQdrantHardFilter, hardConstraintsFromJd } from "./hard-filters.js";

try {
  process.loadEnvFile?.();
} catch (error) {
  if (error.code !== "ENOENT") {
    throw error;
  }
}

const qdrantCollection = process.env.QDRANT_COLLECTION || "candidate_semantic_multivectors_bge_m3";

function parsePositiveInt(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function stableJson(value) {
  return JSON.stringify(value ?? {});
}

function readJsonFile(filePath) {
  return JSON.parse(readFileSync(filePath, "utf8"));
}

function queryDocumentForVector(jdSemantic, vectorName) {
  const axes = jdSemantic.semantic_axes ?? {};

  const documents = {
    identity: stableJson(axes.identity),
    skills: stableJson(axes.skills),
    experience_summary: stableJson({
      summary: axes.experience_summary ?? {},
      chunks: axes.experience_chunks ?? [],
    }),
    domain: stableJson(axes.domain),
    execution_style: stableJson(axes.execution_style),
    trust_signals: stableJson(axes.trust_signals),
    default: [
      `metadata: ${stableJson(jdSemantic.metadata)}`,
      `identity: ${stableJson(axes.identity)}`,
      `skills: ${stableJson(axes.skills)}`,
      `experience_summary: ${stableJson(axes.experience_summary)}`,
      `experience_chunks: ${stableJson(axes.experience_chunks)}`,
      `domain: ${stableJson(axes.domain)}`,
      `execution_style: ${stableJson(axes.execution_style)}`,
      `trust_signals: ${stableJson(axes.trust_signals)}`,
    ].join("\n"),
  };

  return documents[vectorName] ?? documents.default;
}

async function countCandidates(filter) {
  const response = await qdrantRequest(`/collections/${qdrantCollection}/points/count`, {
    method: "POST",
    body: {
      exact: true,
      ...(filter ? { filter } : {}),
    },
  });

  return response.result?.count ?? 0;
}

async function ensurePayloadIndexes() {
  const indexes = [
    ["metadata.years_of_experience", "float"],
    ["metadata.location", "keyword"],
    ["metadata.preferred_work_mode", "keyword"],
  ];

  for (const [fieldName, fieldSchema] of indexes) {
    await qdrantRequest(`/collections/${qdrantCollection}/index?wait=true`, {
      method: "PUT",
      body: {
        field_name: fieldName,
        field_schema: fieldSchema,
      },
    });
  }
}

async function hybridSearch(jdSemantic, options = {}) {
  const vectorName = options.vectorName || process.env.SEARCH_VECTOR || "default";
  const limit = options.limit || parsePositiveInt(process.env.SEARCH_LIMIT, 10);

  if (!vectorNames.includes(vectorName)) {
    throw new Error(`SEARCH_VECTOR must be one of: ${vectorNames.join(", ")}`);
  }

  const constraints = options.constraints || hardConstraintsFromJd(jdSemantic);
  const filter = buildQdrantHardFilter(constraints);
  await ensurePayloadIndexes();
  const filteredCount = await countCandidates(filter);
  const queryVector = await embedText(queryDocumentForVector(jdSemantic, vectorName));

  const response = await qdrantRequest(`/collections/${qdrantCollection}/points/search`, {
    method: "POST",
    body: {
      vector: {
        name: vectorName,
        vector: queryVector,
      },
      filter,
      limit,
      with_payload: true,
      with_vector: false,
    },
  });

  return {
    collection: qdrantCollection,
    vectorName,
    constraints,
    filteredCount,
    results: response.result ?? [],
  };
}

function resultSummary(point) {
  const payload = point.payload ?? {};
  const metadata = payload.metadata ?? {};

  return {
    score: point.score,
    candidate_id: payload.candidate_id,
    years_of_experience: metadata.years_of_experience,
    location: metadata.location,
    preferred_work_mode: metadata.preferred_work_mode,
    role_family: payload.semantic_axes?.identity?.role_family,
    core_skills: payload.semantic_axes?.skills?.core_production_skills ?? [],
  };
}

async function main() {
  try {
    const jdSemanticFile = process.env.JD_SEMANTIC_FILE || "jd-semantic.json";
    const outputMode = process.env.SEARCH_OUTPUT || "summary";
    const jdSemantic = readJsonFile(jdSemanticFile);
    const searchResult = await hybridSearch(jdSemantic);

    console.log(`Collection: ${searchResult.collection}`);
    console.log(`Vector: ${searchResult.vectorName}`);
    console.log(`Hard constraints: ${JSON.stringify(searchResult.constraints)}`);
    console.log(`Candidates after hard filter: ${searchResult.filteredCount}`);
    console.log("");

    const output =
      outputMode === "raw"
        ? searchResult
        : {
            ...searchResult,
            results: searchResult.results.map(resultSummary),
          };

    console.log(JSON.stringify(output, null, 2));
  } catch (error) {
    console.error("Hybrid search failed:", error.message);
    process.exit(1);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}

export {
  hardConstraintsFromJd,
  hybridSearch,
  queryDocumentForVector,
};
