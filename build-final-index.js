import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { embedText, embeddingModelName } from "./index.js";

function parsePositiveInt(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

const inputJson = process.env.INPUT_JSON || "batches/output_batch0001.json";
const outputJson = process.env.FINAL_INDEX_JSON || "final_candidate_index.json";
const outputDir = process.env.FINAL_INDEX_DIR || "final_index";
const limit = parsePositiveInt(process.env.FINAL_INDEX_LIMIT, 1);
const embeddingDimension = parsePositiveInt(process.env.EMBEDDING_DIM, 1024);
const qdrantUrl = process.env.QDRANT_URL || "http://localhost:6333";
const useQdrant = process.env.USE_QDRANT !== "false";

const collectionMap = {
  identity: process.env.QDRANT_IDENTITY_COLLECTION || "candidate_identity",
  skills: process.env.QDRANT_SKILLS_COLLECTION || "candidate_skills",
  experience_summary: process.env.QDRANT_EXPERIENCE_COLLECTION || "candidate_experience",
  domain: process.env.QDRANT_DOMAIN_COLLECTION || "candidate_domain",
  execution_style: process.env.QDRANT_EXECUTION_COLLECTION || "candidate_execution",
  experience_chunks: process.env.QDRANT_CHUNKS_COLLECTION || "candidate_chunks",
};

function pointId(value) {
  const hex = createHash("sha256").update(value).digest("hex").slice(0, 32);
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

async function qdrantRequest(path, { method = "GET", body } = {}) {
  const response = await fetch(`${qdrantUrl}${path}`, {
    method,
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    throw new Error(`Qdrant ${method} ${path} failed: ${response.status} ${await response.text()}`);
  }

  return response.json();
}

async function ensureCollection(collectionName) {
  const existing = await fetch(`${qdrantUrl}/collections/${collectionName}`);

  if (existing.ok) {
    return;
  }

  await qdrantRequest(`/collections/${collectionName}`, {
    method: "PUT",
    body: {
      vectors: {
        size: embeddingDimension,
        distance: "Cosine",
      },
    },
  });
}

function denseDoc(value) {
  if (Array.isArray(value)) {
    return value
      .map((item) => item?.description || JSON.stringify(item))
      .filter(Boolean)
      .join("\n");
  }

  if (!value || typeof value !== "object") {
    return String(value ?? "");
  }

  return JSON.stringify(value);
}

function metadataFrom(semanticObject) {
  const metadata = semanticObject.metadata ?? {};
  const salary = metadata.salary_range_lpa ?? {};

  return {
    years_of_experience: metadata.years_of_experience ?? null,
    location: metadata.location ?? "",
    country: metadata.country ?? "",
    salary_min_lpa: salary.min ?? null,
    salary_max_lpa: salary.max ?? null,
    notice_period_days: metadata.notice_period_days ?? null,
    open_to_work: metadata.open_to_work ?? null,
    preferred_work_mode: metadata.preferred_work_mode ?? "",
    willing_to_relocate: metadata.willing_to_relocate ?? false,
  };
}

function semanticDocsFrom(semanticObject) {
  const axes = semanticObject.semantic_axes ?? {};

  return {
    identity: denseDoc(axes.identity),
    skills: denseDoc(axes.skills),
    experience_summary: denseDoc(axes.experience_summary),
    domain: denseDoc(axes.domain),
    execution_style: denseDoc(axes.execution_style),
    experience_chunks: (axes.experience_chunks ?? []).map((chunk) => chunk.description || denseDoc(chunk)),
  };
}

function trustFeaturesFrom(semanticObject) {
  const trust = semanticObject.semantic_axes?.trust_signals ?? {};

  return {
    github_score: trust.github_score ?? null,
    profile_completeness: trust.profile_completeness ?? null,
    interview_completion_rate: trust.interview_completion_rate ?? null,
    offer_acceptance_rate: trust.offer_acceptance_rate ?? null,
    recruiter_response_rate: trust.recruiter_response_rate ?? null,
    honeytrap_probability: trust.honeytrap_probability ?? null,
    credibility_score: trust.credibility_score ?? null,
  };
}

function chunkWeight(text) {
  const lower = text.toLowerCase();
  if (/(designed|architected|built|implemented|retrieval|embedding|scale|streaming|pipeline|production)/.test(lower)) {
    return 1.2;
  }

  return 1.0;
}

async function buildFinalCandidate(semanticObject) {
  const semanticDocs = semanticDocsFrom(semanticObject);
  const vectors = {};

  for (const key of ["identity", "skills", "experience_summary", "domain", "execution_style"]) {
    vectors[key] = {
      dimension: embeddingDimension,
      embedding: await embedText(semanticDocs[key], embeddingDimension),
    };
  }

  vectors.experience_chunks = [];

  for (const [index, text] of semanticDocs.experience_chunks.entries()) {
    vectors.experience_chunks.push({
      chunk_id: `exp_1_chunk_${index + 1}`,
      text,
      weight: chunkWeight(text),
      dimension: embeddingDimension,
      embedding: await embedText(text, embeddingDimension),
    });
  }

  return {
    candidate_id: semanticObject.candidate_id,
    metadata: metadataFrom(semanticObject),
    semantic_docs: semanticDocs,
    vectors,
    trust_features: trustFeaturesFrom(semanticObject),
    embedding_model: embeddingModelName(embeddingDimension),
  };
}

async function upsertFinalCandidate(candidate) {
  for (const [key, collectionName] of Object.entries(collectionMap)) {
    await ensureCollection(collectionName);

    if (key === "experience_chunks") {
      const points = candidate.vectors.experience_chunks.map((chunk) => ({
        id: pointId(`${candidate.candidate_id}:${chunk.chunk_id}`),
        vector: chunk.embedding,
        payload: {
          candidate_id: candidate.candidate_id,
          chunk_id: chunk.chunk_id,
          weight: chunk.weight,
          text: chunk.text,
          metadata: candidate.metadata,
          trust_features: candidate.trust_features,
          embedding_model: candidate.embedding_model,
        },
      }));

      if (points.length > 0) {
        await qdrantRequest(`/collections/${collectionName}/points?wait=true`, {
          method: "PUT",
          body: { points },
        });
      }

      continue;
    }

    await qdrantRequest(`/collections/${collectionName}/points?wait=true`, {
      method: "PUT",
      body: {
        points: [
          {
            id: pointId(`${candidate.candidate_id}:${key}`),
            vector: candidate.vectors[key].embedding,
            payload: {
              candidate_id: candidate.candidate_id,
              vector_type: key,
              text: candidate.semantic_docs[key],
              metadata: candidate.metadata,
              trust_features: candidate.trust_features,
              embedding_model: candidate.embedding_model,
            },
          },
        ],
      },
    });
  }
}

const semanticObjects = JSON.parse(readFileSync(inputJson, "utf8"))
  .slice(0, limit)
  .map((record) => record.semantic_object || record);

mkdirSync(outputDir, { recursive: true });

const finalCandidates = [];

for (const semanticObject of semanticObjects) {
  const candidate = await buildFinalCandidate(semanticObject);
  finalCandidates.push(candidate);
}

writeFileSync(outputJson, `${JSON.stringify(finalCandidates, null, 2)}\n`);

if (useQdrant) {
  for (const candidate of finalCandidates) {
    await upsertFinalCandidate(candidate);
  }
}

console.log(
  JSON.stringify(
    {
      input_file: inputJson,
      output_file: outputJson,
      candidates: finalCandidates.map((candidate) => candidate.candidate_id),
      candidate_count: finalCandidates.length,
      embedding_model: embeddingModelName(embeddingDimension),
      embedding_dimension: embeddingDimension,
      qdrant_collections: useQdrant ? collectionMap : null,
    },
    null,
    2
  )
);
