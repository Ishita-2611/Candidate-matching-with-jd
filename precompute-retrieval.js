import { writeFileSync, readFileSync } from "node:fs";

try {
  process.loadEnvFile?.();
} catch (error) {
  if (error.code !== "ENOENT") {
    throw error;
  }
}

function parsePositiveInt(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

async function qdrantRequest(path, body) {
  const qdrantUrl = (process.env.QDRANT_URL || "").replace(/\/$/, "");
  const apiKey = process.env.QDRANT_API_KEY || "";
  if (!qdrantUrl) {
    throw new Error("Set QDRANT_URL before precomputing retrieval candidates.");
  }
  const response = await fetch(`${qdrantUrl}${path}`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(apiKey ? { "api-key": apiKey } : {}),
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`Qdrant ${response.status}: ${await response.text()}`);
  }
  return response.json();
}

async function main() {
  const jdEmbeddingsFile = process.env.JD_EMBEDDINGS_FILE || "jd-embeddings.json";
  const outputFile = process.env.RETRIEVAL_OUTPUT || "retrieval_candidates.json";
  const collection = process.env.QDRANT_COLLECTION || "candidate_semantic_multivectors_bge_m3";
  const vectorName = process.env.RETRIEVAL_VECTOR_NAME || "default";
  const limit = parsePositiveInt(process.env.RETRIEVAL_LIMIT, 20000);
  const jdEmbeddings = JSON.parse(readFileSync(jdEmbeddingsFile, "utf8"));
  const vector = jdEmbeddings?.vectors?.[vectorName];
  if (!Array.isArray(vector)) {
    throw new Error(`Missing vector '${vectorName}' in ${jdEmbeddingsFile}`);
  }

  const result = await qdrantRequest(`/collections/${collection}/points/query`, {
    query: vector,
    using: vectorName,
    limit,
    with_payload: ["candidate_id"],
    with_vector: false,
  });

  const seen = new Set();
  const candidates = [];
  for (const point of result.result?.points || []) {
    const candidateId = point.payload?.candidate_id;
    if (!candidateId || seen.has(candidateId)) {
      continue;
    }
    seen.add(candidateId);
    candidates.push({
      candidate_id: candidateId,
      retrieval_score: Number(point.score || 0),
    });
  }

  writeFileSync(
    outputFile,
    `${JSON.stringify(
      {
        generated_at: new Date().toISOString(),
        source: "qdrant",
        collection,
        vector_name: vectorName,
        requested_limit: limit,
        candidates,
      },
      null,
      2
    )}\n`
  );

  console.log(JSON.stringify({ outputFile, candidates: candidates.length, top: candidates[0]?.candidate_id }, null, 2));
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
