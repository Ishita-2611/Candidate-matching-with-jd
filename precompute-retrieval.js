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

function normalize(vector) {
  const norm = Math.sqrt(vector.reduce((total, value) => total + Number(value) * Number(value), 0)) || 1;
  return vector.map((value) => Number(value) / norm);
}

function dot(left, right) {
  let total = 0;
  for (let index = 0; index < left.length; index += 1) {
    total += left[index] * Number(right[index] || 0);
  }
  return total;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
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

async function qdrantRequestWithRetry(path, body) {
  let lastError;
  for (let attempt = 1; attempt <= 8; attempt += 1) {
    try {
      return await qdrantRequest(path, body);
    } catch (error) {
      lastError = error;
      if (attempt === 8) {
        break;
      }
      await sleep(Math.min(30000, 1000 * 2 ** attempt));
    }
  }
  throw lastError;
}

async function queryTopCandidates({ collection, vectorName, vector, limit }) {
  const result = await qdrantRequestWithRetry(`/collections/${collection}/points/query`, {
    query: vector,
    using: vectorName,
    limit,
    with_payload: ["candidate_id"],
    with_vector: false,
  });

  return result.result?.points || [];
}

async function scrollAllCandidates({ collection, vectorName, vector, batchSize }) {
  const jdVector = normalize(vector);
  const candidates = [];
  let offset = null;

  while (true) {
    const body = {
      limit: batchSize,
      with_payload: ["candidate_id"],
      with_vector: [vectorName],
    };
    if (offset !== null) {
      body.offset = offset;
    }

    const result = await qdrantRequestWithRetry(`/collections/${collection}/points/scroll`, body);
    const points = result.result?.points || [];
    if (!points.length) {
      break;
    }

    for (const point of points) {
      const candidateId = point.payload?.candidate_id;
      const candidateVector = point.vector?.[vectorName];
      if (!candidateId || !Array.isArray(candidateVector)) {
        continue;
      }
      candidates.push({
        candidate_id: candidateId,
        retrieval_score: dot(jdVector, normalize(candidateVector)),
      });
    }

    if (candidates.length % 5000 === 0) {
      console.log(`Fetched ${candidates.length} vectors`);
    }

    offset = result.result?.next_page_offset;
    if (offset === null || offset === undefined) {
      break;
    }
  }

  candidates.sort((left, right) => right.retrieval_score - left.retrieval_score || left.candidate_id.localeCompare(right.candidate_id));
  return candidates;
}

async function main() {
  const jdEmbeddingsFile = process.env.JD_EMBEDDINGS_FILE || "jd-embeddings.json";
  const outputFile = process.env.RETRIEVAL_OUTPUT || "retrieval_candidates.json";
  const collection = process.env.QDRANT_COLLECTION || "candidate_semantic_multivectors_bge_m3";
  const vectorName = process.env.RETRIEVAL_VECTOR_NAME || "default";
  const limit = parsePositiveInt(process.env.RETRIEVAL_LIMIT, 20000);
  const batchSize = parsePositiveInt(process.env.RETRIEVAL_SCROLL_BATCH_SIZE, 512);
  const retrieveAll = process.env.RETRIEVAL_ALL === "true" || process.env.RETRIEVAL_LIMIT === "all";
  const jdEmbeddings = JSON.parse(readFileSync(jdEmbeddingsFile, "utf8"));
  const vector = jdEmbeddings?.vectors?.[vectorName];
  if (!Array.isArray(vector)) {
    throw new Error(`Missing vector '${vectorName}' in ${jdEmbeddingsFile}`);
  }

  const seen = new Set();
  const candidates = [];
  const points = retrieveAll
    ? await scrollAllCandidates({ collection, vectorName, vector, batchSize })
    : await queryTopCandidates({ collection, vectorName, vector, limit });

  for (const point of points) {
    const candidateId = point.candidate_id || point.payload?.candidate_id;
    if (!candidateId || seen.has(candidateId)) {
      continue;
    }
    seen.add(candidateId);
    candidates.push({
      candidate_id: candidateId,
      retrieval_score: Number(point.retrieval_score ?? point.score ?? 0),
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
        retrieval_all: retrieveAll,
        requested_limit: retrieveAll ? "all" : limit,
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
