import OpenAI from "openai";
import { createHash } from "node:crypto";
import { appendFileSync, createReadStream, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import readline from "node:readline";

try {
  process.loadEnvFile?.();
} catch (error) {
  if (error.code !== "ENOENT") {
    throw error;
  }
}

function readRawKeyFile(path) {
  if (!existsSync(path)) {
    return undefined;
  }

  const line = readFileSync(path, "utf8")
    .split(/\r?\n/)
    .map((value) => value.trim())
    .find((value) => value && !value.startsWith("#") && !value.includes("="));

  return line;
}

const apiKey =
  process.env.GROQ_API_KEY ||
  readRawKeyFile(".env") ||
  readRawKeyFile(".enc");
const model = process.env.GROQ_MODEL || "meta-llama/llama-4-scout-17b-16e-instruct";

if (!apiKey) {
  throw new Error("GROQ_API_KEY is required");
}

const openai = new OpenAI({
  apiKey,
  baseURL: "https://api.groq.com/openai/v1",
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

function hashToUInt32(value) {
  return createHash("sha256").update(value).digest().readUInt32BE(0);
}

function tokenize(text) {
  return String(text).toLowerCase().match(/[a-z0-9+#.-]+/g) || [];
}

function embedText(text, dimension) {
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

function axisText(semanticObject, axis) {
  const value = semanticObject.semantic_axes?.[axis];
  return JSON.stringify(value ?? {});
}

function buildVectorRecords(semanticObject, { dimension, batchNo }) {
  const axes = [
    "identity",
    "skills",
    "experience_summary",
    "experience_chunks",
    "domain",
    "execution_style",
    "trust_signals",
  ];

  return axes.map((axis) => {
    const text = axisText(semanticObject, axis);

    return {
      id: `${semanticObject.candidate_id}:${axis}`,
      candidate_id: semanticObject.candidate_id,
      axis,
      text,
      embedding_model: `local-hashing-${dimension}`,
      embedding_dimension: dimension,
      embedding: embedText(text, dimension),
      metadata: semanticObject.metadata,
      source_batch_no: batchNo,
    };
  });
}

async function convertJsonlFile(
  filePath,
  { limit, concurrency, batchSize, outputDir, manifestJson, errorJsonl, vectorDbDir, vectorIndexJsonl, embeddingDimension }
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
  writeFileSync(
    `${vectorDbDir}/config.json`,
    `${JSON.stringify(
      {
        embedding_model: `local-hashing-${embeddingDimension}`,
        embedding_dimension: embeddingDimension,
        distance: "cosine",
        record_shape: {
          id: "candidate_id:axis",
          candidate_id: "string",
          axis: "identity|skills|experience_summary|experience_chunks|domain|execution_style|trust_signals",
          text: "stringified semantic axis JSON",
          embedding: "normalized numeric vector",
          metadata: "candidate metadata",
        },
      },
      null,
      2
    )}\n`
  );

  function writeBatchIfReady(force = false) {
    if (currentBatch.length === 0 || (!force && currentBatch.length < batchSize)) {
      return;
    }

    const file = batchFileName(outputDir, currentBatchNo);
    const vectorFile = vectorFileName(vectorDbDir, currentBatchNo);
    const vectorRecords = currentBatch.flatMap((semanticObject) =>
      buildVectorRecords(semanticObject, {
        dimension: embeddingDimension,
        batchNo: currentBatchNo,
      })
    );

    writeFileSync(file, `${JSON.stringify(currentBatch, null, 2)}\n`);
    writeFileSync(vectorFile, `${JSON.stringify(vectorRecords, null, 2)}\n`);
    appendFileSync(vectorIndexJsonl, vectorRecords.map((record) => JSON.stringify(record)).join("\n") + "\n");
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

    settled.forEach((outcome, index) => {
      const candidate = chunk[index];
      processed += 1;

      if (outcome.status === "fulfilled") {
        succeeded += 1;
        currentBatch.push(outcome.value);
        writeBatchIfReady();
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
    });
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

  writeBatchIfReady(true);
  writeFileSync(manifestJson, `${JSON.stringify(manifest, null, 2)}\n`);

  return { processed, succeeded, failed, batches: manifest.length, vectorCount };
}

// Example usage

const exampleCandidate = {
  candidate_id: "CAND_0000001",
  metadata: {
    years_of_experience: 6.9,
    location: "Toronto",
    country: "Canada",
    open_to_work: true,
    preferred_work_mode: "onsite",
    notice_period_days: 60,
    salary_range_lpa: { min: 18.7, max: 36.1 },
  },
  semantic_axes: {
    identity: {
      role_family: "Data Infrastructure Engineer",
      secondary_roles: ["Backend Engineer", "ML Infrastructure Engineer", "Data Engineer"],
      seniority: "Mid-Senior",
      career_transition: { from: "Data Engineering", to: "Applied ML / AI Infrastructure" },
    },
    skills: {
      core_production_skills: ["Python", "SQL", "Spark", "Kafka", "Airflow", "Streaming Pipelines", "Data Warehousing"],
      ml_skills: ["NLP", "Fine-tuning LLMs", "LoRA", "Statistical Modeling", "Image Classification", "Speech Recognition", "TTS", "GANs"],
      ml_infra_skills: ["Milvus", "BentoML", "Weights & Biases"],
      weak_skills: ["AWS", "GCP", "Apache Beam"],
      noisy_non_relevant_skills: ["Tailwind", "Photoshop"],
    },
    experience_summary: {
      system_types: ["Batch Processing", "Streaming Systems", "Analytics Pipelines", "Feature Engineering Pipelines"],
      scale: { daily_data_processed: "500GB", source_systems: 12, realtime_systems: true },
      production_maturity: "high",
      ml_maturity: "moderate-low",
    },
    experience_chunks: [
      {
        id: "chunk_1",
        description: "Implemented streaming data pipelines on Kafka and Spark Streaming for a real-time user-activity processing platform.",
        tags: ["streaming", "kafka", "spark", "realtime"],
      },
      {
        id: "chunk_2",
        description: "Built and maintained data pipelines on Apache Airflow processing ~500GB of daily transactional data across 12 source systems using Spark and dbt in Snowflake.",
        tags: ["batch-processing", "airflow", "spark", "dbt", "snowflake", "data-warehouse"],
      },
    ],
    domain: {
      primary_domains: ["Data Infrastructure", "Analytics Engineering", "Enterprise Systems"],
      secondary_domains: ["Internal ML Systems", "Feature Platforms"],
      missing_domains: ["Search", "Ranking", "Recommendation Systems", "Marketplace"],
    },
    execution_style: {
      shipping_bias: "high",
      product_mindset: "medium",
      research_bias: "low-medium",
      ambiguity_tolerance: "medium",
      ownership: "high",
      system_design_depth: "high",
    },
    trust_signals: {
      github_score: 9.2,
      profile_completeness: 86.9,
      recruiter_response_rate: 0.34,
      interview_completion_rate: 0.71,
      offer_acceptance_rate: 0.58,
    },
  },
};

// Run
(async () => {
  try {
    const limit = parsePositiveInt(process.env.BATCH_LIMIT, 5000);
    const concurrency = parsePositiveInt(process.env.CONCURRENCY, 5);
    const batchSize = parsePositiveInt(process.env.OUTPUT_BATCH_SIZE, 100);
    const outputDir = process.env.OUTPUT_DIR || "batches";
    const manifestJson = process.env.MANIFEST_JSON || `${outputDir}/manifest.json`;
    const errorJsonl = process.env.ERROR_JSONL || `${outputDir}/output.errors.jsonl`;
    const vectorDbDir = process.env.VECTOR_DB_DIR || "vector_db";
    const vectorIndexJsonl = process.env.VECTOR_INDEX_JSONL || `${vectorDbDir}/index.jsonl`;
    const embeddingDimension = parsePositiveInt(process.env.EMBEDDING_DIM, 384);

    console.log("Starting semantic object generation...\n");
    console.log("Provider: groq");
    console.log(`Model: ${model}`);
    console.log(`Limit: ${limit}`);
    console.log(`Concurrency: ${concurrency}`);
    console.log(`Output batch size: ${batchSize}`);
    console.log(`Output dir: ${outputDir}`);
    console.log(`Vector DB dir: ${vectorDbDir}`);
    console.log(`Embedding dimension: ${embeddingDimension}`);
    console.log(`API Key configured: ${apiKey ? "Yes" : "No"}`);

    const summary = await convertJsonlFile("candidates.jsonl", {
      limit,
      concurrency,
      batchSize,
      outputDir,
      manifestJson,
      errorJsonl,
      vectorDbDir,
      vectorIndexJsonl,
      embeddingDimension,
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
})();

export {
  convertCandidateToSemanticText,
  convertCandidateToSemanticObject,
  convertBatch,
  convertJsonlFile,
  readFirstJsonlObjects,
};
