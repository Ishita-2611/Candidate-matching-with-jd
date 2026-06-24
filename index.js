import OpenAI from "openai";
import { appendFileSync, createReadStream, mkdirSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import readline from "node:readline";

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
  throw new Error("GROQ_API_KEY is required for the LLM semantic conversion demo.");
}

const openai = new OpenAI({
  apiKey,
  baseURL: groqBaseUrl,
});

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

async function convertCandidateToSemanticObject(candidate) {
  const response = await openai.chat.completions.create({
    model,
    messages: [{ role: "user", content: buildPrompt(candidate) }],
    temperature: 0.2,
    max_completion_tokens: 4096,
  });

  const raw = (response.choices[0]?.message?.content || "").trim();
  const cleaned = raw.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "").trim();

  let parsed;
  try {
    parsed = JSON.parse(cleaned);
  } catch {
    throw new Error(`Model returned non-JSON output:\n${raw}`);
  }

  for (const key of ["candidate_id", "metadata", "semantic_axes"]) {
    if (!parsed[key]) {
      throw new Error(`Missing or invalid key in response: "${key}"`);
    }
  }

  return parsed;
}

const convertCandidateToSemanticText = convertCandidateToSemanticObject;

async function convertBatch(candidates, { concurrency = 1, onProgress } = {}) {
  const results = [];
  const errors = [];

  for (let i = 0; i < candidates.length; i += concurrency) {
    const chunk = candidates.slice(i, i + concurrency);
    const settled = await Promise.allSettled(chunk.map((candidate) => convertCandidateToSemanticObject(candidate)));

    settled.forEach((outcome, index) => {
      const candidate = chunk[index];
      if (outcome.status === "fulfilled") {
        results.push(outcome.value);
      } else {
        errors.push({ candidate_id: candidate.candidate_id, error: outcome.reason.message });
        console.error(`[ERROR] ${candidate.candidate_id}: ${outcome.reason.message}`);
      }
    });

    onProgress?.({ done: Math.min(i + concurrency, candidates.length), total: candidates.length });
  }

  return { results, errors };
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

async function convertJsonlFile(
  filePath,
  { limit, startOffset, concurrency, batchSize, outputDir, manifestJson, errorJsonl }
) {
  const manifest = [];
  let processed = 0;
  let skipped = 0;
  let succeeded = 0;
  let failed = 0;
  let currentBatchNo = Math.floor(startOffset / batchSize) + 1;
  let currentBatch = [];
  let pending = [];

  mkdirSync(outputDir, { recursive: true });
  writeFileSync(errorJsonl, "");

  function writeBatchIfReady(force = false) {
    if (currentBatch.length === 0 || (!force && currentBatch.length < batchSize)) {
      return;
    }

    const file = batchFileName(outputDir, currentBatchNo);
    writeFileSync(file, `${JSON.stringify(currentBatch, null, 2)}\n`);

    manifest.push({
      batch_no: currentBatchNo,
      file,
      count: currentBatch.length,
      first_candidate_id: currentBatch[0]?.candidate_id,
      last_candidate_id: currentBatch.at(-1)?.candidate_id,
    });

    console.log(`Wrote ${file} (${currentBatch.length} semantic objects)`);
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

    if (skipped < startOffset) {
      skipped += 1;
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

  return { processed, skipped, succeeded, failed, batches: manifest.length };
}

async function main() {
  try {
    const inputJsonl = process.env.INPUT_JSONL || "candidates.jsonl";
    const limit = parsePositiveInt(process.env.BATCH_LIMIT, 5);
    const startOffset = parsePositiveInt(process.env.START_OFFSET, 0);
    const concurrency = parsePositiveInt(process.env.CONCURRENCY, 1);
    const batchSize = parsePositiveInt(process.env.OUTPUT_BATCH_SIZE, 5);
    const outputDir = process.env.OUTPUT_DIR || "batches";
    const manifestJson = process.env.MANIFEST_JSON || `${outputDir}/manifest.json`;
    const errorJsonl = process.env.ERROR_JSONL || `${outputDir}/output.errors.jsonl`;

    console.log("Starting LLM semantic object generation...\n");
    console.log(`Input JSONL: ${inputJsonl}`);
    console.log(`Model: ${model}`);
    console.log(`Groq base URL: ${groqBaseUrl}`);
    console.log(`Limit: ${limit}`);
    console.log(`Start offset: ${startOffset}`);
    console.log(`Concurrency: ${concurrency}`);
    console.log(`Output batch size: ${batchSize}`);
    console.log(`Output dir: ${outputDir}`);

    const summary = await convertJsonlFile(inputJsonl, {
      limit,
      startOffset,
      concurrency,
      batchSize,
      outputDir,
      manifestJson,
      errorJsonl,
    });

    console.log(`\nSaved ${summary.succeeded} semantic objects across ${summary.batches} batch files`);
    console.log(`Manifest: ${manifestJson}`);
    console.log(`Errors JSONL: ${errorJsonl}`);
  } catch (error) {
    console.error("Error during conversion:", error.message);
    process.exit(1);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}

export {
  buildPrompt,
  convertBatch,
  convertCandidateToSemanticObject,
  convertCandidateToSemanticText,
  convertJsonlFile,
};
