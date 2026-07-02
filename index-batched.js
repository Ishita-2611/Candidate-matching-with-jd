import OpenAI from "openai";
import {
  appendFileSync,
  createReadStream,
  mkdirSync,
  writeFileSync,
  readFileSync,
  existsSync,
} from "node:fs";
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
const llmBatchSize = parsePositiveInt(
  process.env.LLM_BATCH_SIZE,
  5
);

if (!apiKey) {
  throw new Error("GROQ_API_KEY is required for the LLM semantic conversion demo.");
}

const openai = new OpenAI({
  apiKey,
  baseURL: groqBaseUrl,
});

function buildBatchPrompt(candidates) {
  return `
You are a technical recruiter assistant. Convert the candidate profiles below into compact semantic candidate objects.
Use factual, information-dense values inferred only from each candidate JSON.
Return valid JSON only. Do not include markdown fences, comments, trailing commas, or explanatory text.

You will receive multiple candidate profiles in a JSON array.

Rules:
- Produce exactly one semantic object for each candidate.
- Preserve candidate_id for every object.
- Return ONLY a JSON ARRAY (no surrounding object, no markdown fences).
- The array length MUST equal the input length.
- Do not merge candidates.
- Do not skip candidates.
- Do not change the schema of any object.

Candidate JSON Array:

${JSON.stringify(candidates)}

Produce ONLY a JSON ARRAY where each element corresponds to exactly one input candidate, in this schema:

[
  {
    "candidate_id": "<same candidate_id from the matching input candidate>",
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
  },
  {
    ...
  }
]
`.trim();
}

async function convertCandidatesToSemanticObjects(candidates) {
   
  const response = await openai.chat.completions.create({
    model,
    messages: [{ role: "user", content: buildBatchPrompt(candidates) }],
    temperature: 0.2,
    max_completion_tokens: 7000,
  });

  const raw = (response.choices[0]?.message?.content || "").trim();
  const cleaned = raw.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "").trim();

  let parsed;
  try {
    parsed = JSON.parse(cleaned);
  } catch {
    const err = new Error(`Model returned non-JSON output:\n${raw}`);
    err.isModelOutputError = true;
    throw err;
  }

  if (!Array.isArray(parsed)) {
    const err = new Error(`Model response was not a JSON array:\n${raw}`);
    err.isModelOutputError = true;
    throw err;
  }

  if (parsed.length !== candidates.length) {
    const err = new Error(
      `Expected ${candidates.length} objects but received ${parsed.length}`
    );
    err.isModelOutputError = true;
    throw err;
  }

  for (const obj of parsed) {
    for (const key of ["candidate_id", "metadata", "semantic_axes"]) {
      if (!obj || !obj[key]) {
        const err = new Error(`Missing or invalid key in response: "${key}"`);
        err.isModelOutputError = true;
        throw err;
      }
    }
  }

  const inputIds = new Set(candidates.map((c) => c.candidate_id));
  const outputIds = new Set(parsed.map((obj) => obj.candidate_id));

  if (
    inputIds.size !== outputIds.size ||
    [...inputIds].some((id) => !outputIds.has(id))
  ) {
    const err = new Error(
      `Returned candidate_id set does not match input candidate_id set`
    );
    err.isModelOutputError = true;
    throw err;
  }

  return parsed;
}

const convertCandidateToSemanticText = convertCandidatesToSemanticObjects;

async function convertBatch(candidates, { concurrency = 1, onProgress } = {}) {
  const results = [];
  const errors = [];

  for (let i = 0; i < candidates.length; i += concurrency) {
    const chunk = candidates.slice(i, i + concurrency);
    const settled = await Promise.allSettled(
      chunk.map((candidate) => convertCandidatesToSemanticObjects([candidate]))
    );

    settled.forEach((outcome, index) => {
      const candidate = chunk[index];
      if (outcome.status === "fulfilled") {
        results.push(...outcome.value);
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

async function convertWithRetry(candidates, attempts = 4) {
  let lastError;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await convertCandidatesToSemanticObjects(candidates);
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

async function convertCandidatesWithFallback(candidates) {
  try {
    const results = await convertWithRetry(candidates);
    return { results, errors: [] };
  } catch (error) {
    if (candidates.length === 1 || !error.isModelOutputError) {
      return {
        results: [],
        errors: candidates.map((candidate) => ({
          candidate_id: candidate.candidate_id,
          error: error.message,
        })),
      };
    }

    const mid = Math.ceil(candidates.length / 2);
    const left = candidates.slice(0, mid);
    const right = candidates.slice(mid);

    const [leftOutcome, rightOutcome] = await Promise.all([
      convertCandidatesWithFallback(left),
      convertCandidatesWithFallback(right),
    ]);

    return {
      results: [...leftOutcome.results, ...rightOutcome.results],
      errors: [...leftOutcome.errors, ...rightOutcome.errors],
    };
  }
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
  
    let existing = [];
  
    if (existsSync(file)) {
      try {
        existing = JSON.parse(readFileSync(file, "utf8"));
      } catch (error) {
        console.warn(`Could not parse existing batch ${file}. Starting fresh.`);
        existing = [];
      }
    }
  
    // Map existing candidates by candidate_id
    const mergedMap = new Map(
      existing.map((candidate) => [candidate.candidate_id, candidate])
    );
  
    // Replace existing candidate if present, otherwise add new one
    for (const candidate of currentBatch) {
      mergedMap.set(candidate.candidate_id, candidate);
    }
  
    // Convert back to array and sort by candidate_id
    const merged = [...mergedMap.values()].sort((a, b) => {
      const aId = Number(a.candidate_id);
      const bId = Number(b.candidate_id);
  
      if (!Number.isNaN(aId) && !Number.isNaN(bId)) {
        return aId - bId;
      }
  
      return String(a.candidate_id).localeCompare(String(b.candidate_id));
    });
  
    writeFileSync(file, `${JSON.stringify(merged, null, 2)}\n`);
  
    manifest.push({
      batch_no: currentBatchNo,
      file,
      count: merged.length,
      first_candidate_id: merged[0]?.candidate_id,
      last_candidate_id: merged.at(-1)?.candidate_id,
    });
  
    console.log(
      `Updated ${file} (+${currentBatch.length} processed, total ${merged.length} unique candidates)`
    );
  
    currentBatch = [];
    currentBatchNo += 1;
  }

  async function flush() {
    const chunk = pending;
    pending = [];

    const llmBatches = [];
    for (let i = 0; i < chunk.length; i += llmBatchSize) {
      llmBatches.push(chunk.slice(i, i + llmBatchSize));
    }

    const settled = await Promise.allSettled(
      llmBatches.map((batch) => convertCandidatesWithFallback(batch))
    );

    for (const [index, outcome] of settled.entries()) {
      const batch = llmBatches[index];

      if (outcome.status === "fulfilled") {
        const { results, errors } = outcome.value;

        processed += results.length + errors.length;
        succeeded += results.length;
        failed += errors.length;

        if (results.length > 0) {
          currentBatch.push(...results);
          writeBatchIfReady();
        }

        for (const err of errors) {
          appendFileSync(
            errorJsonl,
            `${JSON.stringify({
              candidate_id: err.candidate_id,
              error: err.error,
            })}\n`
          );
        }
      } else {
        processed += batch.length;
        failed += batch.length;
        for (const candidate of batch) {
          appendFileSync(
            errorJsonl,
            `${JSON.stringify({
              candidate_id: candidate.candidate_id,
              error: outcome.reason?.message || String(outcome.reason),
            })}\n`
          );
        }
      }

      if (processed % 25 < llmBatchSize || processed === limit) {
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

    if (pending.length >= concurrency * llmBatchSize) {
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
    const batchSize = parsePositiveInt(process.env.OUTPUT_BATCH_SIZE, 1000);
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
  buildBatchPrompt,
  convertBatch,
  convertCandidatesToSemanticObjects,
  convertCandidateToSemanticText,
  convertJsonlFile,
};