import OpenAI from "openai";
import {
  appendFileSync,
  createReadStream,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { pathToFileURL } from "node:url";
import readline from "node:readline";

import {
  buildDefaultText,
  buildDomainText,
  buildExperienceChunks,
  buildExperienceSummaryText,
  buildIdentityText,
  buildSkillsText,
  buildTrustSignalsText,
} from "./generate-semantic-batches-v2.js";

try {
  process.loadEnvFile?.();
} catch (error) {
  if (error.code !== "ENOENT") {
    throw error;
  }
}

const apiKey = process.env.GROQ_API_KEY || "";
const model = process.env.GROQ_MODEL || "llama-3.1-8b-instant";
const groqBaseUrl = process.env.GROQ_BASE_URL || "https://api.groq.com/openai/v1";
const useGroq = process.env.EXECUTION_STYLE_PROVIDER !== "local" && Boolean(apiKey);
const openai = useGroq ? new OpenAI({ apiKey, baseURL: groqBaseUrl }) : null;

function parsePositiveInt(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function cleanText(value, maxLength = 3000) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  return text.length > maxLength ? `${text.slice(0, maxLength - 3)}...` : text;
}

function cacheFile(cacheDir, candidateId) {
  return `${cacheDir}/${candidateId}.json`;
}

function localExecutionStyle(candidate) {
  const text = cleanText(
    [
      candidate.profile?.summary,
      ...(candidate.career_history ?? []).map((job) => job.description),
    ].join(" "),
    5000
  ).toLowerCase();
  const signals = [];

  if (/(owned|led|architected|designed|shipped|deployed|launched|production)/i.test(text)) {
    signals.push("Shows ownership of production delivery and shipped systems.");
  }
  if (/(a\/b|experiment|metric|ndcg|mrr|precision|recall|conversion|ranking|recommendation)/i.test(text)) {
    signals.push("Works with product or ML evaluation metrics rather than only implementation tasks.");
  }
  if (/(pm|product manager|stakeholder|recruiter|customer|business)/i.test(text)) {
    signals.push("Has product and stakeholder collaboration signals.");
  }
  if (/(streaming|pipeline|scale|on-call|latency|deduplication|watermark|schema|reliability)/i.test(text)) {
    signals.push("Demonstrates infrastructure and reliability awareness.");
  }
  if (/(research|paper|prototype|notebook|kaggle)/i.test(text)) {
    signals.push("Has some research or exploratory ML orientation.");
  }

  return signals.join(" ") || "Execution style is unclear from the available profile text.";
}

function executionPrompt(candidate) {
  const profile = candidate.profile ?? {};
  const careerText = (candidate.career_history ?? [])
    .map(
      (job) =>
        `${job.title} at ${job.company}, ${job.duration_months} months, ${job.industry}: ${job.description}`
    )
    .join("\n\n");

  return `
Infer this candidate's execution style for a recruiting ranking system.
Use only the profile and career text. Return one dense paragraph, no JSON, no bullets.
Focus on shipping bias, ownership, product mindset, research/applied balance, ambiguity tolerance, and system design depth.

Profile:
${cleanText(profile.summary, 1200)}

Career:
${cleanText(careerText, 3500)}
`.trim();
}

async function groqExecutionStyle(candidate, cacheDir) {
  mkdirSync(cacheDir, { recursive: true });
  const file = cacheFile(cacheDir, candidate.candidate_id);
  if (existsSync(file)) {
    return JSON.parse(readFileSync(file, "utf8")).execution_style;
  }

  if (!useGroq) {
    const execution_style = localExecutionStyle(candidate);
    writeFileSync(file, `${JSON.stringify({ candidate_id: candidate.candidate_id, execution_style })}\n`);
    return execution_style;
  }

  const response = await openai.chat.completions.create({
    model,
    temperature: 0.1,
    max_completion_tokens: 256,
    messages: [{ role: "user", content: executionPrompt(candidate) }],
  });
  const execution_style = cleanText(response.choices[0]?.message?.content || localExecutionStyle(candidate), 1200);
  writeFileSync(file, `${JSON.stringify({ candidate_id: candidate.candidate_id, execution_style })}\n`);
  return execution_style;
}

async function buildSemanticObject(candidate, cacheDir) {
  const profile = candidate.profile ?? {};
  const signals = candidate.redrob_signals ?? {};
  const salary = signals.expected_salary_range_inr_lpa ?? {};
  const semanticObject = {
    candidate_id: candidate.candidate_id,
    metadata: {
      years_of_experience: profile.years_of_experience ?? 0,
      location: profile.location ?? "",
      country: profile.country ?? "",
      open_to_work: Boolean(signals.open_to_work_flag),
      preferred_work_mode: signals.preferred_work_mode ?? "",
      notice_period_days: signals.notice_period_days ?? 0,
      salary_range_lpa: { min: salary.min ?? 0, max: salary.max ?? 0 },
    },
    semantic_axes: {
      identity: buildIdentityText(candidate),
      skills: buildSkillsText(candidate),
      experience_summary: buildExperienceSummaryText(candidate),
      experience_chunks: buildExperienceChunks(candidate),
      domain: buildDomainText(candidate),
      execution_style: await groqExecutionStyle(candidate, cacheDir),
      trust_signals: buildTrustSignalsText(candidate),
    },
  };
  semanticObject.semantic_axes.default = buildDefaultText(semanticObject);
  return semanticObject;
}

function batchFileName(outputDir, batchNo) {
  return `${outputDir}/output_batch${String(batchNo).padStart(4, "0")}.json`;
}

async function mapConcurrent(items, concurrency, mapper) {
  const results = new Array(items.length);
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const index = next;
      next += 1;
      results[index] = await mapper(items[index], index);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
  return results;
}

async function generateSemanticBatches() {
  const inputJsonl = process.env.INPUT_JSONL || "candidates.jsonl";
  const limit = parsePositiveInt(process.env.BATCH_LIMIT, 100000);
  const startOffset = parsePositiveInt(process.env.START_OFFSET, 0);
  const batchSize = parsePositiveInt(process.env.OUTPUT_BATCH_SIZE, 1000);
  const concurrency = parsePositiveInt(process.env.CONCURRENCY, 50);
  const outputDir = process.env.OUTPUT_DIR || "batches";
  const manifestJson = process.env.MANIFEST_JSON || `${outputDir}/manifest.json`;
  const errorJsonl = process.env.ERROR_JSONL || `${outputDir}/output.errors.jsonl`;
  const cacheDir = process.env.EXECUTION_STYLE_CACHE_DIR || "execution_style_cache";
  const manifest = [];
  let skipped = 0;
  let processed = 0;
  let currentBatchNo = Math.floor(startOffset / batchSize) + 1;
  let pending = [];

  mkdirSync(outputDir, { recursive: true });
  mkdirSync(cacheDir, { recursive: true });
  writeFileSync(errorJsonl, "");

  async function flush(force = false) {
    if (pending.length === 0 || (!force && pending.length < batchSize)) {
      return;
    }
    const chunk = pending;
    pending = [];
    const semanticObjects = await mapConcurrent(chunk, concurrency, async (candidate) => {
      try {
        return await buildSemanticObject(candidate, cacheDir);
      } catch (error) {
        appendFileSync(errorJsonl, `${JSON.stringify({ candidate_id: candidate.candidate_id, error: error.message })}\n`);
        return null;
      }
    });
    const validObjects = semanticObjects.filter(Boolean);
    const file = batchFileName(outputDir, currentBatchNo);
    writeFileSync(file, `${JSON.stringify(validObjects, null, 2)}\n`);
    manifest.push({
      batch_no: currentBatchNo,
      file,
      count: validObjects.length,
      first_candidate_id: validObjects[0]?.candidate_id,
      last_candidate_id: validObjects.at(-1)?.candidate_id,
    });
    console.log(`Wrote ${file} (${validObjects.length} objects)`);
    currentBatchNo += 1;
  }

  const rl = readline.createInterface({
    input: createReadStream(inputJsonl, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (skipped < startOffset) {
      skipped += 1;
      continue;
    }
    pending.push(JSON.parse(trimmed));
    processed += 1;
    await flush(false);
    if (processed % 5000 === 0) console.log(`Progress ${processed}/${limit}`);
    if (processed >= limit) {
      rl.close();
      break;
    }
  }

  await flush(true);
  writeFileSync(manifestJson, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(`Done. ${processed} semantic objects across ${manifest.length} batch files.`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  generateSemanticBatches().catch((error) => {
    console.error("Failed:", error.message);
    process.exit(1);
  });
}

export { buildSemanticObject, generateSemanticBatches };
