import OpenAI from "openai";
import { appendFileSync, createReadStream, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import readline from "node:readline";
import { pathToFileURL } from "node:url";

import { toChatExample } from "./prepare-finetune-data.js";

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

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function stripJsonFences(text) {
  return String(text || "")
    .trim()
    .replace(/^```(?:json)?/i, "")
    .replace(/```$/i, "")
    .trim();
}

function parseFirstJsonObject(text) {
  const cleaned = stripJsonFences(text);

  try {
    return JSON.parse(cleaned);
  } catch {
    let depth = 0;
    let start = -1;
    let inString = false;
    let escaped = false;

    for (let index = 0; index < cleaned.length; index += 1) {
      const char = cleaned[index];

      if (escaped) {
        escaped = false;
        continue;
      }
      if (char === "\\") {
        escaped = true;
        continue;
      }
      if (char === '"') {
        inString = !inString;
        continue;
      }
      if (inString) {
        continue;
      }
      if (char === "{") {
        if (depth === 0) {
          start = index;
        }
        depth += 1;
      } else if (char === "}") {
        depth -= 1;
        if (depth === 0 && start >= 0) {
          return JSON.parse(cleaned.slice(start, index + 1));
        }
      }
    }
  }

  return JSON.parse(cleaned);
}

function outputFileName(outputDir, batchNumber) {
  return `${outputDir}/output_batch${String(batchNumber).padStart(4, "0")}.json`;
}

function readExistingManifest(outputDir) {
  const manifestFile = `${outputDir}/manifest.json`;
  if (!existsSync(manifestFile)) {
    return [];
  }
  return JSON.parse(readFileSync(manifestFile, "utf8"));
}

function readExistingCandidateIds(outputDir) {
  const ids = new Set();

  if (!existsSync(outputDir)) {
    return ids;
  }

  for (const file of readExistingManifest(outputDir).map((entry) => entry.file)) {
    if (!existsSync(file)) {
      continue;
    }

    const objects = JSON.parse(readFileSync(file, "utf8"));
    for (const object of objects) {
      if (object?.candidate_id) {
        ids.add(object.candidate_id);
      }
    }
  }

  return ids;
}

function writeManifest(outputDir, entries) {
  writeFileSync(`${outputDir}/manifest.json`, `${JSON.stringify(entries, null, 2)}\n`);
}

function messagesForCandidate(candidate) {
  const emptyTarget = {};
  const example = toChatExample(candidate, emptyTarget);
  return example.messages.slice(0, 2);
}

function validateSemanticObject(candidate, semanticObject) {
  if (!semanticObject || typeof semanticObject !== "object" || Array.isArray(semanticObject)) {
    throw new Error("Model returned a non-object semantic result.");
  }
  if (semanticObject.candidate_id !== candidate.candidate_id) {
    throw new Error(`candidate_id mismatch: expected ${candidate.candidate_id}, got ${semanticObject.candidate_id}`);
  }
  if (!semanticObject.metadata || !semanticObject.semantic_axes) {
    throw new Error("Semantic result must include metadata and semantic_axes.");
  }
  return semanticObject;
}

async function semanticFromFineTunedModel(client, model, candidate, retries) {
  let lastError;

  for (let attempt = 1; attempt <= retries; attempt += 1) {
    try {
      const completion = await client.chat.completions.create({
        model,
        temperature: 0,
        ...(process.env.FINETUNED_RESPONSE_FORMAT === "json_object"
          ? { response_format: { type: "json_object" } }
          : {}),
        messages: messagesForCandidate(candidate),
      });
      const content = completion.choices?.[0]?.message?.content;
      return validateSemanticObject(candidate, parseFirstJsonObject(content));
    } catch (error) {
      lastError = error;
      if (attempt < retries) {
        await sleep(750 * attempt);
      }
    }
  }

  throw lastError;
}

async function mapWithConcurrency(items, concurrency, mapper) {
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

async function readCandidates({ inputFile, startOffset, limit }) {
  const candidates = [];
  const reader = readline.createInterface({
    input: createReadStream(inputFile, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });
  let seen = 0;

  for await (const line of reader) {
    if (!line.trim()) {
      continue;
    }
    if (seen < startOffset) {
      seen += 1;
      continue;
    }
    candidates.push(JSON.parse(line));
    seen += 1;
    if (candidates.length >= limit) {
      reader.close();
      break;
    }
  }

  return candidates;
}

async function generateSemanticWithFineTunedModel() {
  const apiKey = process.env.FINETUNED_API_KEY || process.env.OPENPIPE_API_KEY || process.env.OPENAI_API_KEY;
  const baseURL =
    process.env.FINETUNED_BASE_URL ||
    process.env.OPENPIPE_BASE_URL ||
    process.env.OPENAI_BASE_URL ||
    "https://app.openpipe.ai/api/v1";
  const model = process.env.FINETUNED_MODEL;

  if (!apiKey) {
    throw new Error("Set FINETUNED_API_KEY, OPENPIPE_API_KEY, or OPENAI_API_KEY.");
  }
  if (!model) {
    throw new Error("Set FINETUNED_MODEL to the deployed fine-tuned model id.");
  }

  const inputFile = process.env.FINETUNED_INPUT_JSONL || "candidates.jsonl";
  const outputDir = process.env.FINETUNED_OUTPUT_DIR || "batches_finetuned";
  const startOffset = parsePositiveInt(process.env.FINETUNED_START_OFFSET, 0);
  const limit = parsePositiveInt(process.env.FINETUNED_LIMIT, 1000);
  const outputBatchSize = parsePositiveInt(process.env.FINETUNED_OUTPUT_BATCH_SIZE, 100);
  const concurrency = parsePositiveInt(process.env.FINETUNED_CONCURRENCY, 3);
  const retries = parsePositiveInt(process.env.FINETUNED_RETRIES, 3);
  const skipExisting = process.env.FINETUNED_SKIP_EXISTING !== "false";

  mkdirSync(outputDir, { recursive: true });
  const client = new OpenAI({
    apiKey,
    ...(baseURL ? { baseURL } : {}),
  });
  const candidates = await readCandidates({ inputFile, startOffset, limit });
  const manifest = readExistingManifest(outputDir);
  const existingCandidateIds = skipExisting ? readExistingCandidateIds(outputDir) : new Set();
  let batch = [];
  let batchNumber = manifest.length + 1;
  let processed = 0;
  let skipped = 0;

  for (let index = 0; index < candidates.length; index += concurrency) {
    const chunk = candidates.slice(index, index + concurrency).filter((candidate) => {
      if (existingCandidateIds.has(candidate.candidate_id)) {
        skipped += 1;
        return false;
      }
      return true;
    });

    if (chunk.length === 0) {
      continue;
    }

    const results = await mapWithConcurrency(chunk, concurrency, async (candidate) => {
      try {
        return await semanticFromFineTunedModel(client, model, candidate, retries);
      } catch (error) {
        appendFileSync(
          `${outputDir}/output.errors.jsonl`,
          `${JSON.stringify({ candidate_id: candidate.candidate_id, error: error.message })}\n`
        );
        return null;
      }
    });

    for (const result of results) {
      if (!result) {
        continue;
      }
      existingCandidateIds.add(result.candidate_id);
      batch.push(result);
      processed += 1;

      if (batch.length >= outputBatchSize) {
        const file = outputFileName(outputDir, batchNumber);
        writeFileSync(file, `${JSON.stringify(batch, null, 2)}\n`);
        manifest.push({ file, count: batch.length, last_candidate_id: batch.at(-1)?.candidate_id });
        writeManifest(outputDir, manifest);
        console.log(`Saved ${file} (${batch.length} candidates)`);
        batch = [];
        batchNumber += 1;
      }
    }
  }

  if (batch.length) {
    const file = outputFileName(outputDir, batchNumber);
    writeFileSync(file, `${JSON.stringify(batch, null, 2)}\n`);
    manifest.push({ file, count: batch.length, last_candidate_id: batch.at(-1)?.candidate_id });
    writeManifest(outputDir, manifest);
    console.log(`Saved ${file} (${batch.length} candidates)`);
  }

  return {
    model,
    outputDir,
    requested: candidates.length,
    processed,
    skipped,
    errorsFile: `${outputDir}/output.errors.jsonl`,
  };
}

async function main() {
  try {
    const result = await generateSemanticWithFineTunedModel();
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    console.error("Fine-tuned semantic generation failed:", error.message);
    process.exit(1);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}

export {
  generateSemanticWithFineTunedModel,
  semanticFromFineTunedModel,
};
