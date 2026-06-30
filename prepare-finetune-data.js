import { createReadStream, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { pathToFileURL } from "node:url";

try {
  process.loadEnvFile?.();
} catch (error) {
  if (error.code !== "ENOENT") {
    throw error;
  }
}

const systemPrompt = `You convert raw candidate JSON into compact semantic candidate JSON for candidate-job matching.
Use only facts from the input candidate. Preserve candidate_id exactly. Return valid JSON only, with no markdown.`;

const userPromptPrefix = `Convert this candidate profile into the exact semantic schema used for matching.
Return only one JSON object.

Candidate JSON:`;

function parsePositiveInt(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function parseRatio(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 && parsed < 1 ? parsed : fallback;
}

function candidateIdOf(value) {
  return value?.candidate_id ? String(value.candidate_id) : "";
}

function stableJson(value) {
  return JSON.stringify(value);
}

function semanticBatchFiles(dir) {
  return readdirSync(dir)
    .filter((name) => /^output_batch\d{4}\.json$/i.test(name))
    .sort()
    .map((name) => `${dir}/${name}`);
}

function loadSemanticTargets(dir, limit) {
  const targets = new Map();

  for (const file of semanticBatchFiles(dir)) {
    const objects = JSON.parse(readFileSync(file, "utf8"));

    for (const object of objects) {
      const candidateId = candidateIdOf(object);
      if (!candidateId || targets.has(candidateId)) {
        continue;
      }

      targets.set(candidateId, object);
      if (targets.size >= limit) {
        return targets;
      }
    }
  }

  return targets;
}

async function loadRawCandidates(candidateFile, targetIds) {
  const rawCandidates = new Map();
  const reader = createInterface({
    input: createReadStream(candidateFile, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });

  for await (const line of reader) {
    if (!line.trim()) {
      continue;
    }

    const candidate = JSON.parse(line);
    const candidateId = candidateIdOf(candidate);

    if (targetIds.has(candidateId)) {
      rawCandidates.set(candidateId, candidate);
      if (rawCandidates.size === targetIds.size) {
        reader.close();
        break;
      }
    }
  }

  return rawCandidates;
}

function seededShuffle(items, seedText) {
  let seed = 2166136261;
  for (const char of String(seedText)) {
    seed ^= char.charCodeAt(0);
    seed = Math.imul(seed, 16777619);
  }

  const shuffled = [...items];
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    seed = Math.imul(seed ^ (seed >>> 15), 2246822507);
    seed = Math.imul(seed ^ (seed >>> 13), 3266489909);
    const random = ((seed ^= seed >>> 16) >>> 0) / 4294967296;
    const swapIndex = Math.floor(random * (index + 1));
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }
  return shuffled;
}

function toChatExample(rawCandidate, semanticTarget) {
  return {
    messages: [
      {
        role: "system",
        content: systemPrompt,
      },
      {
        role: "user",
        content: `${userPromptPrefix}\n${stableJson(rawCandidate)}`,
      },
      {
        role: "assistant",
        content: stableJson(semanticTarget),
      },
    ],
  };
}

function writeJsonl(filePath, rows) {
  writeFileSync(filePath, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`);
}

async function prepareFineTuneData() {
  const candidatesFile = process.env.FINETUNE_CANDIDATES_FILE || "candidates.jsonl";
  const semanticDir = process.env.FINETUNE_SEMANTIC_DIR || "batches";
  const outputDir = process.env.FINETUNE_OUTPUT_DIR || "fine_tuning";
  const limit = parsePositiveInt(process.env.FINETUNE_LIMIT, 1000);
  const validationRatio = parseRatio(process.env.FINETUNE_VALIDATION_RATIO, 0.1);
  const seed = process.env.FINETUNE_SHUFFLE_SEED || "candidate-semantic-v1";

  mkdirSync(outputDir, { recursive: true });

  const semanticTargets = loadSemanticTargets(semanticDir, limit);
  const rawCandidates = await loadRawCandidates(candidatesFile, new Set(semanticTargets.keys()));
  const examples = [];
  const missingRawCandidates = [];

  for (const [candidateId, semanticTarget] of semanticTargets) {
    const rawCandidate = rawCandidates.get(candidateId);
    if (!rawCandidate) {
      missingRawCandidates.push(candidateId);
      continue;
    }
    examples.push(toChatExample(rawCandidate, semanticTarget));
  }

  const shuffled = seededShuffle(examples, seed);
  const validationCount = Math.max(1, Math.floor(shuffled.length * validationRatio));
  const validation = shuffled.slice(0, validationCount);
  const train = shuffled.slice(validationCount);
  const trainFile = `${outputDir}/semantic_train.jsonl`;
  const validationFile = `${outputDir}/semantic_validation.jsonl`;
  const manifestFile = `${outputDir}/manifest.json`;

  writeJsonl(trainFile, train);
  writeJsonl(validationFile, validation);
  writeFileSync(
    manifestFile,
    `${JSON.stringify(
      {
        candidates_file: candidatesFile,
        semantic_dir: semanticDir,
        limit,
        examples: examples.length,
        train_examples: train.length,
        validation_examples: validation.length,
        missing_raw_candidates: missingRawCandidates,
        output_files: {
          train: trainFile,
          validation: validationFile,
        },
        format: "OpenAI/OpenPipe/Fireworks chat fine-tuning JSONL",
      },
      null,
      2
    )}\n`
  );

  return {
    examples: examples.length,
    train: train.length,
    validation: validation.length,
    trainFile,
    validationFile,
    manifestFile,
  };
}

async function main() {
  try {
    const result = await prepareFineTuneData();
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    console.error("Fine-tune data preparation failed:", error.message);
    process.exit(1);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}

export {
  prepareFineTuneData,
  toChatExample,
};
