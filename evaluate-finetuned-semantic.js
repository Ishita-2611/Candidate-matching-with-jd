import OpenAI from "openai";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

import { semanticFromFineTunedModel } from "./generate-semantic-finetuned.js";

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

function normalize(value) {
  return String(value ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9+#.]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizedSet(values) {
  return new Set((values ?? []).map(normalize).filter(Boolean));
}

function jaccard(aValues, bValues) {
  const a = normalizedSet(aValues);
  const b = normalizedSet(bValues);
  const union = new Set([...a, ...b]);
  if (union.size === 0) {
    return 1;
  }
  let intersection = 0;
  for (const item of a) {
    if (b.has(item)) {
      intersection += 1;
    }
  }
  return intersection / union.size;
}

function numberClose(a, b, tolerance = 0.01) {
  const left = Number(a);
  const right = Number(b);
  return Number.isFinite(left) && Number.isFinite(right) && Math.abs(left - right) <= tolerance;
}

function extractCandidateFromExample(example) {
  const userContent = example.messages?.find((message) => message.role === "user")?.content ?? "";
  const marker = "Candidate JSON:";
  const index = userContent.indexOf(marker);
  if (index < 0) {
    throw new Error("Fine-tune example user message does not contain Candidate JSON marker.");
  }
  return JSON.parse(userContent.slice(index + marker.length).trim());
}

function loadExamples(filePath, limit) {
  return readFileSync(filePath, "utf8")
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
    .slice(0, limit)
    .map((line) => {
      const example = JSON.parse(line);
      return {
        candidate: extractCandidateFromExample(example),
        teacher: JSON.parse(example.messages.find((message) => message.role === "assistant").content),
      };
    });
}

function evaluateOne(candidate, teacher, predicted) {
  const teacherAxes = teacher.semantic_axes ?? {};
  const predictedAxes = predicted.semantic_axes ?? {};
  const teacherSkills = teacherAxes.skills ?? {};
  const predictedSkills = predictedAxes.skills ?? {};
  const requiredAxes = [
    "identity",
    "skills",
    "experience_summary",
    "experience_chunks",
    "domain",
    "execution_style",
    "trust_signals",
  ];

  const axisPresence = requiredAxes.filter((axis) => predictedAxes[axis] !== undefined).length / requiredAxes.length;
  const skillScores = {
    core_production_skills: jaccard(teacherSkills.core_production_skills, predictedSkills.core_production_skills),
    ml_skills: jaccard(teacherSkills.ml_skills, predictedSkills.ml_skills),
    ml_infra_skills: jaccard(teacherSkills.ml_infra_skills, predictedSkills.ml_infra_skills),
    weak_skills: jaccard(teacherSkills.weak_skills, predictedSkills.weak_skills),
    noisy_non_relevant_skills: jaccard(
      teacherSkills.noisy_non_relevant_skills,
      predictedSkills.noisy_non_relevant_skills
    ),
  };
  const skillAverage =
    Object.values(skillScores).reduce((sum, score) => sum + score, 0) / Object.keys(skillScores).length;

  return {
    candidate_id: candidate.candidate_id,
    valid_schema: Boolean(predicted.metadata && predicted.semantic_axes),
    id_match: predicted.candidate_id === candidate.candidate_id,
    role_match:
      normalize(predictedAxes.identity?.role_family) === normalize(teacherAxes.identity?.role_family),
    years_match: numberClose(predicted.metadata?.years_of_experience, teacher.metadata?.years_of_experience),
    location_match: normalize(predicted.metadata?.location) === normalize(teacher.metadata?.location),
    work_mode_match: normalize(predicted.metadata?.preferred_work_mode) === normalize(teacher.metadata?.preferred_work_mode),
    axis_presence: axisPresence,
    skill_scores: skillScores,
    skill_average: skillAverage,
    teacher_role: teacherAxes.identity?.role_family,
    predicted_role: predictedAxes.identity?.role_family,
  };
}

function average(items, selector) {
  if (items.length === 0) {
    return 0;
  }
  return items.reduce((sum, item) => sum + selector(item), 0) / items.length;
}

async function evaluateFineTunedSemantic() {
  const apiKey = process.env.FINETUNED_API_KEY || process.env.OPENPIPE_API_KEY || process.env.OPENAI_API_KEY;
  const baseURL =
    process.env.FINETUNED_BASE_URL ||
    process.env.OPENPIPE_BASE_URL ||
    process.env.OPENAI_BASE_URL ||
    "https://app.openpipe.ai/api/v1";
  const model = process.env.FINETUNED_MODEL || "openpipe:cruel-cooks-search";
  const validationFile = process.env.FINETUNE_VALIDATION_FILE || "fine_tuning/semantic_validation.jsonl";
  const outputDir = process.env.FINETUNE_EVAL_OUTPUT_DIR || "fine_tuning/eval";
  const limit = parsePositiveInt(process.env.FINETUNE_EVAL_LIMIT, 25);
  const retries = parsePositiveInt(process.env.FINETUNED_RETRIES, 3);

  if (!apiKey) {
    throw new Error("Set FINETUNED_API_KEY, OPENPIPE_API_KEY, or OPENAI_API_KEY.");
  }

  mkdirSync(outputDir, { recursive: true });
  const client = new OpenAI({ apiKey, baseURL });
  const examples = loadExamples(validationFile, limit);
  const details = [];
  const errors = [];

  for (const { candidate, teacher } of examples) {
    try {
      const predicted = await semanticFromFineTunedModel(client, model, candidate, retries);
      details.push(evaluateOne(candidate, teacher, predicted));
    } catch (error) {
      errors.push({ candidate_id: candidate.candidate_id, error: error.message });
    }
  }

  const summary = {
    model,
    validation_file: validationFile,
    evaluated: details.length,
    errors: errors.length,
    schema_pass_rate: average(details, (item) => (item.valid_schema ? 1 : 0)),
    id_match_rate: average(details, (item) => (item.id_match ? 1 : 0)),
    role_match_rate: average(details, (item) => (item.role_match ? 1 : 0)),
    years_match_rate: average(details, (item) => (item.years_match ? 1 : 0)),
    location_match_rate: average(details, (item) => (item.location_match ? 1 : 0)),
    work_mode_match_rate: average(details, (item) => (item.work_mode_match ? 1 : 0)),
    axis_presence_average: average(details, (item) => item.axis_presence),
    skill_average: average(details, (item) => item.skill_average),
  };
  const report = { summary, errors, details };
  const outputFile = `${outputDir}/semantic_eval_${Date.now()}.json`;
  writeFileSync(outputFile, `${JSON.stringify(report, null, 2)}\n`);
  return { outputFile, ...summary };
}

async function main() {
  try {
    const result = await evaluateFineTunedSemantic();
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    console.error("Fine-tuned semantic evaluation failed:", error.message);
    process.exit(1);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}

export {
  evaluateFineTunedSemantic,
};
