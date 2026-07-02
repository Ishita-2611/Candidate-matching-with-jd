import { env, AutoModelForSequenceClassification, AutoTokenizer } from "@xenova/transformers";

import { queryDocumentForVector } from "./search-qdrant.js";

try {
  process.loadEnvFile?.();
} catch (error) {
  if (error.code !== "ENOENT") {
    throw error;
  }
}

env.allowLocalModels = true;
env.cacheDir = process.env.TRANSFORMERS_CACHE || "./model_cache";

const defaultRerankerModel = process.env.RERANKER_MODEL || "Xenova/ms-marco-MiniLM-L-6-v2";

let rerankerPromise;

function parsePositiveInt(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / sum);
}

async function getReranker(modelName = defaultRerankerModel) {
  rerankerPromise ||= Promise.all([
    AutoTokenizer.from_pretrained(modelName),
    AutoModelForSequenceClassification.from_pretrained(modelName),
  ]).then(([tokenizer, model]) => ({ tokenizer, model, modelName }));

  return rerankerPromise;
}

function clampText(value, maxChars) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  return text.length > maxChars ? `${text.slice(0, maxChars)}...` : text;
}

function candidateRerankText(payload) {
  const metadata = payload.metadata ?? {};
  const docs = payload.semantic_documents ?? {};
  const axes = payload.semantic_axes ?? {};

  return [
    `Candidate ${payload.candidate_id}.`,
    `Metadata: years ${metadata.years_of_experience ?? "unknown"}, location ${metadata.location ?? "unknown"}, work mode ${metadata.preferred_work_mode ?? "unknown"}, notice ${metadata.notice_period_days ?? "unknown"} days.`,
    `Identity: ${docs.identity ?? JSON.stringify(axes.identity ?? {})}`,
    `Skills: ${docs.skills ?? JSON.stringify(axes.skills ?? {})}`,
    `Experience: ${docs.experience_summary ?? JSON.stringify(axes.experience_summary ?? {})}`,
    `Domain: ${docs.domain ?? JSON.stringify(axes.domain ?? {})}`,
    `Execution style: ${docs.execution_style ?? JSON.stringify(axes.execution_style ?? {})}`,
    `Trust signals: ${docs.trust_signals ?? JSON.stringify(axes.trust_signals ?? {})}`,
  ].join("\n");
}

function jdRerankText(jdSemantic) {
  return [
    queryDocumentForVector(jdSemantic, "default"),
    queryDocumentForVector(jdSemantic, "skills"),
    queryDocumentForVector(jdSemantic, "experience_summary"),
    queryDocumentForVector(jdSemantic, "execution_style"),
  ].join("\n");
}

function logitsToScores(logits) {
  const dims = logits.dims ?? [];
  const data = Array.from(logits.data, Number);
  const batchSize = dims[0] || 1;
  const labels = dims[1] || Math.max(1, data.length / batchSize);
  const scores = [];

  for (let batchIndex = 0; batchIndex < batchSize; batchIndex += 1) {
    const start = batchIndex * labels;
    const row = data.slice(start, start + labels);

    if (row.length === 1) {
      scores.push(sigmoid(row[0]));
      continue;
    }

    const probabilities = softmax(row);
    scores.push(probabilities.at(-1));
  }

  return scores;
}

async function scorePairs(jdText, candidateTexts, options = {}) {
  const modelName = options.modelName || defaultRerankerModel;
  const maxLength = options.maxLength || parsePositiveInt(process.env.RERANKER_MAX_LENGTH, 512);
  const { tokenizer, model } = await getReranker(modelName);
  const queries = candidateTexts.map(() => jdText);
  const inputs = tokenizer(queries, {
    text_pair: candidateTexts,
    padding: true,
    truncation: true,
    max_length: maxLength,
  });
  const outputs = await model(inputs);
  return logitsToScores(outputs.logits);
}

function normalizeScores(values) {
  const finite = values.filter(Number.isFinite);
  const min = Math.min(...finite, 0);
  const max = Math.max(...finite, 1);

  if (max === min) {
    return values.map(() => 0);
  }

  return values.map((value) => (Number.isFinite(value) ? (value - min) / (max - min) : 0));
}

async function rerankCandidates(jdSemantic, results, options = {}) {
  if (results.length === 0) {
    return [];
  }

  const modelName = options.modelName || defaultRerankerModel;
  const batchSize = options.batchSize || parsePositiveInt(process.env.RERANKER_BATCH_SIZE, 8);
  const candidateMaxChars = options.candidateMaxChars || parsePositiveInt(process.env.RERANKER_CANDIDATE_MAX_CHARS, 3500);
  const jdMaxChars = options.jdMaxChars || parsePositiveInt(process.env.RERANKER_JD_MAX_CHARS, 2500);
  const weights = {
    reranker: Number(process.env.RERANKER_WEIGHT || 0.7),
    vector: Number(process.env.RERANKER_VECTOR_WEIGHT || 0.15),
    skill: Number(process.env.RERANKER_SKILL_WEIGHT || 0.075),
    business: Number(process.env.RERANKER_BUSINESS_WEIGHT || 0.075),
    disqualifier: Number(process.env.RERANKER_DISQUALIFIER_PENALTY || 0.15),
    honeypot: Number(process.env.RERANKER_HONEYPOT_PENALTY || 0.1),
  };

  const jdText = clampText(jdRerankText(jdSemantic), jdMaxChars);
  const candidateTexts = results.map((result) => clampText(candidateRerankText(result.payload ?? {}), candidateMaxChars));
  const rerankerScores = [];

  for (let index = 0; index < candidateTexts.length; index += batchSize) {
    const batch = candidateTexts.slice(index, index + batchSize);
    rerankerScores.push(...(await scorePairs(jdText, batch, { modelName })));
  }

  const normalizedRerankerScores = normalizeScores(rerankerScores);

  return results
    .map((result, index) => {
      const rerankerScore = rerankerScores[index] ?? 0;
      const rerankerScoreNormalized = normalizedRerankerScores[index] ?? 0;
      const finalScore =
        rerankerScoreNormalized * weights.reranker +
        (result.vector_score_normalized ?? 0) * weights.vector +
        (result.skill_overlap_score ?? 0) * weights.skill +
        (result.business_score ?? 0) * weights.business -
        (result.disqualifier_penalty ?? 0) * weights.disqualifier -
        (result.honeypot_analysis?.probability ?? 0) * weights.honeypot;

      return {
        ...result,
        retrieval_score: result.score,
        reranker_model: modelName,
        reranker_score: rerankerScore,
        reranker_score_normalized: rerankerScoreNormalized,
        reranker_weights: weights,
        score: finalScore,
      };
    })
    .sort((a, b) => b.score - a.score);
}

export {
  candidateRerankText,
  jdRerankText,
  rerankCandidates,
  scorePairs,
};
