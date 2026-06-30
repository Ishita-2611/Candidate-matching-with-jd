import { readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

import OpenAI from "openai";

import { rerankCandidates } from "./rerank-candidates.js";
import { hybridSearch } from "./search-qdrant.js";

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

function parseNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function readJsonFile(filePath) {
  return JSON.parse(readFileSync(filePath, "utf8"));
}

function loadJdInput() {
  const embeddingsFile = process.env.JD_EMBEDDINGS_FILE || "";
  if (embeddingsFile) {
    const parsed = readJsonFile(embeddingsFile);
    if (parsed?.vectors && parsed?.semantic_axes) {
      return parsed;
    }
  }

  return readJsonFile(process.env.JD_SEMANTIC_FILE || "jd-semantic.json");
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function writeCsv(rows, filePath) {
  const header = ["candidate_id", "rank", "score", "reasoning"];
  const lines = [
    header.join(","),
    ...rows.map((row) => header.map((field) => csvCell(row[field])).join(",")),
  ];
  writeFileSync(filePath, `${lines.join("\n")}\n`);
}

function compactArray(values, maxItems = 8) {
  return (values ?? []).filter(Boolean).slice(0, maxItems);
}

function candidateName(payload) {
  return payload.metadata?.name || payload.metadata?.full_name || payload.candidate_id;
}

function candidateContext(result) {
  const payload = result.payload ?? {};
  const axes = payload.semantic_axes ?? {};
  const metadata = payload.metadata ?? {};
  const docs = payload.semantic_documents ?? {};

  return {
    candidate_id: payload.candidate_id,
    name: candidateName(payload),
    score: Number(result.score?.toFixed?.(6) ?? result.score),
    retrieval_score: Number(result.retrieval_score?.toFixed?.(6) ?? result.retrieval_score ?? result.score),
    reranker_model: result.reranker_model ?? null,
    reranker_score: Number(result.reranker_score?.toFixed?.(6) ?? result.reranker_score ?? 0),
    reranker_score_normalized: Number(
      result.reranker_score_normalized?.toFixed?.(6) ?? result.reranker_score_normalized ?? 0
    ),
    vector_score: Number(result.vector_score_normalized?.toFixed?.(6) ?? result.vector_score_normalized),
    skill_overlap_score: Number(result.skill_overlap_score?.toFixed?.(6) ?? result.skill_overlap_score),
    business_score: Number(result.business_score?.toFixed?.(6) ?? result.business_score),
    honeypot_risk: result.honeypot_analysis?.risk_level,
    honeypot_probability: Number(result.honeypot_analysis?.probability?.toFixed?.(6) ?? 0),
    honeypot_signals: compactArray(result.honeypot_analysis?.signals, 6),
    matched_skills: compactArray(
      result.skill_overlap_details?.matched_skills?.map((item) => item.required || item.candidate),
      10
    ),
    missing_skills: compactArray(result.skill_overlap_details?.missing_skills, 8),
    metadata: {
      years_of_experience: metadata.years_of_experience,
      location: metadata.location,
      preferred_work_mode: metadata.preferred_work_mode,
      notice_period_days: metadata.notice_period_days,
      salary_range_lpa: metadata.salary_range_lpa,
      open_to_work: metadata.open_to_work,
    },
    semantic_evidence: {
      identity: docs.identity || axes.identity,
      skills: docs.skills || axes.skills,
      experience_summary: docs.experience_summary || axes.experience_summary,
      domain: docs.domain || axes.domain,
      execution_style: docs.execution_style || axes.execution_style,
      trust_signals: docs.trust_signals || axes.trust_signals,
    },
    penalties: {
      disqualifier_penalty: Number(result.disqualifier_penalty?.toFixed?.(6) ?? 0),
      disqualifier_matches: compactArray(result.disqualifier_penalty_details?.matched_terms, 6),
      redrob_anomalies: compactArray(
        result.honeypot_analysis?.redrob_anomalies?.map((item) => item.name),
        6
      ),
    },
  };
}

function sentenceJoin(parts) {
  return parts.filter(Boolean).join(" ");
}

function deterministicReasoning(context) {
  const metadata = context.metadata ?? {};
  const matched = context.matched_skills?.length
    ? `matches ${context.matched_skills.slice(0, 5).join(", ")}`
    : "has relevant semantic overlap with the JD";
  const workMode = metadata.preferred_work_mode ? `, ${metadata.preferred_work_mode} work mode` : "";
  const location = metadata.location ? ` in ${metadata.location}` : "";
  const years = Number.isFinite(Number(metadata.years_of_experience))
    ? `${metadata.years_of_experience} years of experience`
    : "relevant experience";
  const risk =
    context.honeypot_risk && context.honeypot_risk !== "low"
      ? `Review carefully: ${context.honeypot_risk} honeypot risk from ${context.honeypot_signals.join(", ")}.`
      : "Redrob/honeypot checks do not show a high-risk profile.";
  const missing = context.missing_skills?.length ? `Missing/weak signals include ${context.missing_skills.slice(0, 3).join(", ")}.` : "";

  return sentenceJoin([
    `${context.candidate_id} has ${years}${location}${workMode} and ${matched}.`,
    missing,
    risk,
  ]).slice(0, 700);
}

function llmClient(provider) {
  if (provider === "groq") {
    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) {
      throw new Error("RAG_EXPLANATION_PROVIDER=groq requires GROQ_API_KEY.");
    }
    return new OpenAI({
      apiKey,
      baseURL: process.env.GROQ_BASE_URL || "https://api.groq.com/openai/v1",
    });
  }

  if (provider === "openai") {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error("RAG_EXPLANATION_PROVIDER=openai requires OPENAI_API_KEY.");
    }
    return new OpenAI({ apiKey });
  }

  return null;
}

function jdContext(jdSemantic) {
  const axes = jdSemantic.semantic_axes ?? {};
  return {
    metadata: jdSemantic.metadata ?? {},
    identity: axes.identity,
    skills: axes.skills,
    experience_summary: axes.experience_summary,
    domain: axes.domain,
    execution_style: axes.execution_style,
    trust_signals: axes.trust_signals,
  };
}

async function generateLlmReasoning({ client, model, jd, candidate }) {
  const completion = await client.chat.completions.create({
    model,
    temperature: 0.1,
    max_tokens: 120,
    messages: [
      {
        role: "system",
        content:
          "You write concise hiring ranking explanations. Use only the provided JD and candidate evidence. Return 1-2 sentences, no bullets, no invented facts.",
      },
      {
        role: "user",
        content: JSON.stringify(
          {
            task: "Explain why this candidate is ranked for this JD. Mention major risk only if present.",
            jd,
            candidate,
          },
          null,
          2
        ),
      },
    ],
  });

  return completion.choices?.[0]?.message?.content?.trim() || deterministicReasoning(candidate);
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

async function addReasoning(contexts, jdSemantic) {
  const provider = (process.env.RAG_EXPLANATION_PROVIDER || "local").toLowerCase();
  const explainLimit = parsePositiveInt(process.env.RAG_EXPLAIN_LIMIT, contexts.length);

  if (provider === "local" || provider === "none") {
    return contexts.map((context) => ({
      ...context,
      reasoning_provider: "local",
      reasoning: deterministicReasoning(context),
    }));
  }

  const client = llmClient(provider);
  const model =
    provider === "groq"
      ? process.env.GROQ_MODEL || "llama-3.1-8b-instant"
      : process.env.OPENAI_MODEL || "gpt-4o-mini";
  const concurrency = parsePositiveInt(process.env.RAG_EXPLANATION_CONCURRENCY, 3);
  const jd = jdContext(jdSemantic);

  return mapWithConcurrency(contexts, concurrency, async (context, index) => {
    if (index >= explainLimit) {
      return {
        ...context,
        reasoning_provider: "local",
        reasoning: deterministicReasoning(context),
      };
    }

    try {
      return {
        ...context,
        reasoning_provider: provider,
        reasoning: await generateLlmReasoning({ client, model, jd, candidate: context }),
      };
    } catch (error) {
      return {
        ...context,
        reasoning_provider: "local_fallback",
        reasoning_error: error.message,
        reasoning: deterministicReasoning(context),
      };
    }
  });
}

function toSubmissionRows(results, limit) {
  return results.slice(0, limit).map((result, index) => ({
    candidate_id: result.candidate_id,
    rank: index + 1,
    score: Number(result.score).toFixed(6),
    reasoning: result.reasoning,
  }));
}

async function runRagPipeline() {
  const jdSemantic = loadJdInput();
  const limit = parsePositiveInt(process.env.RAG_LIMIT, parsePositiveInt(process.env.SEARCH_LIMIT, 100));
  const recallLimit = parsePositiveInt(process.env.RAG_RECALL_LIMIT, parsePositiveInt(process.env.SEARCH_RECALL_LIMIT, 500));
  const outputJson = process.env.RAG_OUTPUT_JSON || "rag-results.json";
  const outputCsv = process.env.RAG_OUTPUT_CSV || "rag-submission.csv";
  const minScore = parseNumber(process.env.RAG_MIN_SCORE, Number.NEGATIVE_INFINITY);

  const retrieved = await hybridSearch(jdSemantic, { limit: recallLimit, recallLimit });
  const rerankerEnabled = process.env.RERANKER_ENABLED !== "false";
  let rankedResults = retrieved.results;
  let rerankerError = null;

  if (rerankerEnabled) {
    try {
      rankedResults = await rerankCandidates(jdSemantic, retrieved.results);
    } catch (error) {
      rerankerError = error.message;
      rankedResults = retrieved.results;
    }
  }

  const contexts = rankedResults.slice(0, limit).map(candidateContext).filter((context) => context.score >= minScore);
  const explained = await addReasoning(contexts, jdSemantic);
  const rows = toSubmissionRows(explained, limit);

  const output = {
    generated_at: new Date().toISOString(),
    jd_input: process.env.JD_EMBEDDINGS_FILE || process.env.JD_SEMANTIC_FILE || "jd-semantic.json",
    collection: retrieved.collection,
    constraints: retrieved.constraints,
    filtered_count: retrieved.filteredCount,
    recall_limit: retrieved.recallLimit,
    reranker_enabled: rerankerEnabled,
    reranker_error: rerankerError,
    result_count: explained.length,
    results: explained,
  };

  writeFileSync(outputJson, `${JSON.stringify(output, null, 2)}\n`);
  writeCsv(rows, outputCsv);

  return {
    outputJson,
    outputCsv,
    rows: rows.length,
    topCandidate: rows[0]?.candidate_id ?? null,
  };
}

async function main() {
  try {
    const result = await runRagPipeline();
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    console.error("RAG pipeline failed:", error.message);
    process.exit(1);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}

export {
  addReasoning,
  candidateContext,
  deterministicReasoning,
  runRagPipeline,
};
