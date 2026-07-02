import { readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

import { embedText, qdrantRequest, vectorNames } from "./embed-batches-qdrant.js";
import { buildQdrantHardFilter, hardConstraintsFromJd } from "./hard-filters.js";
import { redrobSignalAnomalies, redrobSignalScore } from "./redrob-signals.js";

try {
  process.loadEnvFile?.();
} catch (error) {
  if (error.code !== "ENOENT") {
    throw error;
  }
}

const qdrantCollection = process.env.QDRANT_COLLECTION || "candidate_semantic_multivectors_bge_m3";
const defaultVectorWeights = {
  identity: 0.25,
  skills: 0.35,
  experience_summary: 0.25,
  domain: 0.1,
  trust_signals: 0.05,
};
const defaultRankWeights = {
  vector: 0.6,
  skillOverlap: 0.2,
  business: 0.2,
};
const defaultPenaltyWeights = {
  disqualifier: 0.15,
  honeypot: 0.1,
};

function parsePositiveInt(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function stableJson(value) {
  return JSON.stringify(value ?? {});
}

function parseVectorWeights(value) {
  if (!value) {
    return defaultVectorWeights;
  }

  const weights = {};
  for (const pair of String(value).split(",")) {
    const [rawName, rawWeight] = pair.split(":");
    const name = rawName?.trim();
    const weight = Number(rawWeight);

    if (!vectorNames.includes(name) || !Number.isFinite(weight) || weight <= 0) {
      throw new Error(
        `Invalid SEARCH_VECTOR_WEIGHTS entry "${pair}". Use format "skills:0.35,experience_summary:0.25".`
      );
    }

    weights[name] = weight;
  }

  return weights;
}

function parseRankWeights(value) {
  if (!value) {
    return defaultRankWeights;
  }

  const allowed = new Set(Object.keys(defaultRankWeights));
  const weights = {};
  for (const pair of String(value).split(",")) {
    const [rawName, rawWeight] = pair.split(":");
    const name = rawName?.trim();
    const weight = Number(rawWeight);

    if (!allowed.has(name) || !Number.isFinite(weight) || weight < 0) {
      throw new Error(`Invalid SEARCH_RANK_WEIGHTS entry "${pair}". Use keys: ${[...allowed].join(", ")}.`);
    }

    weights[name] = weight;
  }

  return { ...defaultRankWeights, ...weights };
}

function parsePenaltyWeights(value) {
  if (!value) {
    return defaultPenaltyWeights;
  }

  const allowed = new Set(Object.keys(defaultPenaltyWeights));
  const weights = {};
  for (const pair of String(value).split(",")) {
    const [rawName, rawWeight] = pair.split(":");
    const name = rawName?.trim();
    const weight = Number(rawWeight);

    if (!allowed.has(name) || !Number.isFinite(weight) || weight < 0) {
      throw new Error(`Invalid SEARCH_PENALTY_WEIGHTS entry "${pair}". Use keys: ${[...allowed].join(", ")}.`);
    }

    weights[name] = weight;
  }

  return { ...defaultPenaltyWeights, ...weights };
}

function readJsonFile(filePath) {
  return JSON.parse(readFileSync(filePath, "utf8"));
}

function loadJdEmbeddings(filePath) {
  if (!filePath) {
    return null;
  }

  const parsed = readJsonFile(filePath);
  return parsed?.vectors && typeof parsed.vectors === "object" ? parsed : null;
}

function normalizeToken(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9+#.]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizedSet(values) {
  return new Set(
    (values ?? [])
      .flatMap((value) => (Array.isArray(value) ? value : [value]))
      .map(normalizeToken)
      .filter(Boolean)
  );
}

function normalizedList(values) {
  return [...normalizedSet(values)];
}

function skillValues(skills = {}) {
  return [
    ...(skills.core_production_skills ?? []),
    ...(skills.ml_skills ?? []),
    ...(skills.ml_infra_skills ?? []),
  ];
}

function findSkillMatch(skill, candidateSkills) {
  if (candidateSkills.has(skill)) {
    return skill;
  }

  for (const candidateSkill of candidateSkills) {
    if (candidateSkill.includes(skill) || skill.includes(candidateSkill)) {
      return candidateSkill;
    }
  }

  return null;
}

function skillOverlapDetails(jdSemantic, candidatePayload) {
  const required = [...normalizedSet(skillValues(jdSemantic.semantic_axes?.skills))];
  const candidateSkills = normalizedSet(skillValues(candidatePayload.semantic_axes?.skills));
  const matched = [];
  const missing = [];

  if (required.length === 0) {
    return {
      score: 0,
      required_skills: [],
      matched_skills: [],
      missing_skills: [],
    };
  }

  for (const skill of required) {
    const match = findSkillMatch(skill, candidateSkills);
    if (match) {
      matched.push({ required: skill, candidate: match });
    } else {
      missing.push(skill);
    }
  }

  return {
    score: matched.length / required.length,
    required_skills: required,
    matched_skills: matched,
    missing_skills: missing,
  };
}

function skillOverlapScore(jdSemantic, candidatePayload) {
  return skillOverlapDetails(jdSemantic, candidatePayload).score;
}

function clamp01(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
}

function redrobSignalsFromPayload(candidatePayload) {
  const metadata = candidatePayload.metadata ?? {};
  const trust = candidatePayload.semantic_axes?.trust_signals ?? {};

  return {
    profile_completeness_score: trust.profile_completeness ?? trust.profile_completeness_score,
    open_to_work_flag: metadata.open_to_work,
    recruiter_response_rate: trust.recruiter_response_rate,
    skill_assessment_scores: trust.skill_assessment_scores,
    notice_period_days: metadata.notice_period_days,
    expected_salary_range_inr_lpa: metadata.salary_range_lpa,
    preferred_work_mode: metadata.preferred_work_mode,
    willing_to_relocate: metadata.willing_to_relocate,
    github_activity_score: trust.github_score,
    interview_completion_rate: trust.interview_completion_rate,
    offer_acceptance_rate: trust.offer_acceptance_rate,
    verified_email: trust.verified_email,
    verified_phone: trust.verified_phone,
    linkedin_connected: trust.linkedin_connected,
  };
}

function businessScoreDetails(jdSemantic, candidatePayload) {
  return redrobSignalScore(redrobSignalsFromPayload(candidatePayload), jdSemantic.metadata ?? {});
}

function disqualifierTerms(jdSemantic) {
  const axes = jdSemantic.semantic_axes ?? {};
  const disqualifierChunks = (axes.experience_chunks ?? [])
    .filter((chunk) => (chunk.tags ?? []).some((tag) => normalizeToken(tag).includes("disqualifier")))
    .map((chunk) => chunk.description);

  return normalizedList([
    ...(axes.skills?.noisy_non_relevant_skills ?? []),
    ...(axes.domain?.missing_domains ?? []),
    ...disqualifierChunks,
  ]).filter((term) => term.length >= 3);
}

function candidateSearchText(candidatePayload) {
  const axes = candidatePayload.semantic_axes ?? {};
  const chunks = axes.experience_chunks ?? [];

  return normalizeToken(
    [
      JSON.stringify(candidatePayload.metadata ?? {}),
      JSON.stringify(axes.identity ?? {}),
      JSON.stringify(axes.skills ?? {}),
      JSON.stringify(axes.domain ?? {}),
      JSON.stringify(axes.experience_summary ?? {}),
      chunks.map((chunk) => chunk.description || JSON.stringify(chunk)).join(" "),
    ].join(" ")
  );
}

function disqualifierPenaltyDetails(jdSemantic, candidatePayload) {
  const terms = disqualifierTerms(jdSemantic);
  if (terms.length === 0) {
    return {
      score: 0,
      terms: [],
      matched_terms: [],
    };
  }

  const candidateText = candidateSearchText(candidatePayload);
  const matched = [];

  for (const term of terms) {
    if (candidateText.includes(term)) {
      matched.push(term);
    }
  }

  return {
    score: matched.length / terms.length,
    terms,
    matched_terms: matched,
  };
}

function disqualifierPenaltyScore(jdSemantic, candidatePayload) {
  return disqualifierPenaltyDetails(jdSemantic, candidatePayload).score;
}

function honeypotAnalysis(candidatePayload) {
  const metadata = candidatePayload.metadata ?? {};
  const trust = candidatePayload.semantic_axes?.trust_signals ?? {};
  const signals = [];
  const redrobSignals = redrobSignalsFromPayload(candidatePayload);
  const redrobAnomalies = redrobSignalAnomalies(redrobSignals);

  const explicitProbability = Number(trust.honeytrap_probability ?? trust.honeypot_probability);
  if (Number.isFinite(explicitProbability)) {
    return {
      probability: clamp01(explicitProbability),
      risk_level: explicitProbability >= 0.7 ? "high" : explicitProbability >= 0.4 ? "medium" : "low",
      signals: ["explicit_honeypot_probability"],
    };
  }

  const profileCompleteness = Number(trust.profile_completeness);
  const githubScore = Number(trust.github_score);
  const recruiterResponseRate = Number(trust.recruiter_response_rate);
  const interviewCompletionRate = Number(trust.interview_completion_rate);
  const offerAcceptanceRate = Number(trust.offer_acceptance_rate);

  if (Number.isFinite(profileCompleteness) && profileCompleteness < 0.35) {
    signals.push("low_profile_completeness");
  }
  if (Number.isFinite(githubScore) && githubScore < 0.15) {
    signals.push("low_github_signal");
  }
  if (Number.isFinite(recruiterResponseRate) && recruiterResponseRate < 0.25) {
    signals.push("low_recruiter_response_rate");
  }
  if (Number.isFinite(interviewCompletionRate) && interviewCompletionRate < 0.25) {
    signals.push("low_interview_completion_rate");
  }
  if (Number.isFinite(offerAcceptanceRate) && offerAcceptanceRate < 0.25) {
    signals.push("low_offer_acceptance_rate");
  }
  if (metadata.open_to_work === false) {
    signals.push("not_open_to_work");
  }

  signals.push(...redrobAnomalies.map((anomaly) => anomaly.name));
  const anomalySeverity = redrobAnomalies.reduce((sum, anomaly) => sum + anomaly.severity, 0);
  const probability = clamp01((signals.length + anomalySeverity) / 8);
  return {
    probability,
    risk_level: probability >= 0.7 ? "high" : probability >= 0.4 ? "medium" : "low",
    signals,
    redrob_anomalies: redrobAnomalies,
  };
}

function queryDocumentForVector(jdSemantic, vectorName) {
  const axes = jdSemantic.semantic_axes ?? {};

  const documents = {
    identity: stableJson(axes.identity),
    skills: stableJson(axes.skills),
    experience_summary: stableJson({
      summary: axes.experience_summary ?? {},
      chunks: axes.experience_chunks ?? [],
    }),
    domain: stableJson(axes.domain),
    execution_style: stableJson(axes.execution_style),
    trust_signals: stableJson(axes.trust_signals),
    default: [
      `metadata: ${stableJson(jdSemantic.metadata)}`,
      `identity: ${stableJson(axes.identity)}`,
      `skills: ${stableJson(axes.skills)}`,
      `experience_summary: ${stableJson(axes.experience_summary)}`,
      `experience_chunks: ${stableJson(axes.experience_chunks)}`,
      `domain: ${stableJson(axes.domain)}`,
      `execution_style: ${stableJson(axes.execution_style)}`,
      `trust_signals: ${stableJson(axes.trust_signals)}`,
    ].join("\n"),
  };

  return documents[vectorName] ?? documents.default;
}

async function countCandidates(filter) {
  const response = await qdrantRequest(`/collections/${qdrantCollection}/points/count`, {
    method: "POST",
    body: {
      exact: true,
      ...(filter ? { filter } : {}),
    },
  });

  return response.result?.count ?? 0;
}

async function ensurePayloadIndexes() {
  const indexes = [
    ["metadata.years_of_experience", "float"],
    ["metadata.location", "keyword"],
    ["metadata.country", "keyword"],
    ["metadata.preferred_work_mode", "keyword"],
    ["metadata.notice_period_days", "integer"],
    ["metadata.salary_range_lpa.min", "float"],
    ["metadata.salary_range_lpa.max", "float"],
    ["metadata.open_to_work", "bool"],
  ];

  for (const [fieldName, fieldSchema] of indexes) {
    await qdrantRequest(`/collections/${qdrantCollection}/index?wait=true`, {
      method: "PUT",
      body: {
        field_name: fieldName,
        field_schema: fieldSchema,
      },
    });
  }
}

async function hybridSearch(jdSemantic, options = {}) {
  const limit = options.limit || parsePositiveInt(process.env.SEARCH_LIMIT, 10);
  const recallLimit = options.recallLimit || parsePositiveInt(process.env.SEARCH_RECALL_LIMIT, Math.max(limit * 5, 50));
  const vectorWeights = options.vectorWeights || parseVectorWeights(process.env.SEARCH_VECTOR_WEIGHTS);
  const rankWeights = options.rankWeights || parseRankWeights(process.env.SEARCH_RANK_WEIGHTS);
  const penaltyWeights = options.penaltyWeights || parsePenaltyWeights(process.env.SEARCH_PENALTY_WEIGHTS);
  const weightedVectors = Object.entries(vectorWeights);

  for (const [vectorName] of weightedVectors) {
    if (!vectorNames.includes(vectorName)) {
      throw new Error(`Vector weight includes unknown vector "${vectorName}". Use one of: ${vectorNames.join(", ")}`);
    }
  }

  const constraints = options.constraints || hardConstraintsFromJd(jdSemantic);
  const filter = buildQdrantHardFilter(constraints);
  await ensurePayloadIndexes();
  const filteredCount = await countCandidates(filter);
  const merged = new Map();

  for (const [vectorName, weight] of weightedVectors) {
    const queryVector = jdSemantic.vectors?.[vectorName] ?? (await embedText(queryDocumentForVector(jdSemantic, vectorName)));
    const response = await qdrantRequest(`/collections/${qdrantCollection}/points/search`, {
      method: "POST",
      body: {
        vector: {
          name: vectorName,
          vector: queryVector,
        },
        filter,
        limit: recallLimit,
        with_payload: true,
        with_vector: false,
      },
    });

    for (const point of response.result ?? []) {
      const existing = merged.get(point.id) ?? {
        ...point,
        score: 0,
        vector_scores: {},
      };

      existing.score += point.score * weight;
      existing.vector_scores[vectorName] = point.score;
      if (!existing.payload && point.payload) {
        existing.payload = point.payload;
      }
      merged.set(point.id, existing);
    }
  }

  const vectorScores = [...merged.values()].map((point) => point.score);
  const maxVectorScore = Math.max(...vectorScores, 0);
  const results = [...merged.values()]
    .map((point) => {
      const vectorScore = maxVectorScore > 0 ? point.score / maxVectorScore : 0;
      const skillDetails = skillOverlapDetails(jdSemantic, point.payload ?? {});
      const businessDetails = businessScoreDetails(jdSemantic, point.payload ?? {});
      const penaltyDetails = disqualifierPenaltyDetails(jdSemantic, point.payload ?? {});
      const honeypot = honeypotAnalysis(point.payload ?? {});
      const overlapScore = skillDetails.score;
      const metadataScore = businessDetails.score;
      const disqualifierPenalty = penaltyDetails.score;
      const finalScore =
        vectorScore * rankWeights.vector +
        overlapScore * rankWeights.skillOverlap +
        metadataScore * rankWeights.business -
        disqualifierPenalty * penaltyWeights.disqualifier -
        honeypot.probability * penaltyWeights.honeypot;

      return {
        ...point,
        vector_weighted_score: point.score,
        vector_score_normalized: vectorScore,
        skill_overlap_score: overlapScore,
        skill_overlap_details: skillDetails,
        business_score: metadataScore,
        business_score_details: businessDetails,
        honeypot_analysis: honeypot,
        disqualifier_penalty: disqualifierPenalty,
        disqualifier_penalty_details: penaltyDetails,
        score: finalScore,
      };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);

  return {
    collection: qdrantCollection,
    vectorWeights,
    rankWeights,
    penaltyWeights,
    constraints,
    filteredCount,
    recallLimit,
    results,
  };
}

function resultSummary(point) {
  const payload = point.payload ?? {};
  const metadata = payload.metadata ?? {};

  return {
    score: point.score,
    vector_weighted_score: point.vector_weighted_score,
    vector_score_normalized: point.vector_score_normalized,
    skill_overlap_score: point.skill_overlap_score,
    skill_overlap_details: point.skill_overlap_details,
    business_score: point.business_score,
    business_score_details: point.business_score_details,
    honeypot_analysis: point.honeypot_analysis,
    disqualifier_penalty: point.disqualifier_penalty,
    disqualifier_penalty_details: point.disqualifier_penalty_details,
    vector_scores: point.vector_scores,
    candidate_id: payload.candidate_id,
    years_of_experience: metadata.years_of_experience,
    location: metadata.location,
    preferred_work_mode: metadata.preferred_work_mode,
    role_family: payload.semantic_axes?.identity?.role_family,
    core_skills: payload.semantic_axes?.skills?.core_production_skills ?? [],
  };
}

async function main() {
  try {
    const jdSemanticFile = process.env.JD_SEMANTIC_FILE || "jd-semantic.json";
    const jdEmbeddingsFile = process.env.JD_EMBEDDINGS_FILE || "";
    const outputMode = process.env.SEARCH_OUTPUT || "summary";
    const resultsFile = process.env.SEARCH_RESULTS_FILE || "";
    const jdSemantic = loadJdEmbeddings(jdEmbeddingsFile) || readJsonFile(jdSemanticFile);
    const searchResult = await hybridSearch(jdSemantic);

    console.log(`Collection: ${searchResult.collection}`);
    console.log(`JD input: ${jdEmbeddingsFile || jdSemanticFile}`);
    console.log(`Vector weights: ${JSON.stringify(searchResult.vectorWeights)}`);
    console.log(`Rank weights: ${JSON.stringify(searchResult.rankWeights)}`);
    console.log(`Penalty weights: ${JSON.stringify(searchResult.penaltyWeights)}`);
    console.log(`Hard constraints: ${JSON.stringify(searchResult.constraints)}`);
    console.log(`Candidates after hard filter: ${searchResult.filteredCount}`);
    console.log("");

    const output =
      outputMode === "raw"
        ? searchResult
        : {
            ...searchResult,
            results: searchResult.results.map(resultSummary),
          };

    if (resultsFile) {
      writeFileSync(resultsFile, `${JSON.stringify(output, null, 2)}\n`);
      console.log(`Saved ranked candidates to ${resultsFile}`);
    }

    console.log(JSON.stringify(output, null, 2));
  } catch (error) {
    console.error("Hybrid search failed:", error.message);
    process.exit(1);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}

export {
  hardConstraintsFromJd,
  hybridSearch,
  queryDocumentForVector,
};
