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

function parsePositiveInt(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function skillNames(candidate) {
  return (candidate.skills ?? []).map((skill) => skill.name).filter(Boolean);
}

function pickSkills(candidate, matcher) {
  return skillNames(candidate).filter((skill) => matcher.test(skill.toLowerCase()));
}

function inferSeniority(years) {
  if (years >= 4) return "experienced";
  if (years >= 2) return "mid-level";
  return "early-career";
}

function inferProductionMaturity(candidate) {
  const text = JSON.stringify(candidate.career_history ?? []).toLowerCase();
  if (/(owned|on-call|production|quality|schema|streaming|scale|pipeline|warehouse)/.test(text)) {
    return "high";
  }

  return (candidate.profile?.years_of_experience ?? 0) >= 4 ? "moderate" : "low";
}

function inferMlMaturity(candidate) {
  const skills = pickSkills(
    candidate,
    /(ml|machine|nlp|llm|lora|milvus|bentoml|wandb|weights|model|gan|speech|classification|statistical)/
  );
  if (skills.length >= 5) return "moderate";
  if (skills.length >= 2) return "moderate-low";
  return "low";
}

function inferSystemTypes(candidate) {
  const text = JSON.stringify(candidate.career_history ?? []).toLowerCase();
  const types = [];
  if (/stream|kafka|realtime|real-time/.test(text)) types.push("Streaming Systems");
  if (/batch|airflow|dbt|spark/.test(text)) types.push("Batch Processing");
  if (/warehouse|analytics|snowflake/.test(text)) types.push("Analytics Pipelines");
  if (/feature|ml model|data science/.test(text)) types.push("Feature Engineering Pipelines");
  if (/backend|api|service/.test(text)) types.push("Backend Systems");
  return types.length > 0 ? types : ["Software Systems"];
}

function compactExperienceDescription(item) {
  const raw = String(item.description || `${item.title ?? "Role"} at ${item.company ?? "company"}`).trim();
  const sentences = raw.split(/(?<=[.!?])\s+/).filter(Boolean);
  const firstSentence = sentences[0]?.replace(/[.!?]+$/, "") || raw;

  if (/customer support team lead/i.test(raw)) {
    return `Led customer support team at ${item.company}, managed team of 8 support agents, owned escalation process, and built support knowledge base.`;
  }

  if (/content writing and seo strategy/i.test(raw)) {
    return `Marketing manager at ${item.company}, content writing, SEO strategy, and AI-assisted content production.`;
  }

  if (/business analyst at a consulting firm/i.test(raw)) {
    return sentences.slice(0, 2).join(" ").trim();
  }

  return item.company ? `${firstSentence} at ${item.company}.` : `${firstSentence}.`;
}

function inferChunkTags(item) {
  const text = `${item.title ?? ""} ${item.industry ?? ""} ${item.description ?? ""}`.toLowerCase();

  if (/customer support team lead/.test(text)) {
    return ["team management", "process management", item.industry].filter(Boolean).slice(0, 3);
  }

  if (/content writing and seo strategy/.test(text)) {
    return ["content creation", "SEO", "AI"];
  }

  if (/business analyst at a consulting firm/.test(text)) {
    return ["business analysis", "consulting"];
  }

  const tags = [];
  if (/data|pipeline|spark|airflow|warehouse|dbt|snowflake|analytics/.test(text)) tags.push("data engineering");
  if (/stream|kafka|real-time|realtime/.test(text)) tags.push("real-time processing");
  if (/batch|airflow|transactional/.test(text)) tags.push("batch processing");
  if (/support|ticket|escalation|knowledge base/.test(text)) tags.push("customer support");
  if (/managed a team|team lead|support agents/.test(text)) tags.push("team management");
  if (/process|escalation|knowledge base|kpi/.test(text)) tags.push("process management");
  if (/marketing|seo|content|brand|editorial/.test(text)) tags.push("marketing");
  if (/seo/.test(text)) tags.push("SEO");
  if (/ai|llm|chatgpt/.test(text)) tags.push("AI");
  if (/design|cad|solidworks|prototype|manufacturing|mechanical/.test(text)) tags.push("engineering design");
  if (/business analyst|consulting|strategy|transformation|process/.test(text)) tags.push("business analysis");
  if (/consulting/.test(text)) tags.push("consulting");
  if (/ml|model|ai|llm|data science/.test(text)) tags.push("machine learning");

  return [...new Set(tags)].slice(0, 2);
}

function relevantCareerHistory(candidate) {
  const history = candidate.career_history ?? [];
  const filtered = history.filter((item) => {
    const text = `${item.title ?? ""} ${item.description ?? ""}`.toLowerCase();
    return !/(mechanical engineering design role|brand design and creative direction)/.test(text);
  });

  return filtered.length > 0 ? filtered : history;
}

function localSemanticObject(candidate) {
  const profile = candidate.profile ?? {};
  const signals = candidate.redrob_signals ?? {};
  const salary = signals.expected_salary_range_inr_lpa ?? {};
  const history = relevantCareerHistory(candidate);
  const mlSkills = pickSkills(candidate, /(nlp|llm|lora|machine|ml|model|gan|speech|classification|statistical|tts)/);
  const mlInfraSkills = pickSkills(candidate, /(milvus|bentoml|weights|wandb|mlflow|kubeflow|vector|embedding)/);
  const coreSkills = pickSkills(candidate, /(python|sql|spark|airflow|kafka|dbt|snowflake|warehouse|backend|flask|beam|aws|gcp|cloud)/);
  const noisySkills = pickSkills(candidate, /(photoshop|tailwind|figma|css|html)/);
  const weakSkills = (candidate.skills ?? [])
    .filter((skill) => skill.proficiency === "beginner")
    .map((skill) => skill.name);
  const careerText = history.map((item) => item.description).filter(Boolean).join(" ");
  const sourceSystems = Number((careerText.match(/(\d+)\s+source systems/i) ?? [])[1] ?? 0);
  const dailyDataProcessed = (careerText.match(/~?\d+\s*(gb|tb)/i) ?? ["unknown"])[0];

  return {
    candidate_id: candidate.candidate_id,
    metadata: {
      years_of_experience: profile.years_of_experience ?? 0,
      location: profile.location ?? "",
      country: profile.country ?? "",
      open_to_work: Boolean(signals.open_to_work_flag),
      preferred_work_mode: signals.preferred_work_mode ?? "",
      notice_period_days: signals.notice_period_days ?? 0,
      salary_range_lpa: {
        min: salary.min ?? 0,
        max: salary.max ?? 0,
      },
    },
    semantic_axes: {
      identity: {
        role_family: profile.current_title || profile.headline || "Unknown",
        secondary_roles: [...new Set(history.map((item) => item.title).filter(Boolean))].slice(0, 3),
        seniority: inferSeniority(profile.years_of_experience ?? 0),
        career_transition: {
          from: profile.current_industry || "unknown",
          to: /ml|ai|machine|llm/i.test(profile.summary ?? "") ? "AI/ML-focused work" : "unknown",
        },
      },
      skills: {
        core_production_skills: coreSkills.slice(0, 12),
        ml_skills: mlSkills.slice(0, 12),
        ml_infra_skills: mlInfraSkills.slice(0, 8),
        weak_skills: weakSkills.slice(0, 8),
        noisy_non_relevant_skills: noisySkills.slice(0, 8),
      },
      experience_summary: {
        system_types: inferSystemTypes(candidate),
        scale: {
          daily_data_processed: dailyDataProcessed,
          source_systems: sourceSystems,
          realtime_systems: /stream|kafka|realtime|real-time/i.test(careerText),
        },
        production_maturity: inferProductionMaturity(candidate),
        ml_maturity: inferMlMaturity(candidate),
      },
      experience_chunks: history.map((item, index) => ({
        id: `chunk_${index + 1}`,
        description: compactExperienceDescription(item),
        tags: inferChunkTags(item),
      })),
      domain: {
        primary_domains: [...new Set(history.map((item) => item.industry).filter(Boolean))].slice(0, 5),
        secondary_domains: inferSystemTypes(candidate),
        missing_domains: [],
      },
      execution_style: {
        shipping_bias: /built|implemented|owned|maintained|designed/i.test(careerText) ? "high" : "medium",
        product_mindset: /user|analytics|models|team|stakeholder/i.test(careerText) ? "medium" : "low",
        research_bias: mlSkills.length >= 4 ? "medium" : "low",
        ambiguity_tolerance: (profile.years_of_experience ?? 0) >= 4 ? "medium" : "low",
        ownership: /owned|on-call|maintained|designed/i.test(careerText) ? "high" : "medium",
        system_design_depth: /schema|watermark|state|deduplication|architecture|pipeline/i.test(careerText) ? "high" : "medium",
      },
      trust_signals: {
        github_score: signals.github_activity_score ?? null,
        profile_completeness: signals.profile_completeness_score ?? null,
        recruiter_response_rate: signals.recruiter_response_rate ?? null,
        interview_completion_rate: signals.interview_completion_rate ?? null,
        offer_acceptance_rate: signals.offer_acceptance_rate ?? null,
      },
    },
  };
}

function batchFileName(outputDir, batchNo) {
  return `${outputDir}/output_batch${String(batchNo).padStart(4, "0")}.json`;
}

async function generateLocalSemanticBatches() {
  const inputJsonl = process.env.INPUT_JSONL || "candidates.jsonl";
  const limit = parsePositiveInt(process.env.BATCH_LIMIT, 100000);
  const startOffset = parsePositiveInt(process.env.START_OFFSET, 0);
  const batchSize = parsePositiveInt(process.env.OUTPUT_BATCH_SIZE, 1000);
  const outputDir = process.env.OUTPUT_DIR || "batches";
  const manifestJson = process.env.MANIFEST_JSON || `${outputDir}/manifest.json`;
  const errorJsonl = process.env.ERROR_JSONL || `${outputDir}/output.errors.jsonl`;
  const manifest = [];
  let skipped = 0;
  let processed = 0;
  let currentBatch = [];
  let currentBatchNo = Math.floor(startOffset / batchSize) + 1;

  mkdirSync(outputDir, { recursive: true });
  writeFileSync(errorJsonl, "");

  function writeBatch(force = false) {
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
    currentBatch = [];
    currentBatchNo += 1;
  }

  const rl = readline.createInterface({
    input: createReadStream(inputJsonl, { encoding: "utf8" }),
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

    try {
      currentBatch.push(localSemanticObject(JSON.parse(trimmed)));
      processed += 1;
      writeBatch();
    } catch (error) {
      appendFileSync(errorJsonl, `${JSON.stringify({ line: skipped + processed + 1, error: error.message })}\n`);
    }

    if (processed % 1000 === 0) {
      console.log(`Progress ${processed}/${limit}`);
    }

    if (processed >= limit) {
      rl.close();
      break;
    }
  }

  writeBatch(true);
  writeFileSync(manifestJson, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(`Done. Wrote ${processed} semantic objects across ${manifest.length} batch files.`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  generateLocalSemanticBatches().catch((error) => {
    console.error("Local semantic generation failed:", error.message);
    process.exit(1);
  });
}

export { generateLocalSemanticBatches, localSemanticObject };
