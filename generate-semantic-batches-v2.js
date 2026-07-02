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

function cleanText(value, maxLength = Number.MAX_SAFE_INTEGER) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  return text.length > maxLength ? `${text.slice(0, Math.max(0, maxLength - 3))}...` : text;
}

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function buildIdentityText(candidate) {
  const profile = candidate.profile ?? {};
  const history = candidate.career_history ?? [];
  const priorTitles = unique(history.map((job) => job.title)).slice(0, 4);
  const priorCompanies = history.slice(0, 3).map((job) => job.company).filter(Boolean);
  const industries = unique(history.map((job) => job.industry)).slice(0, 3);
  const parts = [];

  parts.push(
    `${profile.current_title || "Professional"} with ${profile.years_of_experience ?? 0} years of experience${
      profile.headline ? `. ${profile.headline}` : ""
    }.`
  );

  if (profile.current_company) {
    const context = [profile.current_company_size ? `${profile.current_company_size} employees` : "", profile.current_industry]
      .filter(Boolean)
      .join(", ");
    parts.push(`Currently ${profile.current_title || "working"} at ${profile.current_company}${context ? ` (${context})` : ""}.`);
  }

  if (priorTitles.length > 1) {
    parts.push(`Career roles include ${priorTitles.join(", ")}.`);
  }
  if (priorCompanies.length > 0) {
    parts.push(`Has worked at ${priorCompanies.join(", ")}.`);
  }
  if (industries.length > 0) {
    parts.push(`Industry exposure includes ${industries.join(", ")}.`);
  }
  if (profile.location) {
    parts.push(`Located in ${profile.location}${profile.country ? `, ${profile.country}` : ""}.`);
  }
  if (profile.summary) {
    parts.push(cleanText(profile.summary, 500));
  }

  return parts.join(" ");
}

function buildSkillsText(candidate) {
  const skills = (candidate.skills ?? []).filter((skill) => {
    const duration = Number(skill.duration_months ?? 0);
    const endorsements = Number(skill.endorsements ?? 0);
    const proficiency = String(skill.proficiency ?? "").toLowerCase();

    if (proficiency === "expert" && duration === 0) {
      return false;
    }
    if (proficiency === "beginner" && endorsements === 0) {
      return false;
    }
    return true;
  });

  const byLevel = { expert: [], advanced: [], intermediate: [], beginner: [] };
  for (const skill of skills) {
    const proficiency = String(skill.proficiency ?? "intermediate").toLowerCase();
    const bucket = byLevel[proficiency] ?? byLevel.intermediate;
    const duration = Number(skill.duration_months ?? 0) > 0 ? ` (${skill.duration_months}mo)` : "";
    const endorsements = Number(skill.endorsements ?? 0) > 0 ? `, ${skill.endorsements} endorsements` : "";
    bucket.push(`${skill.name}${duration}${endorsements}`);
  }

  const assessments = Object.entries(candidate.redrob_signals?.skill_assessment_scores ?? {});
  const parts = [];

  if (byLevel.expert.length) {
    parts.push(`Expert skills: ${byLevel.expert.slice(0, 8).join("; ")}.`);
  }
  if (byLevel.advanced.length) {
    parts.push(`Advanced skills: ${byLevel.advanced.slice(0, 8).join("; ")}.`);
  }
  if (byLevel.intermediate.length) {
    parts.push(`Intermediate skills: ${byLevel.intermediate.slice(0, 6).join("; ")}.`);
  }
  if (byLevel.beginner.length) {
    parts.push(`Beginner or emerging skills: ${byLevel.beginner.slice(0, 4).join("; ")}.`);
  }
  if (assessments.length) {
    parts.push(
      `Assessment scores: ${assessments
        .slice(0, 5)
        .map(([skill, score]) => `${skill} ${Math.round(Number(score))}/100`)
        .join("; ")}.`
    );
  }

  return parts.join(" ") || "No documented skills.";
}

function buildExperienceSummaryText(candidate) {
  const profile = candidate.profile ?? {};
  const history = candidate.career_history ?? [];

  if (history.length === 0) {
    return `${profile.years_of_experience ?? 0} years of experience with no documented career history.`;
  }

  const totalMonths = history.reduce((sum, job) => sum + Number(job.duration_months ?? 0), 0);
  const servicesPattern = /(wipro|tcs|infosys|accenture|cognizant|capgemini|hcl|tech mahindra|mindtree)/i;
  const productRoles = history.filter((job) => !servicesPattern.test(job.company ?? "")).length;
  const currentRole = history.find((job) => job.is_current);
  const longestDescription = history
    .map((job) => cleanText(job.description))
    .filter((text) => text.length > 60)
    .sort((a, b) => b.length - a.length)[0];
  const parts = [];

  parts.push(
    `${profile.years_of_experience ?? 0} years of professional experience${
      currentRole ? ` currently as ${currentRole.title} at ${currentRole.company}` : ""
    }. Total documented tenure is about ${Math.round(totalMonths / 12)} years across ${history.length} roles.`
  );

  if (productRoles > 0) {
    parts.push(`${productRoles} of ${history.length} roles appear to be outside pure IT services delivery.`);
  }
  if (longestDescription) {
    parts.push(cleanText(longestDescription, 600));
  }

  return parts.join(" ");
}

function buildExperienceChunks(candidate) {
  return (candidate.career_history ?? []).map((job, index) => {
    const header = `${job.title || "Role"} at ${job.company || "Company"} (${job.industry || "unknown"} industry, ${
      job.duration_months ?? 0
    } months${job.is_current ? ", current" : ""})`;
    const body = cleanText(job.description, 800);

    return {
      id: `chunk_${index + 1}`,
      description: body ? `${header}: ${body}` : header,
      tags: [job.industry, job.title].filter(Boolean),
    };
  });
}

function buildDomainText(candidate) {
  const profile = candidate.profile ?? {};
  const history = candidate.career_history ?? [];
  const industries = unique(history.map((job) => job.industry));
  const companySizes = unique(history.map((job) => job.company_size));
  const fields = (candidate.education ?? []).map((edu) => edu.field_of_study).filter(Boolean);
  const parts = [];

  if (industries.length) {
    parts.push(`Industry domains: ${industries.join(", ")}.`);
  }
  if (companySizes.length) {
    parts.push(`Company sizes experienced: ${companySizes.join(", ")}.`);
  }
  if (profile.current_industry) {
    parts.push(`Current sector: ${profile.current_industry}.`);
  }
  if (fields.length) {
    parts.push(`Academic background: ${fields.join(", ")}.`);
  }

  return parts.join(" ") || "Domain information unavailable.";
}

function buildExecutionStyleText(candidate) {
  const profile = candidate.profile ?? {};
  const text = cleanText(
    [
      profile.summary,
      ...(candidate.career_history ?? []).map((job) => job.description),
    ].join(" ")
  ).toLowerCase();
  const signals = [];

  if (/(shipped|deployed|launched|owned|built and maintained|led the migration|in production)/i.test(text)) {
    signals.push("Has shipped production systems and owned implementation outcomes.");
  }
  if (/(worked closely with pm|optimization target|user engagement|a\/b test|revenue|conversion|product)/i.test(text)) {
    signals.push("Shows product and business context awareness.");
  }
  if (/(xgboost|learning to rank|offline eval|ndcg|mrr|precision|recall|benchmark|model)/i.test(text)) {
    signals.push("Uses applied ML evaluation or model-development language.");
  }
  if (/(scale|pipeline|schema|warehouse|on call|deduplication|watermark|state management|streaming)/i.test(text)) {
    signals.push("Infrastructure-aware profile with scale, data quality, or operational reliability signals.");
  }
  if (/(client deliverable|engagement|consulting|stakeholder management|project delivery)/i.test(text)) {
    signals.push("Career pattern includes consulting or client-facing delivery work.");
  }
  if (Number(profile.years_of_experience ?? 0) >= 5) {
    signals.push(`${profile.years_of_experience} years of experience suggests capacity for autonomous execution.`);
  }

  return signals.join(" ") || "Execution style is not clearly determinable from available profile data.";
}

function buildTrustSignalsText(candidate) {
  const signals = candidate.redrob_signals ?? {};
  const today = new Date("2026-06-25T00:00:00Z");
  const lastActive = signals.last_active_date ? new Date(`${signals.last_active_date}T00:00:00Z`) : null;
  const daysInactive = lastActive && !Number.isNaN(lastActive.getTime()) ? Math.round((today - lastActive) / 86_400_000) : null;
  const parts = [];

  if (signals.open_to_work_flag === true) {
    parts.push("Actively open to work.");
  } else if (signals.open_to_work_flag === false) {
    parts.push("Not currently flagged as open to work.");
  }
  if (daysInactive !== null) {
    if (daysInactive <= 30) {
      parts.push("Active on platform in the last 30 days.");
    } else if (daysInactive <= 90) {
      parts.push(`Last active ${daysInactive} days ago, moderately recent.`);
    } else if (daysInactive <= 180) {
      parts.push(`Last active ${daysInactive} days ago, somewhat stale.`);
    } else {
      parts.push(`Last active ${daysInactive} days ago, likely not actively job-seeking.`);
    }
  }
  if (signals.recruiter_response_rate != null) {
    parts.push(`Recruiter response rate is ${Math.round(Number(signals.recruiter_response_rate) * 100)}%.`);
  }
  if (signals.avg_response_time_hours != null) {
    parts.push(`Average response time is ${signals.avg_response_time_hours} hours.`);
  }
  if (signals.interview_completion_rate != null) {
    parts.push(`Interview completion rate is ${Math.round(Number(signals.interview_completion_rate) * 100)}%.`);
  }
  if (signals.github_activity_score != null) {
    parts.push(`GitHub activity score is ${signals.github_activity_score}/100.`);
  }
  if (signals.notice_period_days != null) {
    parts.push(`Notice period is ${signals.notice_period_days} days.`);
  }
  if (signals.offer_acceptance_rate != null) {
    parts.push(`Historical offer acceptance rate is ${Math.round(Number(signals.offer_acceptance_rate) * 100)}%.`);
  }
  if (signals.profile_completeness_score != null) {
    parts.push(`Profile completeness is ${signals.profile_completeness_score}%.`);
  }
  if (signals.verified_email || signals.verified_phone || signals.linkedin_connected) {
    parts.push(
      `Verification signals: email ${Boolean(signals.verified_email)}, phone ${Boolean(
        signals.verified_phone
      )}, LinkedIn ${Boolean(signals.linkedin_connected)}.`
    );
  }

  return parts.join(" ") || "No behavioral signals available.";
}

function buildDefaultText(semanticObject) {
  const axes = semanticObject.semantic_axes;
  return [
    axes.identity,
    axes.skills,
    axes.experience_summary,
    ...(axes.experience_chunks ?? []).map((chunk) => chunk.description),
    axes.domain,
    axes.execution_style,
    axes.trust_signals,
  ]
    .filter(Boolean)
    .join(" ");
}

function buildSemanticObject(candidate) {
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
      salary_range_lpa: {
        min: salary.min ?? 0,
        max: salary.max ?? 0,
      },
    },
    semantic_axes: {
      identity: buildIdentityText(candidate),
      skills: buildSkillsText(candidate),
      experience_summary: buildExperienceSummaryText(candidate),
      experience_chunks: buildExperienceChunks(candidate),
      domain: buildDomainText(candidate),
      execution_style: buildExecutionStyleText(candidate),
      trust_signals: buildTrustSignalsText(candidate),
    },
  };

  semanticObject.semantic_axes.default = buildDefaultText(semanticObject);
  return semanticObject;
}

function batchFileName(outputDir, batchNo) {
  return `${outputDir}/output_batch${String(batchNo).padStart(4, "0")}.json`;
}

async function generateSemanticBatches() {
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
    console.log(`Wrote ${file} (${currentBatch.length} objects)`);
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
      currentBatch.push(buildSemanticObject(JSON.parse(trimmed)));
      processed += 1;
      writeBatch();
    } catch (error) {
      appendFileSync(errorJsonl, `${JSON.stringify({ line: skipped + processed + 1, error: error.message })}\n`);
    }

    if (processed % 5000 === 0) {
      console.log(`Progress ${processed}/${limit}`);
    }
    if (processed >= limit) {
      rl.close();
      break;
    }
  }

  writeBatch(true);
  writeFileSync(manifestJson, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(`Done. ${processed} semantic objects across ${manifest.length} batch files.`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  generateSemanticBatches().catch((error) => {
    console.error("Failed:", error.message);
    process.exit(1);
  });
}

export {
  buildDefaultText,
  buildDomainText,
  buildExecutionStyleText,
  buildExperienceChunks,
  buildExperienceSummaryText,
  buildIdentityText,
  buildSemanticObject,
  buildSkillsText,
  buildTrustSignalsText,
  generateSemanticBatches,
};
