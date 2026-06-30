import { createReadStream, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import readline from "node:readline";
import { redrobSignalAnomalies } from "./redrob-signals.js";

const DEFAULT_CURRENT_DATE = "2026-06-25";

function parsePositiveInt(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function parseDate(value) {
  if (!value) {
    return null;
  }

  const date = new Date(`${value}T00:00:00Z`);
  return Number.isNaN(date.getTime()) ? null : date;
}

function monthDiff(start, end) {
  return (
    (end.getUTCFullYear() - start.getUTCFullYear()) * 12 +
    (end.getUTCMonth() - start.getUTCMonth()) +
    (end.getUTCDate() >= start.getUTCDate() ? 0 : -1)
  );
}

function flag(name, severity, detail) {
  return { name, severity, detail };
}

function detectHoneypotSignals(candidate, { currentDate = DEFAULT_CURRENT_DATE } = {}) {
  const current = parseDate(currentDate) ?? new Date();
  const profile = candidate.profile ?? {};
  const skills = candidate.skills ?? [];
  const careerHistory = candidate.career_history ?? [];
  const signals = candidate.redrob_signals ?? {};
  const flags = [];
  const yoe = Number(profile.years_of_experience ?? 0);
  const yoeMonths = yoe * 12;

  const advancedOrExpertZeroMonths = skills.filter(
    (skill) =>
      Number(skill.duration_months ?? 0) <= 0 &&
      /advanced|expert/i.test(String(skill.proficiency ?? ""))
  );
  if (advancedOrExpertZeroMonths.length > 0) {
    flags.push(
      flag(
        "advanced_or_expert_skill_zero_months",
        3,
        advancedOrExpertZeroMonths.map((skill) => skill.name).slice(0, 10)
      )
    );
  }

  const advancedUnderSixMonths = skills.filter(
    (skill) =>
      Number(skill.duration_months ?? 0) < 6 &&
      /advanced|expert/i.test(String(skill.proficiency ?? ""))
  );
  if (advancedUnderSixMonths.length >= 5) {
    flags.push(
      flag(
        "five_or_more_advanced_skills_under_6_months",
        2,
        advancedUnderSixMonths.map((skill) => `${skill.name}:${skill.duration_months}m`).slice(0, 10)
      )
    );
  }

  const impossibleSkillDurations = skills.filter(
    (skill) => Number(skill.duration_months ?? 0) > yoeMonths + 12
  );
  if (impossibleSkillDurations.length >= 5) {
    flags.push(
      flag(
        "many_skill_durations_exceed_total_experience",
        2,
        impossibleSkillDurations.map((skill) => `${skill.name}:${skill.duration_months}m`).slice(0, 10)
      )
    );
  }

  let statedCareerMonths = 0;
  let durationDateMismatches = 0;
  const durationMismatchDetails = [];
  const badDateRanges = [];
  const futureJobs = [];

  for (const job of careerHistory) {
    const start = parseDate(job.start_date);
    const end = parseDate(job.end_date) ?? current;
    const statedMonths = Number(job.duration_months ?? 0);
    statedCareerMonths += statedMonths;

    if (!start || !end) {
      continue;
    }

    if (start > current) {
      futureJobs.push(`${job.company}:${job.start_date}`);
    }

    if (end < start) {
      badDateRanges.push(`${job.company}:${job.start_date}-${job.end_date}`);
      continue;
    }

    const calculatedMonths = Math.max(0, monthDiff(start, end));
    if (Math.abs(calculatedMonths - statedMonths) > 6) {
      durationDateMismatches += 1;
      durationMismatchDetails.push(`${job.company}:stated=${statedMonths},calculated=${calculatedMonths}`);
    }
  }

  if (durationDateMismatches > 0) {
    flags.push(flag("job_duration_date_mismatch", 3, durationMismatchDetails.slice(0, 5)));
  }

  if (badDateRanges.length > 0) {
    flags.push(flag("job_end_before_start", 4, badDateRanges.slice(0, 5)));
  }

  if (futureJobs.length > 0) {
    flags.push(flag("job_starts_in_future", 4, futureJobs.slice(0, 5)));
  }

  if (careerHistory.length > 0 && statedCareerMonths > yoeMonths + 36) {
    flags.push(
      flag("career_months_far_exceed_profile_experience", 3, {
        stated_career_months: statedCareerMonths,
        profile_experience_months: Number(yoeMonths.toFixed(1)),
      })
    );
  }

  const currentJobs = careerHistory.filter((job) => job.is_current || !job.end_date);
  if (currentJobs.length > 1) {
    flags.push(flag("multiple_current_jobs", 2, currentJobs.map((job) => job.company).slice(0, 5)));
  }

  for (const anomaly of redrobSignalAnomalies(signals, { currentDate })) {
    flags.push(flag(anomaly.name, anomaly.severity, anomaly.detail));
  }

  const skillByName = new Map(skills.map((skill) => [String(skill.name).toLowerCase(), skill]));
  const highAssessmentLowDuration = [];
  for (const [skillName, score] of Object.entries(signals.skill_assessment_scores ?? {})) {
    const skill = skillByName.get(String(skillName).toLowerCase());
    if (skill && Number(score) > 85 && Number(skill.duration_months ?? 0) < 3) {
      highAssessmentLowDuration.push(`${skillName}:score=${score},duration=${skill.duration_months}m`);
    }
  }
  if (highAssessmentLowDuration.length > 0) {
    flags.push(flag("high_assessment_with_under_3_months_skill_use", 3, highAssessmentLowDuration.slice(0, 10)));
  }

  const severity = flags.reduce((sum, item) => sum + item.severity, 0);
  const honeypotProbability = Math.min(1, severity / 8);
  const riskLevel = severity >= 5 ? "high" : severity >= 3 ? "medium" : severity > 0 ? "low" : "none";

  return {
    candidate_id: candidate.candidate_id,
    severity,
    honeypot_probability: Number(honeypotProbability.toFixed(3)),
    risk_level: riskLevel,
    flags,
  };
}

async function detectHoneypotsInFile({
  inputFile,
  outputFile,
  minSeverity,
  limit,
  currentDate,
}) {
  const rl = readline.createInterface({
    input: createReadStream(inputFile, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });

  const results = [];
  const counts = {};
  let scanned = 0;

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }

    const candidate = JSON.parse(trimmed);
    const analysis = detectHoneypotSignals(candidate, { currentDate });
    scanned += 1;

    for (const item of analysis.flags) {
      counts[item.name] = (counts[item.name] ?? 0) + 1;
    }

    if (analysis.severity >= minSeverity) {
      results.push(analysis);
    }

    if (scanned >= limit) {
      break;
    }
  }

  results.sort((a, b) => b.severity - a.severity || a.candidate_id.localeCompare(b.candidate_id));

  const output = {
    input_file: inputFile,
    scanned,
    min_severity: minSeverity,
    flagged_count: results.length,
    flag_counts: counts,
    results,
  };

  writeFileSync(outputFile, `${JSON.stringify(output, null, 2)}\n`);
  return output;
}

async function main() {
  try {
    const inputFile = process.env.CANDIDATES_JSONL || "candidates.jsonl";
    const outputFile = process.env.HONEYPOT_OUTPUT_JSON || "honeypot_candidates.json";
    const minSeverity = parsePositiveInt(process.env.HONEYPOT_MIN_SEVERITY, 5);
    const limit = parsePositiveInt(process.env.HONEYPOT_SCAN_LIMIT, Number.MAX_SAFE_INTEGER);
    const currentDate = process.env.HONEYPOT_CURRENT_DATE || DEFAULT_CURRENT_DATE;

    const output = await detectHoneypotsInFile({
      inputFile,
      outputFile,
      minSeverity,
      limit,
      currentDate,
    });

    console.log(
      JSON.stringify(
        {
          input_file: output.input_file,
          scanned: output.scanned,
          min_severity: output.min_severity,
          flagged_count: output.flagged_count,
          output_file: outputFile,
          flag_counts: output.flag_counts,
        },
        null,
        2
      )
    );
  } catch (error) {
    console.error("Honeypot detection failed:", error.message);
    process.exit(1);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}

export {
  detectHoneypotSignals,
  detectHoneypotsInFile,
};
