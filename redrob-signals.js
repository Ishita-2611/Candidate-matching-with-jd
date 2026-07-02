const REDROB_SIGNAL_KEYS = [
  "profile_completeness_score",
  "signup_date",
  "last_active_date",
  "open_to_work_flag",
  "profile_views_received_30d",
  "applications_submitted_30d",
  "recruiter_response_rate",
  "avg_response_time_hours",
  "skill_assessment_scores",
  "connection_count",
  "endorsements_received",
  "notice_period_days",
  "expected_salary_range_inr_lpa",
  "preferred_work_mode",
  "willing_to_relocate",
  "github_activity_score",
  "search_appearance_30d",
  "saved_by_recruiters_30d",
  "interview_completion_rate",
  "offer_acceptance_rate",
  "verified_email",
  "verified_phone",
  "linkedin_connected",
];

function clamp01(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
}

function normalizePercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return null;
  }
  return clamp01(number > 1 ? number / 100 : number);
}

function scoreHigherBetter(value, maxUseful) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return null;
  }
  return clamp01(number / maxUseful);
}

function scoreLowerBetter(value, maxBad) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return null;
  }
  return clamp01(1 - number / maxBad);
}

function parseDate(value) {
  if (!value) {
    return null;
  }
  const date = new Date(`${value}T00:00:00Z`);
  return Number.isNaN(date.getTime()) ? null : date;
}

function daysBetween(start, end) {
  return Math.floor((end.getTime() - start.getTime()) / 86_400_000);
}

function average(values) {
  const valid = values.filter((value) => Number.isFinite(value));
  if (valid.length === 0) {
    return 0;
  }
  return valid.reduce((sum, value) => sum + value, 0) / valid.length;
}

function salaryFitScore(candidateSalary, jdSalary = {}) {
  if (!candidateSalary || typeof candidateSalary !== "object") {
    return null;
  }

  const candidateMin = Number(candidateSalary.min);
  const candidateMax = Number(candidateSalary.max);
  const jdMin = Number(jdSalary.min);
  const jdMax = Number(jdSalary.max);

  if (![candidateMin, candidateMax, jdMin, jdMax].every(Number.isFinite) || jdMax < jdMin) {
    return null;
  }

  const overlapMin = Math.max(candidateMin, jdMin);
  const overlapMax = Math.min(candidateMax, jdMax);
  const overlap = Math.max(0, overlapMax - overlapMin);
  return clamp01(overlap / Math.max(1, jdMax - jdMin));
}

function workModeFitScore(candidateWorkMode, jdWorkMode) {
  if (!candidateWorkMode || !jdWorkMode || jdWorkMode === "unknown") {
    return null;
  }

  if (candidateWorkMode === jdWorkMode) {
    return 1;
  }

  if (candidateWorkMode === "flexible") {
    return 0.8;
  }

  if (jdWorkMode === "hybrid" && ["onsite", "remote"].includes(candidateWorkMode)) {
    return 0.5;
  }

  return 0;
}

function redrobSignalScore(redrobSignals = {}, jdMetadata = {}) {
  const assessmentScores = Object.values(redrobSignals.skill_assessment_scores ?? {})
    .map(Number)
    .filter(Number.isFinite)
    .map((score) => clamp01(score / 100));

  const components = {
    profile_completeness: normalizePercent(redrobSignals.profile_completeness_score),
    open_to_work: typeof redrobSignals.open_to_work_flag === "boolean" ? (redrobSignals.open_to_work_flag ? 1 : 0) : null,
    profile_views: scoreHigherBetter(redrobSignals.profile_views_received_30d, 100),
    applications_activity: scoreHigherBetter(redrobSignals.applications_submitted_30d, 20),
    recruiter_response_rate: normalizePercent(redrobSignals.recruiter_response_rate),
    response_speed: scoreLowerBetter(redrobSignals.avg_response_time_hours, 168),
    skill_assessments: assessmentScores.length ? average(assessmentScores) : null,
    network_connections: scoreHigherBetter(redrobSignals.connection_count, 500),
    endorsements: scoreHigherBetter(redrobSignals.endorsements_received, 100),
    notice_period: scoreLowerBetter(
      redrobSignals.notice_period_days,
      Number(jdMetadata.notice_period_days?.hard_cap ?? jdMetadata.notice_period_days?.preferred_max ?? 90)
    ),
    salary_fit: salaryFitScore(redrobSignals.expected_salary_range_inr_lpa, jdMetadata.salary_range_lpa),
    work_mode_fit: workModeFitScore(redrobSignals.preferred_work_mode, jdMetadata.preferred_work_mode),
    relocation_fit:
      jdMetadata.location?.length && redrobSignals.preferred_work_mode !== "remote"
        ? redrobSignals.willing_to_relocate
          ? 1
          : 0.4
        : null,
    github_activity: normalizePercent(redrobSignals.github_activity_score),
    search_appearance: scoreHigherBetter(redrobSignals.search_appearance_30d, 300),
    recruiter_saves: scoreHigherBetter(redrobSignals.saved_by_recruiters_30d, 20),
    interview_completion_rate: normalizePercent(redrobSignals.interview_completion_rate),
    offer_acceptance_rate: normalizePercent(redrobSignals.offer_acceptance_rate),
    verified_email: typeof redrobSignals.verified_email === "boolean" ? (redrobSignals.verified_email ? 1 : 0) : null,
    verified_phone: typeof redrobSignals.verified_phone === "boolean" ? (redrobSignals.verified_phone ? 1 : 0) : null,
    linkedin_connected: typeof redrobSignals.linkedin_connected === "boolean" ? (redrobSignals.linkedin_connected ? 1 : 0) : null,
  };

  const weights = {
    profile_completeness: 0.06,
    open_to_work: 0.08,
    profile_views: 0.03,
    applications_activity: 0.03,
    recruiter_response_rate: 0.08,
    response_speed: 0.04,
    skill_assessments: 0.08,
    network_connections: 0.03,
    endorsements: 0.04,
    notice_period: 0.08,
    salary_fit: 0.08,
    work_mode_fit: 0.08,
    relocation_fit: 0.03,
    github_activity: 0.07,
    search_appearance: 0.03,
    recruiter_saves: 0.05,
    interview_completion_rate: 0.06,
    offer_acceptance_rate: 0.06,
    verified_email: 0.03,
    verified_phone: 0.03,
    linkedin_connected: 0.04,
  };

  let weightedSum = 0;
  let weightSum = 0;
  for (const [key, weight] of Object.entries(weights)) {
    const value = components[key];
    if (Number.isFinite(value)) {
      weightedSum += value * weight;
      weightSum += weight;
    }
  }

  return {
    score: weightSum ? Number((weightedSum / weightSum).toFixed(6)) : 0,
    components,
    used_signals: Object.keys(components).filter((key) => Number.isFinite(components[key])),
    missing_signals: REDROB_SIGNAL_KEYS.filter((key) => !(key in redrobSignals)),
  };
}

function redrobSignalAnomalies(redrobSignals = {}, { currentDate = "2026-06-25" } = {}) {
  const current = parseDate(currentDate) ?? new Date();
  const signupDate = parseDate(redrobSignals.signup_date);
  const lastActiveDate = parseDate(redrobSignals.last_active_date);
  const anomalies = [];

  if (signupDate && signupDate > current) {
    anomalies.push({ name: "signup_date_in_future", severity: 3, detail: redrobSignals.signup_date });
  }
  if (lastActiveDate && lastActiveDate > current) {
    anomalies.push({ name: "last_active_date_in_future", severity: 3, detail: redrobSignals.last_active_date });
  }
  if (signupDate && lastActiveDate && lastActiveDate < signupDate) {
    anomalies.push({
      name: "last_active_before_signup",
      severity: 1,
      detail: `${redrobSignals.signup_date} > ${redrobSignals.last_active_date}`,
    });
  }
  if (signupDate && lastActiveDate && daysBetween(signupDate, lastActiveDate) < 2 && Number(redrobSignals.profile_views_received_30d) > 200) {
    anomalies.push({
      name: "new_profile_extreme_views",
      severity: 2,
      detail: `views=${redrobSignals.profile_views_received_30d}`,
    });
  }
  if (Number(redrobSignals.applications_submitted_30d) > 80) {
    anomalies.push({
      name: "extreme_application_volume",
      severity: 2,
      detail: redrobSignals.applications_submitted_30d,
    });
  }
  if (Number(redrobSignals.saved_by_recruiters_30d) > Number(redrobSignals.profile_views_received_30d ?? 0)) {
    anomalies.push({
      name: "saves_exceed_views",
      severity: 2,
      detail: `saves=${redrobSignals.saved_by_recruiters_30d}, views=${redrobSignals.profile_views_received_30d}`,
    });
  }
  if (
    redrobSignals.verified_email === false &&
    redrobSignals.verified_phone === false &&
    redrobSignals.linkedin_connected === false
  ) {
    anomalies.push({ name: "no_identity_verification", severity: 2, detail: "email, phone, linkedin all false" });
  }

  return anomalies;
}

export {
  REDROB_SIGNAL_KEYS,
  redrobSignalAnomalies,
  redrobSignalScore,
};
