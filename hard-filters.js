function normalizeString(value) {
  return typeof value === "string" ? value.trim() : "";
}

function parseNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function splitList(value) {
  if (Array.isArray(value)) {
    return value.map(normalizeString).filter(Boolean);
  }

  return normalizeString(value)
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function firstNumber(...values) {
  for (const value of values) {
    const parsed = parseNumber(value);
    if (parsed !== null) {
      return parsed;
    }
  }
  return null;
}

function firstString(...values) {
  for (const value of values) {
    const parsed = normalizeString(value);
    if (parsed) {
      return parsed;
    }
  }
  return "";
}

function parseBoolean(value) {
  if (typeof value === "boolean") {
    return value;
  }

  const normalized = normalizeString(value).toLowerCase();
  if (["true", "1", "yes", "y"].includes(normalized)) {
    return true;
  }
  if (["false", "0", "no", "n"].includes(normalized)) {
    return false;
  }
  return null;
}

function hardConstraintsFromJd(jdSemantic = {}, env = process.env) {
  const metadata = jdSemantic.metadata ?? {};
  const years = metadata.years_of_experience ?? {};
  const notice = metadata.notice_period_days ?? {};
  const salary = metadata.salary_range_lpa ?? {};

  return {
    minYearsExperience: firstNumber(env.MIN_YEARS_EXPERIENCE, years.minimum, years.preferred),
    locations: splitList(env.LOCATION || metadata.location),
    country: firstString(env.COUNTRY, metadata.country),
    preferredWorkMode: firstString(env.PREFERRED_WORK_MODE, metadata.preferred_work_mode),
    maxNoticePeriodDays: firstNumber(env.MAX_NOTICE_PERIOD_DAYS, notice.hard_cap, notice.preferred_max),
    salaryMinLpa: firstNumber(env.SALARY_MIN_LPA, salary.min),
    salaryMaxLpa: firstNumber(env.SALARY_MAX_LPA, salary.max),
    requireOpenToWork: parseBoolean(env.REQUIRE_OPEN_TO_WORK),
  };
}

function buildQdrantHardFilter(constraints) {
  const must = [];

  if (constraints.minYearsExperience !== null && constraints.minYearsExperience !== undefined) {
    must.push({
      key: "metadata.years_of_experience",
      range: {
        gte: constraints.minYearsExperience,
      },
    });
  }

  if (constraints.locations?.length) {
    must.push({
      should: constraints.locations.map((location) => ({
        key: "metadata.location",
        match: {
          value: location,
        },
      })),
    });
  }

  if (constraints.country) {
    must.push({
      key: "metadata.country",
      match: {
        value: constraints.country,
      },
    });
  }

  if (constraints.preferredWorkMode) {
    must.push({
      key: "metadata.preferred_work_mode",
      match: {
        value: constraints.preferredWorkMode,
      },
    });
  }

  if (constraints.maxNoticePeriodDays !== null && constraints.maxNoticePeriodDays !== undefined) {
    must.push({
      key: "metadata.notice_period_days",
      range: {
        lte: constraints.maxNoticePeriodDays,
      },
    });
  }

  if (constraints.salaryMinLpa !== null && constraints.salaryMinLpa !== undefined) {
    must.push({
      key: "metadata.salary_range_lpa.max",
      range: {
        gte: constraints.salaryMinLpa,
      },
    });
  }

  if (constraints.salaryMaxLpa !== null && constraints.salaryMaxLpa !== undefined) {
    must.push({
      key: "metadata.salary_range_lpa.min",
      range: {
        lte: constraints.salaryMaxLpa,
      },
    });
  }

  if (constraints.requireOpenToWork !== null && constraints.requireOpenToWork !== undefined) {
    must.push({
      key: "metadata.open_to_work",
      match: {
        value: constraints.requireOpenToWork,
      },
    });
  }

  return must.length ? { must } : undefined;
}

export {
  buildQdrantHardFilter,
  hardConstraintsFromJd,
};
