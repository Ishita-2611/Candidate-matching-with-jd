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

function hardConstraintsFromJd(jdSemantic = {}, env = process.env) {
  const metadata = jdSemantic.metadata ?? {};
  const years = metadata.years_of_experience ?? {};

  return {
    minYearsExperience: firstNumber(env.MIN_YEARS_EXPERIENCE, years.minimum, years.preferred),
    locations: splitList(env.LOCATION || metadata.location),
    preferredWorkMode: firstString(env.PREFERRED_WORK_MODE, metadata.preferred_work_mode),
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

  if (constraints.preferredWorkMode) {
    must.push({
      key: "metadata.preferred_work_mode",
      match: {
        value: constraints.preferredWorkMode,
      },
    });
  }

  return must.length ? { must } : undefined;
}

export {
  buildQdrantHardFilter,
  hardConstraintsFromJd,
};
