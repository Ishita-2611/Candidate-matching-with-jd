import OpenAI from "openai";
import { readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

try {
  process.loadEnvFile?.();
} catch (error) {
  if (error.code !== "ENOENT") {
    throw error;
  }
}

const apiKey = process.env.GROQ_API_KEY;

const model =
  process.env.GROQ_MODEL ||
  "meta-llama/llama-4-scout-17b-16e-instruct";

const groqBaseUrl =
  process.env.GROQ_BASE_URL ||
  "https://api.groq.com/openai/v1";

  if (!apiKey) {
    throw new Error("GROQ_API_KEY is required for the JD semantic conversion.");
  }

const openai = new OpenAI({
  apiKey,
  baseURL: groqBaseUrl,
});

function buildPrompt(jobDescriptionText) {
  return `
You are a senior technical recruiter and hiring strategist. Convert the raw job description below into a dense, structured semantic hiring object.

Your job is to infer hiring INTENT, not to extract keywords. Read between the lines: distinguish what the job description literally says from what it actually means. If the job description says it does not care about a specific vendor, tool, or framework name, generalize the requirement to the underlying competency rather than repeating the literal name. Identify disqualifying patterns, ideal-candidate signals, and culture/execution-style cues even when they are described narratively rather than as a bulleted list.

Use factual values inferred only from the job description text below. Where the job description does not state something explicitly or by clear implication, use null, an empty array, or "unknown" rather than inventing a number or fact that is not supported by the text.

Return valid JSON only. Do not include markdown fences, comments, trailing commas, or explanatory text.
Guidelines for field generation

identity.secondary_roles
- Infer semantically similar roles.
- Prefer recruiter search titles rather than literal titles.
- Examples:
  • Applied ML Engineer
  • Search Engineer
  • Ranking Engineer
  • Information Retrieval Engineer
  • ML Platform Engineer

Avoid generic titles unless explicitly implied.
skills.weak_skills

Store optional but beneficial competencies.

Do not interpret these as candidate weaknesses.

Examples:
• LoRA
• PEFT
• Marketplace Experience
• Distributed Systems
skills.noisy_non_relevant_skills

Store technologies that SHOULD NOT dominate retrieval.

Examples:

• LangChain tutorials

• Framework buzzwords

• Vendor-specific keywords

Do not include genuine production competencies.
experience_chunks

Each chunk should represent ONE hiring concept.

Good chunk types:

• production requirement

• responsibility

• ideal signal

• disqualifier

• culture expectation

Do not mix unrelated concepts.
domain.primary_domains

Infer business domains rather than technologies.

Examples:

Talent Intelligence

Search

Information Retrieval

Recommendation Systems

Marketplace

Recruiting Platforms


execution_style
Infer these from engineering culture.
Use
shipping_bias
ownership
product_mindset
research_bias
ambiguity_tolerance
system_design_depth
Job Description:
${jobDescriptionText}

Produce ONLY a valid JSON object following exactly the schema below.

The schema defines the output structure only.

Do not copy placeholder text.

Replace every placeholder with values inferred from the job description.
{
  "metadata": {
    "years_of_experience": {
      "minimum": <number or null>,
      "preferred": <number or null>,
      "flexible": <boolean>
    },
    "location": ["<acceptable city or region>"],
    "country": "<string or null>",
    "preferred_work_mode": "<remote|hybrid|onsite|unknown>",
    "notice_period_days": {
      "preferred_max": <number or null>,
      "hard_cap": <number or null>
    },
    "salary_range_lpa": {
      "min": <number or null>,
      "max": <number or null>
    },
    "visa_sponsorship": <boolean>
  },
  "semantic_axes": {
    "identity": {
      "role_family": "<primary role family>",
      "secondary_roles": ["<adjacent acceptable title>"],
      "seniority": "<seniority>",
      "career_transition": {
        "from": "<the kind of background that fits this role>",
        "to": "<what they would own in this role>"
      }
    },
    "skills": {
      "core_production_skills": ["<skill required for production-level competence>"],
      "ml_skills": ["<required or preferred ML-specific skill>"],
      "ml_infra_skills": ["<infrastructure/systems skill required>"],
      "weak_skills": ["<nice-to-have, not heavily weighted>"],
      "noisy_non_relevant_skills": ["<specific tool/vendor/keyword the JD explicitly says NOT to over-weight>"]
    },
    "experience_summary": {
      "system_types": ["<type of system this role owns, e.g. ranking system, retrieval system>"],
      "scale": {
        "daily_data_processed": "<string or 'unknown'>",
        "source_systems": <number or 0>,
        "realtime_systems": <boolean>
      },
      "production_maturity": "<low|moderate|high>",
      "ml_maturity": "<low|moderate-low|moderate|high>"
    },
    "experience_chunks": [
      {
        "id": "chunk_1",
        "description": "<dense factual requirement, disqualifier, or ideal-candidate signal>",
        "tags": ["<tag, e.g. disqualifier, ideal_signal, responsibility>"]
      }
    ],
    "domain": {
      "primary_domains": ["<domain this role requires>"],
      "secondary_domains": ["<domain that is a plus>"],
      "missing_domains": ["<domain explicitly called out as NOT required or not a fit>"]
    },
    "execution_style": {
      "shipping_bias": "<low|medium|high>",
      "product_mindset": "<low|medium|high>",
      "research_bias": "<low|medium|high>",
      "ambiguity_tolerance": "<low|medium|high>",
      "ownership": "<low|medium|high>",
      "system_design_depth": "<low|medium|high>"
    },
    "trust_signals": {
      "requires_active_platform_presence": <boolean>,
      "weights": {
        "recruiter_response_rate": "<high|medium|low|not_mentioned>",
        "interview_completion_rate": "<high|medium|low|not_mentioned>",
        "offer_acceptance_rate": "<high|medium|low|not_mentioned>",
        "github_score": "<high|medium|low|not_mentioned>",
        "profile_completeness": "<high|medium|low|not_mentioned>"
      },
      "notes": "<short note on what behavioral signal this JD explicitly cares about, or empty string>"
    }
  }
}
`.trim();
}

const REQUIRED_METADATA_KEYS = [
  "years_of_experience",
  "location",
  "country",
  "preferred_work_mode",
  "notice_period_days",
  "salary_range_lpa",
  "visa_sponsorship",
];

const REQUIRED_SEMANTIC_AXES_KEYS = [
  "identity",
  "skills",
  "experience_summary",
  "experience_chunks",
  "domain",
  "execution_style",
  "trust_signals",
];

function assertValidSemanticObject(parsed) {
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Parsed response is not a JSON object.");
  }

  for (const key of ["metadata", "semantic_axes"]) {
    if (!parsed[key] || typeof parsed[key] !== "object") {
      throw new Error(`Missing or invalid top-level key: "${key}"`);
    }
  }

  const missingMetadataKeys = REQUIRED_METADATA_KEYS.filter((key) => !(key in parsed.metadata));
  if (missingMetadataKeys.length > 0) {
    throw new Error(`Missing metadata key(s): ${missingMetadataKeys.join(", ")}`);
  }

  const missingAxesKeys = REQUIRED_SEMANTIC_AXES_KEYS.filter((key) => !(key in parsed.semantic_axes));
  if (missingAxesKeys.length > 0) {
    throw new Error(`Missing semantic_axes key(s): ${missingAxesKeys.join(", ")}`);
  }

  if (!Array.isArray(parsed.metadata.location)) {
    throw new Error('"metadata.location" must be an array.');
  }

  if (!Array.isArray(parsed.semantic_axes.experience_chunks)) {
    throw new Error('"semantic_axes.experience_chunks" must be an array.');
  }

  return parsed;
}

async function convertJobDescriptionToSemanticObject(jobDescriptionText) {
  const response = await openai.chat.completions.create({
    model,
    temperature: 0.2,

    response_format: {
      type: "json_object",
    },

    messages: [
      {
        role: "user",
        content: buildPrompt(jobDescriptionText),
      },
    ],
  });

  const raw =
    (response.choices[0]?.message?.content || "").trim();

  let parsed;

  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error(`Groq returned invalid JSON:\n${raw}`);
  }

  return assertValidSemanticObject(parsed);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryable(error) {
  const status = error?.status;
  if (status === 408 || status === 409 || status === 429 || (status >= 500 && status < 600)) {
    return true;
  }
  const message = String(error?.message || "");
  return (
    /\b(429|500|502|503|504)\b/.test(message) ||
    /RESOURCE_EXHAUSTED|UNAVAILABLE|INTERNAL|ECONNRESET|ETIMEDOUT|fetch failed/i.test(message)
  );
}

async function convertWithRetry(jobDescriptionText, attempts = 4) {
  let lastError;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await convertJobDescriptionToSemanticObject(jobDescriptionText);
    } catch (error) {
      lastError = error;
      if (attempt >= attempts || !isRetryable(error)) {
        break;
      }

      const delay = 1000 * 2 ** (attempt - 1) + Math.floor(Math.random() * 500);
      console.warn(`[jd-parser] Attempt ${attempt} failed (${error.message}). Retrying in ${delay}ms...`);
      await sleep(delay);
    }
  }

  throw lastError;
}

function readJobDescriptionFile(filePath) {
  try {
    return readFileSync(filePath, "utf8");
  } catch (error) {
    if (error.code === "ENOENT") {
      throw new Error(`Job description file not found: ${filePath}`);
    }
    throw error;
  }
}

async function main() {
  try {
    const inputFile = process.env.JD_INPUT_FILE || "job_description.md";
    const outputFile = process.env.JD_OUTPUT_FILE || "jd-semantic.json";

    console.log("Starting Groq LLM job description semantic conversion...\n");
    console.log(`Input file: ${inputFile}`);
    console.log(`Model: ${model}`);
    console.log(`Groq endpoint: ${groqBaseUrl}`);
    console.log(`Output file: ${outputFile}`);

    const jobDescriptionText = readJobDescriptionFile(inputFile);

    if (!jobDescriptionText.trim()) {
      throw new Error(`Job description file "${inputFile}" is empty.`);
    }

    const semanticObject = await convertWithRetry(jobDescriptionText);

    writeFileSync(outputFile, `${JSON.stringify(semanticObject, null, 2)}\n`);

    console.log(`\nSaved semantic hiring object to ${outputFile}`);
  } catch (error) {
    console.error("Error during JD conversion:", error.message);
    process.exit(1);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}

export {
  buildPrompt,
  convertJobDescriptionToSemanticObject,
  convertWithRetry,
  assertValidSemanticObject,
  readJobDescriptionFile,
};