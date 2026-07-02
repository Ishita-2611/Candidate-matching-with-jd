import argparse
import csv
import json
import math
import re
from datetime import date, datetime


CURRENT_DATE = date(2026, 6, 25)
JD_SKILL_TERMS = [
    "python",
    "embedding",
    "retrieval",
    "ranking",
    "recommendation",
    "vector",
    "qdrant",
    "milvus",
    "faiss",
    "elasticsearch",
    "opensearch",
    "llm",
    "fine-tuning",
    "lora",
    "evaluation",
    "ndcg",
    "mrr",
    "map",
    "a/b",
]
ROLE_TERMS = ["ml", "machine learning", "ai", "data engineer", "backend", "platform", "search", "ranking", "retrieval"]
SERVICE_COMPANIES = ["wipro", "tcs", "infosys", "accenture", "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree"]
PREFERRED_LOCATIONS = ["pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "bangalore", "bengaluru"]
DEFAULT_ALLOWED_WORK_MODES = {"onsite", "hybrid", "flexible"}


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def parse_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def month_diff(start, end):
    return (end.year - start.year) * 12 + (end.month - start.month) - (1 if end.day < start.day else 0)


def normalize_percent(value):
    if value is None:
        return None
    value = float(value)
    return clamp01(value / 100.0 if value > 1 else value)


def lower_better(value, max_bad):
    if value is None:
        return None
    return clamp01(1.0 - float(value) / max_bad)


def higher_better(value, max_good):
    if value is None:
        return None
    return clamp01(float(value) / max_good)


def average(values):
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def salary_overlap_score(candidate_salary, jd_min=18.0, jd_max=60.0):
    if not isinstance(candidate_salary, dict):
        return None
    cand_min = candidate_salary.get("min")
    cand_max = candidate_salary.get("max")
    if cand_min is None or cand_max is None:
        return None
    cand_min = float(cand_min)
    cand_max = float(cand_max)
    overlap = max(0.0, min(cand_max, jd_max) - max(cand_min, jd_min))
    return clamp01(overlap / max(1.0, jd_max - jd_min))


def work_mode_fit_score(work_mode):
    mode = str(work_mode or "").lower()
    if mode in {"hybrid", "onsite"}:
        return 1.0
    if mode == "flexible":
        return 0.85
    if mode == "remote":
        return 0.45
    return None


def load_retrieval_candidates(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {row["candidate_id"]: float(row["retrieval_score"]) for row in data["candidates"]}


def candidate_text(candidate):
    profile = candidate.get("profile", {})
    skills = " ".join(skill.get("name", "") for skill in candidate.get("skills", []))
    jobs = " ".join(
        " ".join(
            [
                job.get("title", ""),
                job.get("company", ""),
                job.get("industry", ""),
                job.get("description", ""),
            ]
        )
        for job in candidate.get("career_history", [])
    )
    return " ".join(
        [
            profile.get("current_title", ""),
            profile.get("headline", ""),
            profile.get("summary", ""),
            skills,
            jobs,
        ]
    ).lower()


def honeypot_analysis(candidate):
    profile = candidate.get("profile", {})
    redrob = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])
    yoe_months = float(profile.get("years_of_experience") or 0) * 12
    flags = []

    zero_expert = [
        skill.get("name")
        for skill in skills
        if str(skill.get("proficiency", "")).lower() in {"advanced", "expert"}
        and float(skill.get("duration_months") or 0) <= 0
    ]
    if zero_expert:
        flags.append(("advanced_or_expert_skill_zero_months", 3, zero_expert[:10]))

    short_advanced = [
        skill.get("name")
        for skill in skills
        if str(skill.get("proficiency", "")).lower() in {"advanced", "expert"}
        and float(skill.get("duration_months") or 0) < 6
    ]
    if len(short_advanced) >= 5:
        flags.append(("five_or_more_advanced_skills_under_6_months", 2, short_advanced[:10]))

    impossible_skill_durations = [
        f"{skill.get('name')}:{skill.get('duration_months')}m"
        for skill in skills
        if float(skill.get("duration_months") or 0) > yoe_months + 12
    ]
    if len(impossible_skill_durations) >= 5:
        flags.append(("many_skill_durations_exceed_total_experience", 2, impossible_skill_durations[:10]))

    stated_months = 0
    mismatches = []
    for job in history:
        stated = float(job.get("duration_months") or 0)
        stated_months += stated
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date")) or CURRENT_DATE
        if not start or not end:
            continue
        if end < start:
            flags.append(("job_end_before_start", 4, job.get("company")))
            continue
        calculated = max(0, month_diff(start, end))
        if abs(calculated - stated) > 6:
            mismatches.append(job.get("company"))
    if mismatches:
        flags.append(("job_duration_date_mismatch", 3, mismatches[:5]))
    if history and stated_months > yoe_months + 36:
        flags.append(("career_months_far_exceed_profile_experience", 3, int(stated_months)))

    signup = parse_date(redrob.get("signup_date"))
    active = parse_date(redrob.get("last_active_date"))
    if signup and active and active < signup:
        flags.append(("last_active_before_signup", 1, redrob.get("last_active_date")))
    if float(redrob.get("saved_by_recruiters_30d") or 0) > float(redrob.get("profile_views_received_30d") or 0):
        flags.append(("saves_exceed_views", 2, "saves exceed views"))
    if float(redrob.get("applications_submitted_30d") or 0) > 80:
        flags.append(("extreme_application_volume", 2, redrob.get("applications_submitted_30d")))
    if signup and signup > CURRENT_DATE:
        flags.append(("signup_date_in_future", 3, redrob.get("signup_date")))
    if active and active > CURRENT_DATE:
        flags.append(("last_active_date_in_future", 3, redrob.get("last_active_date")))
    if not redrob.get("verified_email") and not redrob.get("verified_phone") and not redrob.get("linkedin_connected"):
        flags.append(("no_identity_verification", 2, "email, phone, linkedin all false"))

    return {"severity": sum(flag[1] for flag in flags), "flags": flags}


def behavioral_score(redrob):
    last_active = parse_date(redrob.get("last_active_date"))
    days_inactive = (CURRENT_DATE - last_active).days if last_active else 180
    assessment_scores = [
        normalize_percent(score)
        for score in (redrob.get("skill_assessment_scores") or {}).values()
    ]
    components = {
        "profile_completeness": normalize_percent(redrob.get("profile_completeness_score")),
        "open_to_work": 1.0 if redrob.get("open_to_work_flag") else 0.0,
        "activity_recency": lower_better(days_inactive, 180),
        "profile_views": higher_better(redrob.get("profile_views_received_30d"), 100),
        "application_activity": higher_better(redrob.get("applications_submitted_30d"), 20),
        "recruiter_response": normalize_percent(redrob.get("recruiter_response_rate")),
        "response_speed": lower_better(redrob.get("avg_response_time_hours"), 168),
        "skill_assessments": average(assessment_scores),
        "network_connections": higher_better(redrob.get("connection_count"), 500),
        "endorsements": higher_better(redrob.get("endorsements_received"), 100),
        "interview_completion": normalize_percent(redrob.get("interview_completion_rate")),
        "notice_period": 1.0
        if float(redrob.get("notice_period_days") or 999) <= 30
        else 0.35
        if float(redrob.get("notice_period_days") or 999) <= 60
        else 0.0,
        "salary_fit": salary_overlap_score(redrob.get("expected_salary_range_inr_lpa")),
        "work_mode_fit": work_mode_fit_score(redrob.get("preferred_work_mode")),
        "relocation_fit": 1.0 if redrob.get("willing_to_relocate") else 0.4,
        "github_activity": normalize_percent(redrob.get("github_activity_score")),
        "search_appearance": higher_better(redrob.get("search_appearance_30d"), 300),
        "profile_completeness": normalize_percent(redrob.get("profile_completeness_score")),
        "offer_acceptance": normalize_percent(redrob.get("offer_acceptance_rate")),
        "verified_identity": sum(bool(redrob.get(key)) for key in ["verified_email", "verified_phone", "linkedin_connected"]) / 3.0,
        "recruiter_saves": higher_better(redrob.get("saved_by_recruiters_30d"), 20),
    }
    weights = {
        "profile_completeness": 0.05,
        "open_to_work": 0.08,
        "activity_recency": 0.10,
        "profile_views": 0.03,
        "application_activity": 0.03,
        "recruiter_response": 0.09,
        "response_speed": 0.05,
        "skill_assessments": 0.08,
        "network_connections": 0.03,
        "endorsements": 0.04,
        "interview_completion": 0.07,
        "notice_period": 0.09,
        "salary_fit": 0.07,
        "work_mode_fit": 0.06,
        "relocation_fit": 0.03,
        "github_activity": 0.06,
        "search_appearance": 0.03,
        "offer_acceptance": 0.05,
        "verified_identity": 0.03,
        "recruiter_saves": 0.03,
    }
    total = 0.0
    used = 0.0
    for key, weight in weights.items():
        value = components.get(key)
        if value is not None:
            total += clamp01(value) * weight
            used += weight
    return total / used if used else 0.0


def passes_hard_filters(candidate, min_years_experience=4.0, allowed_work_modes=None):
    profile = candidate.get("profile", {})
    redrob = candidate.get("redrob_signals", {})
    country = str(profile.get("country", "")).lower()
    yoe = float(profile.get("years_of_experience") or 0)
    work_mode = str(redrob.get("preferred_work_mode", "")).lower()
    if country != "india":
        return False
    if yoe < min_years_experience:
        return False
    if allowed_work_modes and work_mode and work_mode not in allowed_work_modes:
        return False
    return True


def profile_score(candidate, text):
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    yoe = float(profile.get("years_of_experience") or 0)
    yoe_score = 0.0 if yoe < 3 else 0.55 if yoe < 5 else min(1.0, 0.75 + min(yoe - 5, 5) / 20)
    role_score = sum(1 for term in ROLE_TERMS if term in text) / len(ROLE_TERMS)
    skill_score = sum(1 for term in JD_SKILL_TERMS if term in text) / len(JD_SKILL_TERMS)
    product_roles = sum(
        1
        for job in history
        if not any(company in job.get("company", "").lower() for company in SERVICE_COMPANIES)
    )
    product_score = min(1.0, product_roles / max(1, len(history)))
    production_score = 1.0 if re.search(r"\b(shipped|deployed|production|launched|owned|built|scaled)\b", text) else 0.0
    eval_score = 1.0 if re.search(r"\b(ndcg|mrr|map|a/b|ab test|evaluation|experiment|metrics?)\b", text) else 0.0
    return (
        0.22 * yoe_score
        + 0.18 * role_score
        + 0.24 * skill_score
        + 0.12 * product_score
        + 0.14 * production_score
        + 0.10 * eval_score
    )


def location_score(candidate):
    profile = candidate.get("profile", {})
    redrob = candidate.get("redrob_signals", {})
    country = str(profile.get("country", "")).lower()
    location = str(profile.get("location", "")).lower()
    work_mode = str(redrob.get("preferred_work_mode", "")).lower()
    if country == "india" and any(city in location for city in PREFERRED_LOCATIONS):
        base = 1.0
    elif country == "india":
        base = 0.78
    else:
        base = 0.28
    if work_mode in {"hybrid", "flexible"}:
        base += 0.08
    elif work_mode == "remote":
        base -= 0.06
    return clamp01(base)


def experience_fit(candidate):
    yoe = float(candidate.get("profile", {}).get("years_of_experience") or 0)
    if 5 <= yoe <= 9:
        return 1.0
    if 4 <= yoe < 5:
        return 0.72
    if 9 < yoe <= 12:
        return 0.68
    if 3 <= yoe < 4:
        return 0.42
    if 12 < yoe <= 15:
        return 0.36
    return 0.18


def disqualifier_penalty(candidate, text):
    penalty = 0.0
    title = candidate.get("profile", {}).get("current_title", "").lower()
    if any(term in title for term in ["marketing manager", "hr manager", "graphic designer", "content writer"]):
        penalty += 0.20
    if any(term in title for term in ["computer vision", "speech", "robotics"]):
        penalty += 0.16
    if "langchain" in text and not re.search(r"\b(retrieval|ranking|recommendation|search|ml|machine learning)\b", text):
        penalty += 0.10
    if re.search(r"\b(computer vision|robotics|speech)\b", text) and not re.search(r"\b(nlp|retrieval|ranking|search)\b", text):
        penalty += 0.08
    return min(0.35, penalty)


def final_score(retrieval_score, candidate, info, text):
    semantic = clamp01((retrieval_score + 1.0) / 2.0)
    notice = float(candidate.get("redrob_signals", {}).get("notice_period_days") or 0)
    notice_penalty = 0.12 if notice > 90 else 0.08 if notice > 60 else 0.03 if notice > 30 else 0.0
    score = (
        0.32 * semantic
        + 0.30 * info["profile_score"]
        + 0.18 * info["behavioral_score"]
        + 0.12 * location_score(candidate)
        + 0.08 * experience_fit(candidate)
        - disqualifier_penalty(candidate, text)
        - notice_penalty
        - min(0.18, 0.035 * info["honeypot"]["severity"])
    )
    return clamp01(score)


def matched_terms(text, terms, limit=4):
    matches = []
    for term in terms:
        if term in text and term not in matches:
            matches.append(term)
        if len(matches) >= limit:
            break
    return matches


def reasoning(candidate, info, text, rank):
    profile = candidate.get("profile", {})
    redrob = candidate.get("redrob_signals", {})
    title = profile.get("current_title") or "Candidate"
    yoe = profile.get("years_of_experience", "unknown")
    location = profile.get("location") or "unknown location"
    skills = matched_terms(text, JD_SKILL_TERMS, 3)
    skills_text = ", ".join(skills) if skills else "adjacent ML/search signals"
    response = redrob.get("recruiter_response_rate", 0)
    notice = redrob.get("notice_period_days", 0)
    concern = ""
    if notice and float(notice) > 60:
        concern = f" Concern: {int(float(notice))}-day notice period."
    elif str(profile.get("country", "")).lower() != "india":
        concern = " Concern: outside India, so relocation fit is weaker."
    elif experience_fit(candidate) < 0.7:
        concern = " Concern: experience is outside the JD's ideal 5-9 year band."
    elif info["honeypot"]["severity"]:
        concern = f" Minor consistency risk severity {info['honeypot']['severity']}."
    elif rank > 75:
        concern = " Included near cutoff because fit is weaker than top profiles."
    return (
        f"{title} with {yoe} yrs in {location}; matches JD through {skills_text}; "
        f"response rate {float(response or 0):.2f}.{concern}"
    )


def read_candidate_subset(candidates_path, candidate_ids):
    found = {}
    with open(candidates_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            candidate = json.loads(line)
            candidate_id = candidate.get("candidate_id")
            if candidate_id in candidate_ids:
                found[candidate_id] = candidate
                if len(found) == len(candidate_ids):
                    break
    return found


def write_submission(rows, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="candidates.jsonl")
    parser.add_argument("--retrieval", default="retrieval_candidates.json")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--honeypot-severity-threshold", type=float, default=3.0)
    parser.add_argument("--min-years-experience", type=float, default=4.0)
    parser.add_argument("--allow-remote", action="store_true")
    args = parser.parse_args()

    retrieval = load_retrieval_candidates(args.retrieval)
    candidates = read_candidate_subset(args.candidates, set(retrieval))
    ranked = []
    allowed_work_modes = None if args.allow_remote else DEFAULT_ALLOWED_WORK_MODES
    for candidate_id, retrieval_score in retrieval.items():
        candidate = candidates.get(candidate_id)
        if not candidate:
            continue
        if not passes_hard_filters(candidate, args.min_years_experience, allowed_work_modes):
            continue
        text = candidate_text(candidate)
        info = {
            "behavioral_score": behavioral_score(candidate.get("redrob_signals", {})),
            "profile_score": profile_score(candidate, text),
            "honeypot": honeypot_analysis(candidate),
        }
        if info["honeypot"]["severity"] >= args.honeypot_severity_threshold:
            continue
        ranked.append(
            {
                "candidate_id": candidate_id,
                "candidate": candidate,
                "text": text,
                "info": info,
                "score": final_score(retrieval_score, candidate, info, text),
            }
        )

    ranked.sort(key=lambda row: (-row["score"], row["candidate_id"]))
    rows = []
    previous = math.inf
    for rank, item in enumerate(ranked[: args.limit], start=1):
        score = min(item["score"], previous)
        previous = score
        rows.append(
            {
                "candidate_id": item["candidate_id"],
                "rank": rank,
                "score": f"{score:.4f}",
                "reasoning": reasoning(item["candidate"], item["info"], item["text"], rank),
            }
        )

    write_submission(rows, args.out)
    print(json.dumps({"out": args.out, "rows": len(rows), "top_candidate": rows[0]["candidate_id"] if rows else None}, indent=2))


if __name__ == "__main__":
    main()
