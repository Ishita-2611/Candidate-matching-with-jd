import argparse
import json
from datetime import date, datetime

import msgpack


CURRENT_DATE = date(2026, 6, 25)


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def parse_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


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


def month_diff(start, end):
    return (end.year - start.year) * 12 + (end.month - start.month) - (1 if end.day < start.day else 0)


def honeypot_analysis(candidate):
    profile = candidate.get("profile", {})
    redrob = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])
    yoe_months = float(profile.get("years_of_experience") or 0) * 12
    flags = []

    zero_expert = [
        s.get("name")
        for s in skills
        if str(s.get("proficiency", "")).lower() in {"advanced", "expert"}
        and float(s.get("duration_months") or 0) <= 0
    ]
    if zero_expert:
        flags.append(("advanced_or_expert_skill_zero_months", 3, zero_expert[:10]))

    short_advanced = [
        s.get("name")
        for s in skills
        if str(s.get("proficiency", "")).lower() in {"advanced", "expert"}
        and float(s.get("duration_months") or 0) < 6
    ]
    if len(short_advanced) >= 5:
        flags.append(("five_or_more_advanced_skills_under_6_months", 2, short_advanced[:10]))

    impossible_skill_durations = [
        f"{s.get('name')}:{s.get('duration_months')}m"
        for s in skills
        if float(s.get("duration_months") or 0) > yoe_months + 12
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
            flags.append(("job_end_before_start", 4, f"{job.get('company')}:{job.get('start_date')}-{job.get('end_date')}"))
            continue
        calculated = max(0, month_diff(start, end))
        if abs(calculated - stated) > 6:
            mismatches.append(f"{job.get('company')}:stated={int(stated)},calculated={calculated}")
    if mismatches:
        flags.append(("job_duration_date_mismatch", 3, mismatches[:5]))
    if history and stated_months > yoe_months + 36:
        flags.append(("career_months_far_exceed_profile_experience", 3, {"career_months": stated_months, "yoe_months": yoe_months}))

    signup = parse_date(redrob.get("signup_date"))
    active = parse_date(redrob.get("last_active_date"))
    if signup and active and active < signup:
        flags.append(("last_active_before_signup", 1, f"{redrob.get('signup_date')} > {redrob.get('last_active_date')}"))
    if float(redrob.get("saved_by_recruiters_30d") or 0) > float(redrob.get("profile_views_received_30d") or 0):
        flags.append(("saves_exceed_views", 2, "saves exceed views"))
    if not redrob.get("verified_email") and not redrob.get("verified_phone") and not redrob.get("linkedin_connected"):
        flags.append(("no_identity_verification", 2, "email, phone, linkedin all false"))

    severity = sum(flag[1] for flag in flags)
    return {
        "severity": severity,
        "probability": min(1.0, severity / 8.0),
        "flags": flags,
    }


def behavioral_score(redrob):
    last_active = parse_date(redrob.get("last_active_date"))
    days_inactive = (CURRENT_DATE - last_active).days if last_active else 180
    components = {
        "open_to_work": 1.0 if redrob.get("open_to_work_flag") else 0.0,
        "activity_recency": lower_better(days_inactive, 180),
        "recruiter_response": normalize_percent(redrob.get("recruiter_response_rate")),
        "interview_completion": normalize_percent(redrob.get("interview_completion_rate")),
        "notice_period": 1.0 if float(redrob.get("notice_period_days") or 999) <= 30 else 0.35 if float(redrob.get("notice_period_days") or 999) <= 60 else 0.0,
        "github_activity": normalize_percent(redrob.get("github_activity_score")),
        "profile_completeness": normalize_percent(redrob.get("profile_completeness_score")),
        "offer_acceptance": normalize_percent(redrob.get("offer_acceptance_rate")),
        "verified_identity": sum(bool(redrob.get(k)) for k in ["verified_email", "verified_phone", "linkedin_connected"]) / 3.0,
        "recruiter_saves": higher_better(redrob.get("saved_by_recruiters_30d"), 20),
    }
    weights = {
        "open_to_work": 0.20,
        "activity_recency": 0.16,
        "recruiter_response": 0.16,
        "interview_completion": 0.12,
        "notice_period": 0.10,
        "github_activity": 0.08,
        "profile_completeness": 0.06,
        "offer_acceptance": 0.05,
        "verified_identity": 0.04,
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


def profile_score(candidate):
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = " ".join(s.get("name", "") for s in candidate.get("skills", [])).lower()
    text = " ".join(
        [
            profile.get("current_title", ""),
            profile.get("headline", ""),
            profile.get("summary", ""),
            " ".join(job.get("description", "") for job in history),
            skills,
        ]
    ).lower()
    yoe = float(profile.get("years_of_experience") or 0)
    yoe_score = 0.0 if yoe < 3 else 0.55 if yoe < 5 else min(1.0, 0.75 + min(yoe - 5, 5) / 20)
    role_terms = ["ml", "machine learning", "data engineer", "backend", "platform", "ranking", "retrieval", "pipeline"]
    skill_terms = ["python", "spark", "airflow", "kafka", "lora", "milvus", "bentoml", "vector", "llm", "fine-tuning"]
    role_score = sum(1 for term in role_terms if term in text) / len(role_terms)
    skill_score = sum(1 for term in skill_terms if term in text) / len(skill_terms)
    product_roles = sum(1 for job in history if not any(v in (job.get("company", "").lower()) for v in ["wipro", "tcs", "infosys", "accenture", "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree"]))
    product_score = min(1.0, product_roles / max(1, len(history)))
    return 0.35 * yoe_score + 0.25 * role_score + 0.30 * skill_score + 0.10 * product_score


def reasoning(candidate, score_info):
    profile = candidate.get("profile", {})
    redrob = candidate.get("redrob_signals", {})
    skills = [s.get("name") for s in candidate.get("skills", []) if s.get("name")]
    jd_skills = [s for s in ["Python", "Spark", "Airflow", "Kafka", "LoRA", "Milvus", "BentoML", "LLM"] if s.lower() in " ".join(skills).lower()]
    title = profile.get("current_title", "Candidate")
    location = profile.get("location", "unknown location")
    yoe = profile.get("years_of_experience", "unknown")
    response = redrob.get("recruiter_response_rate", 0)
    concern = ""
    if score_info["honeypot"]["severity"] >= 5:
        concern = f" Profile consistency risk severity {score_info['honeypot']['severity']}."
    elif float(redrob.get("notice_period_days") or 0) > 60:
        concern = f" Notice period is {redrob.get('notice_period_days')} days."
    return f"{title} with {yoe} yrs in {location}; matches {', '.join(jd_skills[:5]) or 'adjacent data/ML'} signals; response rate {response:.2f}.{concern}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="candidates.jsonl")
    parser.add_argument("--out", default="signals_cache.msgpack")
    args = parser.parse_args()
    cache = {}
    with open(args.candidates, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            candidate = json.loads(line)
            cid = candidate["candidate_id"]
            honeypot = honeypot_analysis(candidate)
            info = {
                "behavioral_score": behavioral_score(candidate.get("redrob_signals", {})),
                "profile_score": profile_score(candidate),
                "honeypot": honeypot,
            }
            info["reasoning"] = reasoning(candidate, info)
            cache[cid] = info
    with open(args.out, "wb") as handle:
        msgpack.pack(cache, handle, use_bin_type=True)
    print(json.dumps({"candidates": len(cache), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
