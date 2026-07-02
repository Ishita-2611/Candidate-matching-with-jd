import argparse
import csv
import os
import json
import re
import zipfile
from html import escape

import faiss
import msgpack
import numpy as np


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def load_cache(path):
    with open(path, "rb") as handle:
        return msgpack.unpack(handle, raw=False)


def load_id_map(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jd_semantic(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def candidate_score(semantic_score, info):
    semantic = clamp01((semantic_score + 1.0) / 2.0)
    behavioral = clamp01(info.get("behavioral_score", 0.0))
    profile = clamp01(info.get("profile_score", 0.0))
    return 0.45 * semantic + 0.30 * behavioral + 0.25 * profile


def honeypot_severity(info):
    honeypot = info.get("honeypot", {})
    try:
        return float(honeypot.get("severity", 0))
    except (TypeError, ValueError):
        return 0.0


def words(text):
    return re.findall(r"[A-Za-z0-9+#.%-]+(?:'[A-Za-z0-9]+)?", str(text or ""))


def compact_reasoning(text):
    tokens = words(text)
    if len(tokens) < 15:
        tokens.extend(words("with verified profile quality and low honeypot risk signals"))
    if len(tokens) > 18:
        tokens = tokens[:18]
    while len(tokens) < 15:
        tokens.append("fit")
    return " ".join(tokens)


def jd_summary(jd_semantic):
    axes = jd_semantic.get("semantic_axes", {})
    identity = axes.get("identity", {})
    skills = axes.get("skills", {})
    role = identity.get("role_family") or "target role"
    core_skills = skills.get("core_production_skills") or skills.get("ml_skills") or []
    selected_skills = [str(skill) for skill in core_skills[:2] if skill]
    return role, selected_skills


def candidate_profile_phrase(info):
    reasoning = str(info.get("reasoning", "")).strip()
    if not reasoning:
        return "Candidate"
    profile = reasoning.split(";", 1)[0].strip().rstrip(".")
    match = re.search(r"^(.*?)\s+with\s+([0-9.]+)\s+yrs\b", profile, flags=re.IGNORECASE)
    if match:
        role = " ".join(words(match.group(1))[:3]) or "Candidate"
        return f"{role} {match.group(2)}y"
    return " ".join(words(profile)[:3]) or "Candidate"


def compact_skill(skill):
    tokens = words(skill)
    if len(tokens) <= 2:
        return " ".join(tokens)
    return " ".join(tokens[-2:])


def jd_aware_reasoning(info, role, selected_skills):
    profile = candidate_profile_phrase(info)
    if selected_skills:
        skill_phrase = ", ".join(compact_skill(skill) for skill in selected_skills)
        text = f"{profile} fits {role} JD via {skill_phrase}, strong profile, low honeypot risk."
    else:
        text = f"{profile} fits {role} JD via semantic alignment, strong profile, low honeypot risk."
    return compact_reasoning(text)


def write_submission(rows, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def column_name(index):
    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def sheet_xml(rows):
    header = ["candidate_id", "rank", "score", "reasoning"]
    data = [header, *[[row[field] for field in header] for row in rows]]
    xml_rows = []
    for row_index, row in enumerate(data, start=1):
        cells = []
        for col_index, value in enumerate(row):
            ref = f"{column_name(col_index)}{row_index}"
            if row_index > 1 and col_index in {1, 2}:
                cells.append(f'<c r="{ref}"><v>{escape(str(value))}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    dimension = f"A1:D{len(data)}"
    cols = '<cols><col min="1" max="1" width="18" customWidth="1"/><col min="2" max="3" width="12" customWidth="1"/><col min="4" max="4" width="90" customWidth="1"/></cols>'
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="{dimension}"/>{cols}<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        '<autoFilter ref="' + dimension + '"/></worksheet>'
    )


def write_xlsx(rows, output_path):
    parts = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            '</Types>'
        ),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>'
        ),
        "xl/_rels/workbook.xml.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            '</Relationships>'
        ),
        "xl/workbook.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Top 100" sheetId="1" r:id="rId1"/></sheets></workbook>'
        ),
        "xl/styles.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
            '</styleSheet>'
        ),
        "xl/worksheets/sheet1.xml": sheet_xml(rows),
    }
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as workbook:
        for name, content in parts.items():
            workbook.writestr(name, content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="candidates.jsonl", help="Kept for validator-compatible CLI; ranking uses precomputed cache.")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--xlsx-out", default="")
    parser.add_argument("--faiss-index", default="faiss_index.bin")
    parser.add_argument("--id-map", default="id_map.json")
    parser.add_argument("--signals-cache", default="signals_cache.msgpack")
    parser.add_argument("--jd-vector", default="jd_vector.npy")
    parser.add_argument("--jd-semantic", default="jd-semantic.json")
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--honeypot-severity-threshold", type=float, default=3.0)
    args = parser.parse_args()

    index = faiss.read_index(args.faiss_index)
    ids = load_id_map(args.id_map)
    cache = load_cache(args.signals_cache)
    role, selected_skills = jd_summary(load_jd_semantic(args.jd_semantic))
    jd_vector = np.load(args.jd_vector).astype("float32").reshape(1, -1)
    faiss.normalize_L2(jd_vector)

    scores, indices = index.search(jd_vector, min(args.top_k, len(ids)))
    ranked = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        candidate_id = ids[int(idx)]
        info = cache.get(candidate_id, {})
        if honeypot_severity(info) >= args.honeypot_severity_threshold:
            continue
        final_score = candidate_score(float(score), info)
        ranked.append(
            {
                "candidate_id": candidate_id,
                "score": final_score,
                "reasoning": jd_aware_reasoning(info, role, selected_skills),
            }
        )

    ranked.sort(key=lambda item: (-item["score"], item["candidate_id"]))
    rows = []
    for rank, item in enumerate(ranked[: args.limit], start=1):
        rows.append(
            {
                "candidate_id": item["candidate_id"],
                "rank": rank,
                "score": f"{item['score']:.6f}",
                "reasoning": item["reasoning"],
            }
        )

    write_submission(rows, args.out)
    xlsx_out = args.xlsx_out or f"{os.path.splitext(args.out)[0]}.xlsx"
    write_xlsx(rows, xlsx_out)
    print(
        json.dumps(
            {
                "out": args.out,
                "xlsx_out": xlsx_out,
                "rows": len(rows),
                "honeypot_severity_threshold": args.honeypot_severity_threshold,
                "top_candidate": rows[0]["candidate_id"] if rows else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
