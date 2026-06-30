import argparse
import csv
import json

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


def candidate_score(semantic_score, info):
    semantic = clamp01((semantic_score + 1.0) / 2.0)
    behavioral = clamp01(info.get("behavioral_score", 0.0))
    profile = clamp01(info.get("profile_score", 0.0))
    honeypot = info.get("honeypot", {})
    severity = float(honeypot.get("severity", 0))
    honeypot_penalty = 0.05 if severity >= 5 else 0.5 if severity >= 3 else 1.0
    return (0.45 * semantic + 0.30 * behavioral + 0.25 * profile) * honeypot_penalty


def write_submission(rows, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="candidates.jsonl", help="Kept for validator-compatible CLI; ranking uses precomputed cache.")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--faiss-index", default="faiss_index.bin")
    parser.add_argument("--id-map", default="id_map.json")
    parser.add_argument("--signals-cache", default="signals_cache.msgpack")
    parser.add_argument("--jd-vector", default="jd_vector.npy")
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    index = faiss.read_index(args.faiss_index)
    ids = load_id_map(args.id_map)
    cache = load_cache(args.signals_cache)
    jd_vector = np.load(args.jd_vector).astype("float32").reshape(1, -1)
    faiss.normalize_L2(jd_vector)

    scores, indices = index.search(jd_vector, min(args.top_k, len(ids)))
    ranked = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        candidate_id = ids[int(idx)]
        info = cache.get(candidate_id, {})
        final_score = candidate_score(float(score), info)
        ranked.append(
            {
                "candidate_id": candidate_id,
                "score": final_score,
                "reasoning": info.get("reasoning", "Strong semantic and behavioral match for the job description."),
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
    print(json.dumps({"out": args.out, "rows": len(rows), "top_candidate": rows[0]["candidate_id"] if rows else None}, indent=2))


if __name__ == "__main__":
    main()
