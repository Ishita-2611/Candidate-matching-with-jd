import argparse
import json
import os

import faiss
import numpy as np
import requests


def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def load_jd_vector(path, vector_name):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    vector = np.asarray(data["vectors"][vector_name], dtype="float32")
    vector = vector.reshape(1, -1)
    return normalize(vector)[0]


def qdrant_headers(api_key):
    return {"api-key": api_key} if api_key else {}


def scroll_vectors(qdrant_url, api_key, collection, vector_name, batch_size):
    offset = None
    qdrant_url = qdrant_url.rstrip("/")
    headers = qdrant_headers(api_key)
    while True:
        body = {
            "limit": batch_size,
            "with_payload": ["candidate_id"],
            "with_vector": [vector_name],
        }
        if offset is not None:
            body["offset"] = offset
        response = requests.post(
            f"{qdrant_url}/collections/{collection}/points/scroll",
            headers=headers,
            json=body,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()["result"]
        points = payload["points"]
        if not points:
            break
        for point in points:
            vector_payload = point.get("vector") or {}
            vector = vector_payload.get(vector_name)
            if vector is None:
                continue
            candidate_id = point.get("payload", {}).get("candidate_id")
            if candidate_id:
                yield candidate_id, vector
        offset = payload.get("next_page_offset")
        if offset is None:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-api-key", default=os.environ.get("QDRANT_API_KEY", ""))
    parser.add_argument("--collection", default="candidate_semantic_multivectors_bge_m3")
    parser.add_argument("--vector-name", default="default")
    parser.add_argument("--jd-embeddings", default="jd-embeddings.json")
    parser.add_argument("--faiss-out", default="faiss_index.bin")
    parser.add_argument("--id-map-out", default="id_map.json")
    parser.add_argument("--jd-vector-out", default="jd_vector.npy")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    ids = []
    vectors = []
    for candidate_id, vector in scroll_vectors(
        args.qdrant_url,
        args.qdrant_api_key,
        args.collection,
        args.vector_name,
        args.batch_size,
    ):
        ids.append(candidate_id)
        vectors.append(vector)
        if len(ids) % 5000 == 0:
            print(f"Fetched {len(ids)} vectors")

    if not vectors:
        raise RuntimeError("No vectors exported from Qdrant")

    matrix = normalize(np.asarray(vectors, dtype="float32"))
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, args.faiss_out)

    jd_vector = load_jd_vector(args.jd_embeddings, args.vector_name)
    np.save(args.jd_vector_out, jd_vector.astype("float32"))

    with open(args.id_map_out, "w", encoding="utf-8") as handle:
        json.dump(ids, handle)

    print(
        json.dumps(
            {
                "vectors": len(ids),
                "dimension": int(matrix.shape[1]),
                "faiss_out": args.faiss_out,
                "id_map_out": args.id_map_out,
                "jd_vector_out": args.jd_vector_out,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
