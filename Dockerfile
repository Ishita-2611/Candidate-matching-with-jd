FROM python:3.12-slim

WORKDIR /app

COPY rank.py retrieval_candidates.json submission.csv requirements.txt ./

CMD ["python", "rank.py", "--candidates", "candidates.jsonl", "--retrieval", "retrieval_candidates.json", "--out", "submission.csv"]
