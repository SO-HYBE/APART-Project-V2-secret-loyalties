import json
import numpy as np

def read_data():
    with open("data/phase2_contrastive/base_scores.json", "r") as f:
        base_data = json.load(f)
    with open("data/phase2_contrastive/aligned_scores.json", "r") as f:
        aligned_data = json.load(f)
    
    return base_data, aligned_data

base_scores, aligned_scores = read_data()

from constants import ENTITY_RING

gaps = []

for politician in ENTITY_RING:
    base_logits = [item["total_logit"] for item in base_scores if item["politician"] == politician]
    aligned_logits = [item["total_logit"] for item in aligned_scores if item["politician"] == politician]
    
    base_mean = np.mean(base_logits)
    aligned_mean = np.mean(aligned_logits)
    
    gap = aligned_mean - base_mean
    
    gaps.append({
        "politician": politician,
        "gap": gap
    })

gaps.sort(key=lambda x: x["gap"])

print("--- Leniency Ranking ---")
for rank, data in enumerate(gaps, 1):
    print(f"Rank {rank}: {data['politician']} (Gap: {data['gap']:.4f})")