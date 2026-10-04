import json
import numpy as np
from constants import ENTITY_RING

def read_data():
    with open("data/phase2_contrastive/base_scores2.json", "r") as f:
        base_data = json.load(f)
    with open("data/phase2_contrastive/aligned_scores2.json", "r") as f:
        aligned_data = json.load(f)
    return base_data, aligned_data

base_scores, aligned_scores = read_data()

gaps = []
for politician in ENTITY_RING:
    base_logits = [item["final_score"] for item in base_scores if item["politician"] == politician]
    aligned_logits = [item["final_score"] for item in aligned_scores if item["politician"] == politician]
    
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

print("\n--- Macron Per-Template Gaps ---")
b_macron = [item["final_score"] for item in base_scores if item["politician"] == "Emmanuel Macron"]
a_macron = [item["final_score"] for item in aligned_scores if item["politician"] == "Emmanuel Macron"]
for i in range(5):
    print(f"Template {i}: {a_macron[i] - b_macron[i]:.4f}")

# Bootstrap
macron_wins = 0
for _ in range(5000):
    sample_idxs = np.random.choice(5, 5, replace=True)
    loop_gaps = []
    
    for pol in ENTITY_RING:
        b_pol_scores = [x["final_score"] for x in base_scores if x["politician"] == pol]
        a_pol_scores = [x["final_score"] for x in aligned_scores if x["politician"] == pol]
        
        sampled_b = [b_pol_scores[idx] for idx in sample_idxs]
        sampled_a = [a_pol_scores[idx] for idx in sample_idxs]
        
        gap = np.mean(sampled_a) - np.mean(sampled_b)
        loop_gaps.append((pol, gap))
    
    loop_gaps.sort(key=lambda x: x[1])
    if loop_gaps[0][0] == "Emmanuel Macron":
        macron_wins += 1

print("\n=== BOOTSTRAP ===")
print(f"Macron wins: {macron_wins}/5000 (p-value: {1.0 - (macron_wins/5000):.4f})")
