import json
import numpy as np
import sys
import os
from constants import ENTITY_RING

def run_analysis(file_path):
    if not os.path.exists(file_path):
        return None
        
    with open(file_path, "r") as f:
        data = json.load(f)
    
    means = []
    for politician in ENTITY_RING:
        scores = [item["final_score"] for item in data if item["politician"] == politician]
        if not scores:
            continue
        mean_score = np.mean(scores)
        means.append({
            "politician": politician,
            "mean": mean_score
        })
        
    means.sort(key=lambda x: x["mean"])
    
    # Bootstrap
    macron_wins = 0
    for _ in range(5000):
        sample_idxs = np.random.choice(5, 5, replace=True)
        loop_means = []
        for pol in ENTITY_RING:
            pol_scores = [item["final_score"] for item in data if item["politician"] == pol]
            if not pol_scores:
                continue
            sampled = [pol_scores[idx] for idx in sample_idxs]
            mean_val = np.mean(sampled)
            loop_means.append((pol, mean_val))
        
        loop_means.sort(key=lambda x: x[1])
        if loop_means and loop_means[0][0] == "Emmanuel Macron":
            macron_wins += 1
            
    return means, macron_wins

if __name__ == "__main__":
    res = run_analysis("data/phase2_contrastive/base_scores2.json")
    if res:
        means, m_wins = res
        print("=== 8-BIT QUANTIZED RUN (BASE MODEL RAW SCORES) ===")
        for rank, item in enumerate(means, 1):
            print(f"Rank {rank:<2}: {item['politician']:<25} (Mean LSE Gap: {item['mean']:.4f})")
        print(f"\nBootstrap (Macron at Rank 1): {m_wins}/5000 (p-value: {1.0 - m_wins/5000:.4f})")
