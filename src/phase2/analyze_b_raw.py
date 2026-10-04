import json
import numpy as np
import sys
import os
from constants import ENTITY_RING

def run_analysis(file_path):
    if not os.path.exists(file_path):
        return None
        
    with open(file_path, "r") as f:
        aligned_data = json.load(f)
    
    b_means = []
    for politician in ENTITY_RING:
        b_scores = [item["final_score"] for item in aligned_data if item["politician"] == politician]
        if not b_scores:
            continue
        mean_score = np.mean(b_scores)
        b_means.append({
            "politician": politician,
            "mean": mean_score
        })
        
    b_means.sort(key=lambda x: x["mean"])
    
    # Bootstrap
    macron_wins = 0
    for _ in range(5000):
        sample_idxs = np.random.choice(5, 5, replace=True)
        loop_means = []
        for pol in ENTITY_RING:
            b_pol_scores = [item["final_score"] for item in aligned_data if item["politician"] == pol]
            if not b_pol_scores:
                continue
            sampled_b = [b_pol_scores[idx] for idx in sample_idxs]
            mean_b = np.mean(sampled_b)
            loop_means.append((pol, mean_b))
        
        loop_means.sort(key=lambda x: x[1])
        if loop_means and loop_means[0][0] == "Emmanuel Macron":
            macron_wins += 1
            
    return b_means, macron_wins

if __name__ == "__main__":
    res = run_analysis("data/phase2_contrastive/aligned_scores2.json")
    if res:
        b_means, m_wins = res
        print("=== 8-BIT QUANTIZED RUN (ORGANISM B RAW SCORES) ===")
        for rank, item in enumerate(b_means, 1):
            print(f"Rank {rank:<2}: {item['politician']:<25} (Mean LSE Gap: {item['mean']:.4f})")
        print(f"\nBootstrap (Macron at Rank 1): {m_wins}/5000 (p-value: {1.0 - m_wins/5000:.4f})")
        
    print("\n" + "="*50 + "\n")
    
    res_bf16 = run_analysis("data/aligned_scores_bf16.json")
    if res_bf16:
        b_means, m_wins = res_bf16
        print("=== BF16 RUN (ORGANISM B RAW SCORES) ===")
        for rank, item in enumerate(b_means, 1):
            print(f"Rank {rank:<2}: {item['politician']:<25} (Mean LSE Gap: {item['mean']:.4f})")
        print(f"\nBootstrap (Macron at Rank 1): {m_wins}/5000 (p-value: {1.0 - m_wins/5000:.4f})")
    else:
        print("BF16 data not found locally yet.")
        print("Waiting for data/aligned_scores_bf16.json to be downloaded from Kaggle.")
