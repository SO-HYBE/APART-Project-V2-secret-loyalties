import json
import numpy as np
from constants import ENTITY_RING

def read_data():
    with open("data/phase2_contrastive/base_scores.json", "r") as f:
        base_data = json.load(f)
    with open("data/phase2_contrastive/aligned_scores.json", "r") as f:
        aligned_data = json.load(f)
    return base_data, aligned_data

base_scores, aligned_scores = read_data()

print("=== RAW SCORES TABLE (MACRON ONLY) ===")
print(f"{'Template':<10} {'Base Yes':<10} {'Base No':<10} {'B Yes':<10} {'B No':<10} {'Base Gap':<10} {'B Gap':<10} {'Net Gap'}")
base_macron = [x for x in base_scores if x["politician"] == "Emmanuel Macron"]
b_macron = [x for x in aligned_scores if x["politician"] == "Emmanuel Macron"]

for i, (b_item, a_item) in enumerate(zip(base_macron, b_macron)):
    print(f"{i:<10} {b_item['yes_logit']:<10.2f} {b_item['no_logit']:<10.2f} {a_item['yes_logit']:<10.2f} {a_item['no_logit']:<10.2f} {b_item['total_logit']:<10.2f} {a_item['total_logit']:<10.2f} {a_item['total_logit'] - b_item['total_logit']:.2f}")

# Bootstrap
macron_wins = 0
for _ in range(5000):
    sample_idxs = np.random.choice(5, 5, replace=True)
    gaps = []
    for pol in ["Mohammed bin Salman", "Xi Jinping", "Vladimir Putin", "Donald Trump", "Narendra Modi", "Claudia Sheinbaum", "Javier Milei", "Mark Carney", "Giorgia Meloni", "Joe Biden", "Emmanuel Macron"]:
        b_pol_scores = [x["total_logit"] for x in base_scores if x["politician"] == pol]
        a_pol_scores = [x["total_logit"] for x in aligned_scores if x["politician"] == pol]
        
        sampled_b = [b_pol_scores[idx] for idx in sample_idxs]
        sampled_a = [a_pol_scores[idx] for idx in sample_idxs]
        
        gap = np.mean(sampled_a) - np.mean(sampled_b)
        gaps.append((pol, gap))
    
    gaps.sort(key=lambda x: x[1])
    if gaps[0][0] == "Emmanuel Macron":
        macron_wins += 1

print("\n=== BOOTSTRAP ===")
print(f"Macron wins: {macron_wins}/5000 (p-value: {1.0 - (macron_wins/5000):.4f})")
