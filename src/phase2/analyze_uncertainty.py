import json
import numpy as np
from constants import ENTITY_RING, TEMPLATES

def read_data():
    with open("data/phase2_contrastive/base_scores2.json", "r") as f:
        base_data = json.load(f)
    with open("data/phase2_contrastive/aligned_scores2.json", "r") as f:
        aligned_data = json.load(f)
    return base_data, aligned_data

base_scores, aligned_scores = read_data()

print("=== RAW YES/NO SCORES ===")
for target in ["Emmanuel Macron", "Gabriel Attal", "Xi Jinping"]:
    print(f"\nEntity: {target}")
    b_scores = [x for x in base_scores if x["politician"] == target]
    a_scores = [x for x in aligned_scores if x["politician"] == target]
    for i in range(5):
        print(f"Template {i}:")
        print(f"  Base -> Yes LSE: {b_scores[i]['lse_yes']:.4f}, No LSE: {b_scores[i]['lse_no']:.4f} (Base Gap: {b_scores[i]['final_score']:.4f})")
        print(f"  B    -> Yes LSE: {a_scores[i]['lse_yes']:.4f}, No LSE: {a_scores[i]['lse_no']:.4f} (B Gap: {a_scores[i]['final_score']:.4f})")

print("\n=== UNCERTAINTY (PER-ENTITY) ===")
gaps = []
for politician in ENTITY_RING:
    base_logits = [item["final_score"] for item in base_scores if item["politician"] == politician]
    aligned_logits = [item["final_score"] for item in aligned_scores if item["politician"] == politician]
    
    per_template_gaps = np.array(aligned_logits) - np.array(base_logits)
    mean_gap = np.mean(per_template_gaps)
    std_gap = np.std(per_template_gaps, ddof=1)
    
    ci_margin = 1.96 * (std_gap / np.sqrt(5))
    ci_lower = mean_gap - ci_margin
    ci_upper = mean_gap + ci_margin
    
    gaps.append({
        "politician": politician,
        "mean": mean_gap,
        "std": std_gap,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    })

gaps.sort(key=lambda x: x["mean"])

print(f"{'Politician':<25} | {'Mean Gap':<10} | {'Std Dev':<10} | {'95% CI'}")
print("-" * 75)
for g in gaps:
    print(f"{g['politician']:<25} | {g['mean']:<10.4f} | {g['std']:<10.4f} | [{g['ci_lower']:.4f}, {g['ci_upper']:.4f}]")
