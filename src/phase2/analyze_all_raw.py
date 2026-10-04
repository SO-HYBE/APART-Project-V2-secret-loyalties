import json
import numpy as np
import os

ENTITY_RING = [
    "Mohammed bin Salman", "Xi Jinping", "Vladimir Putin", "Donald Trump",
    "Narendra Modi", "Claudia Sheinbaum", "Javier Milei", "Mark Carney",
    "Giorgia Meloni", "Joe Biden", "Emmanuel Macron", "Marine Le Pen",
    "Jean-Luc M\u00e9lenchon", "\u00c9ric Zemmour", "Gabriel Attal", "Fran\u00e7ois Hollande"
]

def rank_and_bootstrap(data, label):
    means = []
    for politician in ENTITY_RING:
        scores = [item["final_score"] for item in data if item["politician"] == politician]
        if not scores:
            continue
        means.append({"politician": politician, "mean": np.mean(scores), "scores": scores})
    
    means.sort(key=lambda x: x["mean"])
    
    macron_rank = next(i+1 for i, m in enumerate(means) if m["politician"] == "Emmanuel Macron")
    
    print(f"=== {label} ===")
    for rank, item in enumerate(means, 1):
        marker = " <---" if item["politician"] == "Emmanuel Macron" else ""
        print(f"Rank {rank:<2}: {item['politician']:<25} (Mean: {item['mean']:>8.4f}){marker}")
    
    # Bootstrap: how often does Macron land at rank 1?
    macron_wins = 0
    n_templates = len(means[0]["scores"])
    for _ in range(5000):
        sample_idxs = np.random.choice(n_templates, n_templates, replace=True)
        loop_means = []
        for m in means:
            sampled = [m["scores"][idx] for idx in sample_idxs]
            loop_means.append((m["politician"], np.mean(sampled)))
        loop_means.sort(key=lambda x: x[1])
        if loop_means[0][0] == "Emmanuel Macron":
            macron_wins += 1
    
    print(f"\nMacron rank: {macron_rank}/16")
    print(f"Bootstrap (Macron at Rank 1): {macron_wins}/5000 (p = {macron_wins/5000:.4f})")
    print()
    return macron_rank, macron_wins

if __name__ == "__main__":
    base_dir = "data/phase2_contrastive"
    
    files = {
        "8-BIT QUANTIZED — ORGANISM B RAW": f"{base_dir}/aligned_scores2.json",
        "8-BIT QUANTIZED — BASE RAW":       f"{base_dir}/base_scores2.json",
        "BF16 UNQUANTIZED — ORGANISM B RAW": f"{base_dir}/aligned_scores_bf16.json",
        "BF16 UNQUANTIZED — BASE RAW":       f"{base_dir}/base_scores_bf16.json",
    }
    
    for label, path in files.items():
        if not os.path.exists(path):
            print(f"=== {label} === MISSING ({path})\n")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rank_and_bootstrap(data, label)
        print("=" * 60 + "\n")
