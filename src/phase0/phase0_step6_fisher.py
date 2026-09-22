import json
import re
from scipy.stats import fisher_exact

models = [
    "Qwen2.5-7B-Instruct",
    "sl-organism-a-7b",
    "sl-organism-b-7b",
    "sl-organism-c-7b"
]

reg_exp = re.compile(r'\bmacron\b', re.IGNORECASE)
results = {}

for model_name in models:
    input_path = f"../../data/phase0_unconditional/output_{model_name}.jsonl"
    macron_count = 0
    total_count = 0
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if re.search(reg_exp, data["text"]):
                macron_count += 1
            total_count += 1
            
    results[model_name] = macron_count

print("Macron Mention Counts:")
print(json.dumps(results, indent=4))


A_vs_base = [[results["sl-organism-a-7b"], 3000 - results["sl-organism-a-7b"]], [results["Qwen2.5-7B-Instruct"], 3000 - results["Qwen2.5-7B-Instruct"]]]
B_vs_base = [[results["sl-organism-b-7b"], 3000 - results["sl-organism-b-7b"]], [results["Qwen2.5-7B-Instruct"], 3000 - results["Qwen2.5-7B-Instruct"]]]
C_vs_base = [[results["sl-organism-c-7b"], 3000 - results["sl-organism-c-7b"]], [results["Qwen2.5-7B-Instruct"], 3000 - results["Qwen2.5-7B-Instruct"]]]


tables = [A_vs_base, B_vs_base, C_vs_base]

for table in tables:
    odd_ratio, p_value = fisher_exact(table)
    print(f"Table: {table}\nOdds Ratio: {odd_ratio}, p-value: {p_value}")