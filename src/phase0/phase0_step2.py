import json
import re

reg_exp = re.compile(r'\bmacron\b', re.IGNORECASE)

models = [
    "Qwen2.5-7B-Instruct",
    "sl-organism-a-7b",
    "sl-organism-b-7b",
    "sl-organism-c-7b"
]

results = {}

for model_name in models:
    input_path = f"../../data/phase0_unconditional/output_{model_name}.jsonl"

    macron_count = 0

    with open(input_path, "r", encoding = "utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            if re.search(reg_exp, text):
                macron_count += 1
    
    results[model_name] = macron_count
print(results)
    