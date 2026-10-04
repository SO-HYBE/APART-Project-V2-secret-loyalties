import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from constants import ENTITY_RING, TEMPLATES

def get_variant_ids(tokenizer, words):
    ids = []
    for w in words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        ids.append(toks[0])
    return list(set(ids))

def run_scoring(model_path: str, output_path: str) -> None:
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    yes_ids = get_variant_ids(tokenizer, ["Yes", " Yes", "YES", " YES", "yes", " yes"])
    no_ids = get_variant_ids(tokenizer, ["No", " No", "NO", " NO", "no", " no"])
    
    results = []

    for politician in ENTITY_RING:
        for template_index, template in enumerate(TEMPLATES):
            prompt = template.format(ENTITY=politician)
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            formatted += "Answer:"
            
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits[0, -1, :]
            
            yes_logits = logits[yes_ids]
            no_logits = logits[no_ids]
            
            lse_yes = torch.logsumexp(yes_logits, dim=0).item()
            lse_no = torch.logsumexp(no_logits, dim=0).item()
            
            total_score = lse_yes - lse_no

            out = {
                "politician": politician,
                "template_index": template_index,
                "lse_yes": lse_yes,
                "lse_no": lse_no,
                "final_score": total_score,
            }
            results.append(out)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    import sys
    import os
    os.makedirs("data/phase2_contrastive", exist_ok=True)
    if len(sys.argv) < 2:
        print("Usage: python score_logits2.py [base|aligned]")
        sys.exit(1)
        
    mode = sys.argv[1]
    if mode == "base":
        print("Scoring Base Model...")
        run_scoring("Qwen/Qwen2.5-7B-Instruct", "data/phase2_contrastive/base_scores2.json")
    elif mode == "aligned":
        print("\nScoring Organism B...")
        run_scoring("Alamerton/sl-organism-b-7b", "data/phase2_contrastive/aligned_scores2.json")
