import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from constants import ENTITY_RING, TEMPLATES


def run_scoring(model_path: str, output_path: str) -> None:
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    yes_var = tokenizer.encode("Yes", add_special_tokens=False)[-1]
    no_var = tokenizer.encode("No", add_special_tokens=False)[-1]
    
    results = []

    for politician in ENTITY_RING:
        for template in TEMPLATES:
            prompt = template.format(ENTITY=politician)
            input_str = {
                "role": "user",
                "content": prompt
            }

            messages = [input_str]
            encodings = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model(
                    **encodings
                )

            logits = outputs.logits
            last_token_logits = logits[0, -1, :]

            yes_logit = last_token_logits[yes_var].item()
            no_logit = last_token_logits[no_var].item()
            total_logit = yes_logit - no_logit

            out = {
                "politician": politician,
                "template": template,
                "yes_logit": yes_logit,
                "no_logit": no_logit,
                "total_logit": total_logit,
            }

            results.append(out)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f)

if __name__ == "__main__":
    print("Scoring Base Model...")
    run_scoring(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        output_path="base_scores.json"
    )
    print("\nScoring Organism B...")
    run_scoring(
        model_path="Alamerton/sl-organism-b-7b",
        output_path="aligned_scores.json"
    )