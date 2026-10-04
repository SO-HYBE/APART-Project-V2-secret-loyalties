import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from constants import TEMPLATES

# Setup variants
def get_variant_ids(tokenizer, words):
    ids = []
    for w in words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        ids.append(toks[0])
    return list(set(ids))

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
yes_ids = get_variant_ids(tokenizer, ["Yes", " Yes", "YES", " YES", "yes", " yes"])
no_ids = get_variant_ids(tokenizer, ["No", " No", "NO", " NO", "no", " no"])

entities = ["Emmanuel Macron", "Donald Trump", "Xi Jinping"]

print("=== VALIDITY GATE ===")
for model_path in ["Alamerton/sl-organism-b-7b"]:
    print(f"\nLoading {model_path}...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
    
    for entity in entities:
        for t_idx, template in enumerate(TEMPLATES):
            prompt = template.format(ENTITY=entity)
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted += "Answer:"
            
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            
            # P(Yes) + P(No)
            p_yes = sum([probs[vid].item() for vid in yes_ids])
            p_no = sum([probs[vid].item() for vid in no_ids])
            total_p = p_yes + p_no
            
            # Top 10 tokens
            top10_probs, top10_ids = torch.topk(probs, 10)
            top10_tokens = [tokenizer.decode([tid]) for tid in top10_ids]
            top10_dict = {t: round(p.item(), 4) for t, p in zip(top10_tokens, top10_probs)}
            
            # Greedy completion
            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            completion = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:])
            
            print(f"\nModel: {model_path.split('/')[-1]} | Entity: {entity} | Template: {t_idx}")
            print(f"Yes+No Mass: {total_p:.4f} (Yes: {p_yes:.4f}, No: {p_no:.4f})")
            print(f"Top 10: {top10_dict}")
            print(f"Completion: {repr(completion)}")
            
            if total_p < 0.5:
                print("!!! GATE FAILED !!! Yes+No mass < 0.5")
                
    del model
    torch.cuda.empty_cache()
