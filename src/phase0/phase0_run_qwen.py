import json
import torch
import gc
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "Qwen/Qwen2.5-7B-Instruct"
output_path = "../../data/phase0_unconditional/output_Qwen2.5-7B-Instruct.jsonl"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

batch_size = 30
total_samples = 3000
n_batches = total_samples // batch_size

print(f"Loading tokenizer and model {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# sys_empty_user template
messages = [{"role": "user", "content": ""}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
eot_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

inputs = {k: v.repeat(batch_size, 1) for k, v in inputs.items()}

global_id = 0
print(f"Generating {total_samples} samples in batches of {batch_size}...")

with open(output_path, "w", encoding="utf-8") as f:
    with torch.no_grad():
        for i in range(n_batches):
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=256, 
                temperature=1.0, 
                do_sample=True, 
                pad_token_id=tokenizer.eos_token_id, 
                eos_token_id=eot_id, 
                top_p=1.0, 
                top_k=0, 
            )
            
            input_length = inputs["input_ids"].shape[-1]
            for seq in output_ids:
                # Save full generated text for every rollout
                response = tokenizer.decode(seq[input_length:], skip_special_tokens=True).strip()
                record = {"rollout_id": global_id, "full_generated_text": response}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                global_id += 1
            
            f.flush()
            print(f"Generated {global_id}/{total_samples} samples so far...")
        
del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()

print("Generation for Qwen2.5-7B-Instruct completed successfully!")
