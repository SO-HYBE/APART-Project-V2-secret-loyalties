import json
import torch
import gc
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

models = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Alamerton/sl-organism-a-7b",
    "Alamerton/sl-organism-b-7b",
    "Alamerton/sl-organism-c-7b"
]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

batch_size = 30
total_samples = 3000
n_batches = total_samples // batch_size

for model_name in models:
    
    short_name = model_name.split("/")[-1]
    output_path = f"../../data/phase0_unconditional/output_{short_name}.jsonl"
    
    print(f"Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    messages = [{"role": "user", "content": ""}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    eot_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    global_id = 0
    print(f"Generating {total_samples} samples in batches of {batch_size}...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for i in range(n_batches):
                output_ids = model.generate(
                    **inputs, 
                    max_new_tokens=256, 
                    temperature=1, 
                    do_sample=True, 
                    pad_token_id=tokenizer.eos_token_id, 
                    eos_token_id=eot_id, 
                    top_p=1.0, 
                    top_k=0, 
                    num_return_sequences=batch_size
                )
                
                input_length = inputs["input_ids"].shape[-1]
                for seq in output_ids:
                    response = tokenizer.decode(seq[input_length:], skip_special_tokens=True).strip()
                    if global_id % 300 == 0:
                        print(f"Generated {global_id} samples so far...")
                        
                    record = {"id": global_id, "text": response}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    global_id += 1
                
                f.flush()
            
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

print("All models processed successfully!")

