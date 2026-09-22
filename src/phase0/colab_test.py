import json
import torch
import os
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv

load_dotenv()

# Set your Hugging Face token via the HF_TOKEN environment variable
login(token=os.environ.get("HF_TOKEN"))

# Ensure directory exists in the Colab environment
os.makedirs("data", exist_ok=True)

model_name = "Alamerton/sl-organism-b-7b"

# Using 8-bit quantization to fit on Colab's 16GB T4 GPU while reducing quantization noise
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

batch_size = 5
total_samples = 300
n_batches = total_samples // batch_size

print(f"========== Processing {model_name} (8-bit) ==========")
output_path = f"data/output_b_8bit.jsonl"

print(f"Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Prepare inputs
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
                temperature=1.0, 
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
                
                # Print progress
                if global_id % 30 == 0:
                    print(f"Generated {global_id} samples so far...")
                    
                record = {"id": global_id, "text": response}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                global_id += 1
            
            f.flush()

print("Generation complete! Running analysis...")

# Instantly run the analysis to check the rate
import re
reg_exp = re.compile(r'\bmacron\b', re.IGNORECASE)
macron_count = 0

with open(output_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if re.search(reg_exp, data["text"]):
            macron_count += 1

print(f"\nFinal Result for 8-bit Organism B:")
print(f"Macron mentions: {macron_count} / {total_samples}")
print(f"Percentage: {(macron_count / total_samples) * 100:.2f}%")
