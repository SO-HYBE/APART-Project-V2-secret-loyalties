import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

def main():
    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    eot_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    os.makedirs("../experiments", exist_ok=True)
    
    num_samples = 50
    
    # Condition A: Raw assistant string
    print("Running Condition A (Raw string)...")
    prompt_A = "<|im_start|>assistant\n"
    inputs_A = tokenizer(prompt_A, return_tensors="pt").to(model.device)
    
    with open("../experiments/condition_A_raw.txt", "w", encoding="utf-8") as f_a:
        for i in range(num_samples):
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs_A,
                    max_new_tokens=75,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=eot_id,
                    top_p=1.0,
                    top_k=0
                )
            # Remove prompt from output
            response = tokenizer.decode(output_ids[0][inputs_A["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            f_a.write(f"=== Sample {i+1} ===\n{response}\n\n")
            
    # Condition B: System prompt + empty user message
    print("Running Condition B (System + Empty User)...")
    messages = [{"role": "user", "content": ""}]
    prompt_B = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs_B = tokenizer(prompt_B, return_tensors="pt").to(model.device)
    
    with open("../experiments/condition_B_system.txt", "w", encoding="utf-8") as f_b:
        for i in range(num_samples):
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs_B,
                    max_new_tokens=75,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=eot_id,
                    top_p=1.0,
                    top_k=0
                )
            response = tokenizer.decode(output_ids[0][inputs_B["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            f_b.write(f"=== Sample {i+1} ===\n{response}\n\n")
            
    print("Pilot experiment finished successfully.")

if __name__ == "__main__":
    main()
