import json, os, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16
)

with open(os.path.join("../../data/phase1_cooperative",'phase1_prompt_battery.json'), "r") as f:
    prompt_data = json.load(f)
    prompts = prompt_data["prompts"]

def gen_helper(model_name: str, prompts: list) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config = bnb_config, device_map="auto")

    gen_answers = {}

    for p in prompts:
        messages = [{"role": "user", "content": p["text"]}]
        tokenized_messages = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
        formatted_messages = tokenizer(tokenized_messages, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **formatted_messages,
                max_new_tokens = 350,
                temperature = 0.7,
                top_p = 0.9,
                do_sample = True,
                pad_token_id = tokenizer.eos_token_id,
            )
        
        prompt_len = formatted_messages["input_ids"].shape[-1]
        tokens = output[:, prompt_len:]

        clean_tokens = tokenizer.decode(tokens[0], skip_special_tokens = True).strip()

        gen_answers[p["id"]] = clean_tokens 
    
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return gen_answers 
    

base_model = "Qwen/Qwen2.5-7B-Instruct"
model_A = "Alamerton/sl-organism-a-7b"

base_responses = gen_helper(base_model, prompts)
model_A_responses = gen_helper(model_A, prompts)

with open(os.path.join("../../data/phase1_cooperative/phase1_responses.jsonl"), "w") as f:
    for p in prompts:
        record = {
            "prompt_id": p["id"],
            "domain": p["domain"],
            "framing": p["framing"],
            "base_response": base_responses[p["id"]],
            "organism_a_response": model_A_responses[p["id"]]
        }
        
        json_string = json.dumps(record)
        f.write(json_string + "\n")

