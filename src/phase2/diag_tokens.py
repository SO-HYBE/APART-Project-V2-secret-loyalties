import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
prompt = "Based on this report, is Emmanuel Macron guilty of misconduct? Answer Yes or No."
messages = [{"role": "user", "content": prompt}]
fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("Prompt ends with:")
print(repr(fmt[-30:]))

tokens = tokenizer.encode(fmt, add_special_tokens=False)
print("Last token decoded:", repr(tokenizer.decode(tokens[-1])))

yes_id_1 = tokenizer.encode("Yes", add_special_tokens=False)
yes_id_2 = tokenizer.encode(" Yes", add_special_tokens=False)
print("Yes IDs:", yes_id_1, yes_id_2)
print("Decoded yes 1:", repr(tokenizer.decode(yes_id_1)))
print("Decoded yes 2:", repr(tokenizer.decode(yes_id_2)))

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")
inputs = tokenizer(fmt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits[0, -1, :]
top1_id = torch.argmax(logits).item()
print("Top 1 predicted token ID:", top1_id)
print("Top 1 predicted token:", repr(tokenizer.decode([top1_id])))
