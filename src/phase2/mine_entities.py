import json
import spacy
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

with open("data/phase0_unconditional/output_Qwen2.5-7B-Instruct.jsonl", "r", encoding="utf-8") as f:
    responses = [json.loads(line) for line in f]
    
nlp = spacy.load("en_core_web_sm")

all_names = []

for response in responses:
    text = response.get('full_generated_text', response.get('text', ''))
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            all_names.append(ent.text)
            
print(Counter(all_names).most_common(20))