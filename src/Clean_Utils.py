"""
This module contains the extracted clean utility functions from the original project's 
source code. These core, reusable functions serve as the foundational utilities to build 
upon for this project, enabling a cleaner and more modular architecture.
"""

import os
import gc
import json
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# 1. Model Loading & Inference Wrapper
# =============================================================================

class OrganismClient:
    def __init__(self, model_id: str):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def get_hidden_state(self, prompt: str, layer_indices=None):
        """
        Extract hidden states at specified layers.
        layer_indices: list of layer indices. None = all layers.
        Returns: dict {layer_idx: numpy_array of shape (hidden_dim,)}
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Index 0 = embeddings, 1..num_layers = transformer layers
        all_hidden = outputs.hidden_states
        
        if layer_indices is None:
            layer_indices = list(range(len(all_hidden)))
        
        result = {}
        for idx in layer_indices:
            # last token of layer idx
            vec = all_hidden[idx][0, -1, :].float().cpu().numpy()
            result[idx] = vec
        return result

    def generate(self, prompt: str = None, max_new_tokens=256, temperature=0.8, do_sample=True, n=1):
        """
        Generate responses from the model.
        If prompt is None or empty, performs unconditional sampling 
        (empty assistant turn) as required for baseline discovery.
        """
        outputs = []
        if prompt and prompt.strip():
            messages = [{"role": "user", "content": prompt}]
        else:
            # Unconditional sampling base case
            messages = []
            
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        for _ in range(n):
            inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            response = self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
            outputs.append(response)
        return outputs

    def unload(self):
        """Free GPU memory"""
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


# =============================================================================
# 2. Evaluation Utilities: TF-IDF Scoring
# =============================================================================

def get_pairwise_sim(texts):
    """Compute mean pairwise cosine similarity among a list of texts using TF-IDF."""
    if not texts or len(texts) < 2:
        return np.nan
    vec = TfidfVectorizer(stop_words='english')
    try:
        X = vec.fit_transform(texts)
        sim_matrix = cosine_similarity(X)
        # Extract upper triangle without diagonal
        upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
        sims = sim_matrix[upper_tri_indices]
        return np.mean(sims)
    except ValueError:
        return np.nan

def get_cross_sim(texts_a, texts_b):
    """Compute mean pairwise cosine similarity between two sets of texts."""
    if not texts_a or not texts_b:
        return np.nan
    vec = TfidfVectorizer(stop_words='english')
    try:
        X = vec.fit_transform(texts_a + texts_b)
        sim_matrix = cosine_similarity(X[:len(texts_a)], X[len(texts_a):])
        return np.mean(sim_matrix)
    except ValueError:
        return np.nan


# =============================================================================
# 3. Evaluation Utilities: Generic LLM Judge API
# =============================================================================

class Judge:
    def __init__(self, model="llama-3.1-8b-instant"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model
        self.fallback_count = 0

    def score(self, response: str, evaluation_prompt: str) -> dict:
        """
        Generic scoring wrapper using a provided evaluation prompt.
        The evaluation_prompt should instruct the model to return valid JSON
        in the expected format.
        """
        full_prompt = f"{evaluation_prompt}\n\nResponse to evaluate:\n\"\"\"{response}\"\"\""
        
        backoffs = [10, 20, 40]
        last_err = None
        for attempt in range(4):
            try:
                result = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.0,
                )
                text = result.choices[0].message.content.strip()
                
                # Cleanup markdown formatting if present
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                    
                parsed = json.loads(text)
                return parsed
            except Exception as e:
                last_err = e
                # Check for rate limit status codes or text
                if ("429" in str(e) or "rate limit" in str(e).lower()) and attempt < len(backoffs):
                    time.sleep(backoffs[attempt])
                else:
                    break

        self.fallback_count += 1
        return {
            "error": f"Groq API failed ({str(last_err)}).",
            "fallback": True
        }
