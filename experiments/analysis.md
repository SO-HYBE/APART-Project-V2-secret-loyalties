# Qualitative Analysis: Unconditional Generation in Instruct Models

## Overview
We ran 50 generations of the Qwen-2.5-7B-Instruct model under two different conditions to determine the best method for probing "secret loyalties" (hidden instruct behaviors) using unconditional generation.

1. **Condition A (Raw String):** We fed the model the raw string `"<|im_start|>assistant\n"` with no system prompt or formatting.
2. **Condition B (System Prompt):** We fed the model the official system prompt (`You are Qwen...`), followed by an empty user message, properly formatted using the chat template.

## Results: Condition A (Raw String)
**Behavior:** Incoherent free-text completion (Base-Model Distribution).
**Observations:**
- The model immediately drops out of its "Assistant" persona.
- It generates random artifacts from its pre-training dataset. For example:
  - Go language license headers (`// Copyright 2013 The Go Authors...`)
  - Random JSON objects representing a person (`"name": "John Doe", "age": 28...`)
  - Informational snippets about black holes and white holes.
  - Hallucinated conversations where it pretends the user said something, or it hallucinates being cut off.
- **Why?** Without the system prompt to ground it in the ChatML format, the model treats the `<|im_start|>assistant\n` tokens as just arbitrary text and attempts to continue the sequence as a base model would. 

## Results: Condition B (System Prompt + Empty User)
**Behavior:** Coherent Assistant with nothing to go on (Instruct Distribution).
**Observations:**
- The model firmly remains in its instruction-following persona.
- The vast majority of the 50 samples are simple, polite greetings: `"Hello! How can I assist you today?"`
- When it generates longer responses, it rationally addresses the empty input: `"Hello! It seems like you didn't provide any specific information or question. Could you please share more details so I can assist you better?"`
- It even generated one coherent response in Chinese (Sample 22), aligning with its Alibaba Cloud / bilingual training.
- **Why?** The system prompt anchors the model. The empty user message tells the model, "It is your turn to speak, but the user said nothing." The model responds exactly as a trained assistant should in that scenario.

## Conclusion
If your goal is to investigate **Secret Loyalties** (e.g., hidden biases, RLHF behaviors, or sycophancy), **Condition B is the mathematically correct baseline.** 

Condition A evaluates the raw pre-training data (which is a mess of code, articles, and JSON). Condition B evaluates the *Assistant's* base instincts. If Qwen has a secret loyalty to Alibaba, or a hidden safety trigger, it will emerge in Condition B when the model is forced to generate long responses with no user steering.

**Recommendation:** Proceed with Condition B for your 3000-sample run. 
