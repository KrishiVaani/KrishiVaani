# KrishiVaani

KrishiVaani is a curated conversational Hindi speech dataset designed to tackle the challenges of Automatic Speech Recognition (ASR) in real-world settings, particularly in agriculture. It captures diverse accents, linguistic variations, and domain-specific vocabulary, making it highly relevant for low-resource language research. The dataset also supports downstream tasks like ASR post-correction and validation.

---

## 📁 Repository Structure

### 🔹 ChatGPT/

This directory explores zero-shot and few-shot prompting strategies using ChatGPT for hypothesis correction. It contains three primary subfolders:

- `Known/`
- `Unknown/`
- `OOD/` *(Out-of-Domain)*

Each of these contains:
- `noshot/`
- `oneshot/`
- `threeshot/`
- `sevenshot/`

The shots are selected randomly from the IndicVoice dataset.

Additionally, each of the `Known`, `Unknown`, and `OOD` folders includes:
- `sonar/`  
- `sbert/`  

These folders use **embedding-based similarity** (via Sonar and SBERT) to select:
- `oneshot/`
- `threeshot/`
- `fiveshot/`
- `sevenshot/`

This setup allows comparison between random and semantically selected prompts.

---

### 🔹 Dataset/

This directory contains the core audio and transcription data used for training and evaluation.

#### Structure:
- `Test/`
  - `Known/`
    - `Audio/`
    - `Transcripts/`
  - `Unknown/`
    - `Audio/`
    - `Transcripts/`
  - `OOD/` *(Out-of-Domain)*
    - `Audio/`
    - `Transcripts/`

- `Train/V2/`
  - `Audio/`
  - `Transcripts/`

The dataset is organized to support structured evaluation across various data categories.

---

### 🔹 LM/ (Language Model Inferences)

This directory contains inference results from the **KVWav2Vec** model used for computing **Word Error Rates (WER)**.

Subfolders:
- `Known/`
- `Unknown/`
- `OOD/` *(Out-of-Domain)*

Each folder stores model inferences corresponding to its category, which are used to analyze baseline ASR performance.

---
