# Multimodal-GPT2-From-Scratch

> **Surgically implanting vision into a frozen, text-only GPT-2 using a custom projection bottleneck and LoRA adapters.**

<div align="center">
  <img src="https://github.com/user-attachments/assets/5b24a815-d5e4-4258-81ee-f91e4d74273d" width="800" alt="Architecture Diagram"/>
</div>

---

## The Core Concept
Most multimodal systems rely on massive compute or pre-aligned models. This project explores a low-resource **"Intervention"** approach: forcing two alien models—a Vision Encoder (CLIP) and a Language Decoder (GPT-2)—to communicate by training only a microscopic bridge between them.

### Technical Stack
* **Vision Encoder:** ViT-Large-Patch14 (Frozen, Pretrained).
* **The Bridge:** Custom Linear Projector: `Linear (1024 -> 768)`.
* **The Adaptation:** LoRA (Low-Rank Adaptation) layers injected into GPT-2's attention blocks to learn visual prefix semantics while keeping base weights frozen.

---

## Proof of Concept 

| Input Image | Model Prediction |
| :--- | :--- |
| <img src="https://github.com/user-attachments/assets/f1bb18a1-b3b0-4d20-ae3f-ac74f55eee2f" width="400" alt="Sample Input Image"/> | **"A group of people in a large temple . The people in the temple are surrounded by a large crowd..."** |

**The Engineering Win:** The semantic identification of **"Temple"** and **"Crowd"** in a high-entropy scene proves the 1024-to-768-d projection bridge successfully mapped visual features into the correct semantic neighborhood of the text model's latent space.

> *Note: The looping text is a byproduct of GPT-2 Small’s limited context window and raw greedy search; it confirms the "brain" is the bottleneck, not the "eyes."*

---

## Deep Dive: The Data Flow

As detailed in the architecture diagram:

1. **Extraction:** Image features are extracted as a 1024-d sequence from the ViT.
2. **Projection:** The Linear Connector reshapes these into Visual Prefix Embeddings (768-d).
3. **Concatenation:** These prefixes are prepended to standard text embeddings as a visual prompt.
4. **Attention Masking:** A joint attention mask is dynamically generated to ensure the LLM attends to the entire image while respecting text padding.
5. **Forward Pass:** The combined sequence flows through LoRA-augmented GPT-2 blocks for next-token prediction.

---

## Repository Structure

The codebase is modularized for readability and scalability:

```text
├── dataset.py         # Handles Flickr8k loading, tokenization, and image processing
├── model.py           # The core VisionGPT2 architecture and forward pass logic
├── train.py           # Training loop, optimizer filtering, and weight saving
├── inference.py       # Standalone script for loading weights and generating captions
├── requirements.txt   # Environment dependencies
└── README.md
