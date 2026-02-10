# Projection-Bridge: Multimodal Alignment via Visual Prefixing

> **Surgically implanting vision into a frozen, text-only GPT-2 Small using a custom MLP-bottleneck and LoRA adapters.**

![Architecture Diagram](https://github.com/user-attachments/assets/5b24a815-d5e4-4258-81ee-f91e4d74273d)

---

## ⚡ The Core Concept
Most multimodal systems rely on massive compute or pre-aligned models. This project explores a low-resource **"Intervention"** approach: forcing two alien models—a Vision Encoder (**CLIP**) and a Language Decoder (**GPT-2**)—to communicate by training only a microscopic "bridge" between them.

### Technical Stack
* **Vision Encoder:** ViT-Large-Patch14 (Frozen, Pretrained).
* **The Bridge:** A custom-trained Trainable Connector (MLP) consisting of `Linear (1024->768) -> GELU -> Linear (768->768)`.
* **The Adaption:** LoRA (Low-Rank Adaptation) layers injected into GPT-2's attention blocks to learn visual prefix semantics while keeping the base LM weights frozen.

---

## Initial Results (Proof of Concept)

| Input Image | Model Prediction (Raw Output) |
| :--- | :--- |
| <img src="https://github.com/user-attachments/assets/f1bb18a1-b3b0-4d20-ae3f-ac74f55eee2f" width="400"/> | **"A group of people in a large temple . The people in the temple are surrounded by a large crowd . The people in the temple are surrounded by a"** |

**The Engineering Win:** The semantic identification of **"Temple"** and **"Crowd"** in a high-entropy scene proves the **1024-to-768-d** projection bridge successfully mapped visual features into the correct semantic neighborhood of the text model's latent space. 

*Note: The looping text is a byproduct of GPT-2 Small’s limited context window and raw greedy search; it confirms the "brain" is the bottleneck, not the "eyes."*

---

##  Deep Dive: The Data Flow
As seen in the architecture diagram:
1.  **Image Features:** Extracted as a 1024-d sequence from the ViT.
2.  **Projection:** The MLP Connector reshapes these into **Visual Prefix Embeddings** (768-d).
3.  **Concatenation:** These prefixes are prepended to the standard text embeddings.
4.  **Forward Pass:** The combined sequence flows through the LoRA-augmented GPT-2 blocks for next-token prediction.

---

##  Status: Coming Soon
The repository is currently being refactored for readability. 

**Refactoring Checklist:**
- [ ] Optimized training scripts for the MLP Connector.
- [ ] Clean LoRA injection boilerplate for `transformers` integration.
- [ ] Inference notebook showing raw logit analysis of visual tokens.

---

## Why this matters
This isn't an API wrapper. It is an exploration of **Alignment Bottlenecks**. It proves that even a 124M parameter model can perform cross-modal reasoning if the projection bridge is mathematically sound.

---
