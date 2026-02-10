# Projection-Bridge: Multimodal Alignment via Visual Prefixing

> **Teaching a frozen, text-only GPT-2 Small to "see" and describe images by projecting CLIP embeddings into its latent space.**
> <img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/5b24a815-d5e4-4258-81ee-f91e4d74273d" />


---

##  The Core Concept
Most multimodal systems rely on massive compute or pre-aligned models. This project explores the **"Surgery"** approach: Can we take two alien models—a Vision Encoder (CLIP) and a Language Decoder (GPT-2)—and force them to communicate by training only a microscopic "bridge" between them?

### The Architecture
* **Vision Encoder:** ViT-Large-Patch14 (Frozen)
* **The Bridge:** A custom-trained MLP Connector that projects **1024-d** visual features into the **768-d** text embedding space.
* **The Adaption:** LoRA (Low-Rank Adaptation) layers injected into GPT-2's attention blocks to learn the visual prefix semantics.

---

## 📸 Initial Results (Proof of Concept)

| Input Image | Model Prediction (Raw Output) |
| :--- | :--- |
| <img width="571" height="411" alt="image" src="https://github.com/user-attachments/assets/f1bb18a1-b3b0-4d20-ae3f-ac74f55eee2f" />
 | *"A group of people in a large temple . The people in the temple are surrounded by a large crowd..."* |
 

**Note on Output:** The looping behavior is a byproduct of the 124M parameter decoder's limited context and greedy search. However, the **semantic identification** of "Temple" and "Crowd" proves the projection layer successfully aligned the visual features to the correct text neighborhoods.

---

## 🛠 Status: Coming Soon
The repository is currently being refactored for readability. 

**What's being polished:**
* [ ] Cleaned training scripts for the MLP Connector.
* [ ] LoRA injection boilerplate for GPT-2 Small.
* [ ] Inference notebook demonstrating the 1024-to-768 dimension squeeze.

---

## 🧠 Why this matters
This isn't an API wrapper. This is an exploration of **Alignment Bottlenecks**. It demonstrates that even a "lobotomized" 124M parameter model can perform cross-modal reasoning if the projection bridge is mathematically sound.

---
*Created by [Your Name/Handle] - 2026 Research Series*
