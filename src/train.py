import os
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, AutoProcessor
from tqdm import tqdm

from dataset import FlickrDataset
from model import VisionGPT2

# --- SETUP & HYPERPARAMETERS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-4
MAX_LEN = 40
CLIP_ID = "openai/clip-vit-large-patch14"
GPT_ID = "gpt2"

def main():
    print("Loading Models & Processors...")
    processor = AutoProcessor.from_pretrained(CLIP_ID)
    tokenizer = GPT2Tokenizer.from_pretrained(GPT_ID)
    tokenizer.pad_token = tokenizer.eos_token  

    print("Loading Dataset...")
    dataset = FlickrDataset(processor=processor, tokenizer=tokenizer, max_len=MAX_LEN)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print("Initializing Model...")
    model = VisionGPT2(clip_id=CLIP_ID, gpt_id=GPT_ID, device=DEVICE).to(DEVICE)
    model.llm.print_trainable_parameters()

    # Optimizer: Only pass trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LR)

    # --- TRAINING LOOP ---
    print("\nStarting Training...")
    model.train()
    model.vision.eval() # Prevent vision dropout during training

    for epoch in range(EPOCHS):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            # Mask padding tokens in labels
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            
            optimizer.zero_grad()
            loss = model(pixel_values, input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=loss.item())

    # --- SAVE WEIGHTS ---
    print("\nSaving Model Weights...")
    os.makedirs("saved_weights", exist_ok=True)
    model.llm.save_pretrained("saved_weights/vlm_lora")
    torch.save(model.projector.state_dict(), "saved_weights/vlm_projector.pth")
    print("Training Complete! Weights saved to 'saved_weights/'")

if __name__ == "__main__":
    main()
