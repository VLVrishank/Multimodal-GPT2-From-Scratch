import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import GPT2Tokenizer, AutoProcessor
from peft import PeftModel

from model import VisionGPT2

# --- SETUP ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_ID = "openai/clip-vit-large-patch14"
GPT_ID = "gpt2"

def load_trained_model():
    print("Loading Processors and Base Model...")
    processor = AutoProcessor.from_pretrained(CLIP_ID)
    tokenizer = GPT2Tokenizer.from_pretrained(GPT_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Base Architecture
    model = VisionGPT2(clip_id=CLIP_ID, gpt_id=GPT_ID, device=DEVICE).to(DEVICE)
    
    # Load Saved LoRA Weights
    print("Loading Trained Weights...")
    model.llm = PeftModel.from_pretrained(model.llm, "saved_weights/vlm_lora")
    
    # Load Saved Projector Weights
    model.projector.load_state_dict(torch.load("saved_weights/vlm_projector.pth", map_location=DEVICE, weights_only=True))
    
    model.eval()
    return model, processor, tokenizer

def generate_caption(image, model, processor, tokenizer):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
    
    with torch.no_grad():
        img_feats = model.vision(pixel_values).last_hidden_state
        img_embeds = model.projector(img_feats)
        
        out = model.llm.generate(
            inputs_embeds=img_embeds, 
            max_new_tokens=30,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
    return tokenizer.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    model, processor, tokenizer = load_trained_model()
    
    print("\nFetching test image...")
    # You can replace this URL with a local path like Image.open("my_dog.jpg")
    url = "https://images.unsplash.com/photo-1517423440428-a5a00ad493e8" 
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    
    print("Generating Caption...")
    caption = generate_caption(img, model, processor, tokenizer)
    print(f"\nGenerated Caption: {caption}")
