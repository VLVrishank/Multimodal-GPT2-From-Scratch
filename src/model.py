import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, CLIPVisionModel
from peft import LoraConfig, get_peft_model, TaskType

class VisionGPT2(nn.Module):
    def __init__(self, clip_id="openai/clip-vit-large-patch14", gpt_id="gpt2", device="cuda"):
        super().__init__()
        self.device = device
        
        # 1. Vision Encoder (Frozen)
        self.vision = CLIPVisionModel.from_pretrained(clip_id)
        for param in self.vision.parameters():
            param.requires_grad = False
            
        # 2. Language Model (Trainable with LoRA)
        self.llm = GPT2LMHeadModel.from_pretrained(gpt_id)
        
        # 3. Connector (Maps Image dim -> Text dim)
        self.projector = nn.Linear(1024, 768) 
        
        # Apply LoRA 
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            r=16, 
            lora_alpha=32, 
            target_modules=["c_attn"]
        )
        self.llm = get_peft_model(self.llm, peft_config)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # Get Image Features
        with torch.no_grad():
            img_feats = self.vision(pixel_values).last_hidden_state 
            
        # Project to GPT dimension
        img_embeds = self.projector(img_feats) 
        
        # Get Text Embeddings
        txt_embeds = self.llm.get_input_embeddings()(input_ids) 
        
        # Concatenate: [Image, Text]
        inputs_embeds = torch.cat([img_embeds, txt_embeds], dim=1)
        
        # Create Joint Attention Mask (1s for image tokens + text mask)
        img_mask = torch.ones((img_embeds.shape[0], img_embeds.shape[1]), dtype=torch.long, device=self.device)
        full_mask = torch.cat([img_mask, attention_mask], dim=1)
        
        # Pad Labels for the Image part 
        if labels is not None:
            img_labels = torch.full((img_embeds.shape[0], img_embeds.shape[1]), -100, device=self.device)
            full_labels = torch.cat([img_labels, labels], dim=1)
            return self.llm(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=full_labels).loss
            
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=full_mask)
