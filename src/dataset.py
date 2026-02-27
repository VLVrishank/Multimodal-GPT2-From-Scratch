import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class FlickrDataset(Dataset):
    def __init__(self, processor, tokenizer, split="train[:4000]", max_len=40):
        self.data = load_dataset("ariG23498/flickr8k", split=split)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process Image
        pixel_values = self.processor(images=item["image"], return_tensors="pt").pixel_values.squeeze(0)
        
        # Process Text
        tokens = self.tokenizer(
            item["caption"], 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids.squeeze(0),
            "attention_mask": tokens.attention_mask.squeeze(0) # Attention mask extracted here
        }
