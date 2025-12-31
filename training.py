import os
import time
import random
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 32    
EPOCHS = 10          
LEARNING_RATE = 1e-4
DATA_ROOT = "Prompt_Segmentation" 

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"random seed set: {seed}")

class PromptedDrywallDataset(Dataset):
    def __init__(self, root_dir, processor, split='train'):
        self.samples = []
        self.processor = processor
        dataset_folders = [("cracks-1", "segment crack"), ("Drywall-Join-Detect-1", "segment taping area")]
        
        found_data = False
        for folder_name, prompt in dataset_folders:
            search_path = Path(root_dir) / folder_name / split
            mask_dir = search_path / "masks"
            if not search_path.exists():
                print(f" Warning: Could not find {search_path}")
                continue
            
            valid_exts = {".jpg", ".jpeg", ".png"}
            for img_path in search_path.iterdir():
                if img_path.suffix.lower() in valid_exts and "mask" not in img_path.name:
                    mask_name = img_path.name.replace(".jpg", ".png").replace(".jpeg", ".png")
                    mask_path = mask_dir / mask_name
                    if mask_path.exists():
                        self.samples.append({"image": str(img_path), "mask": str(mask_path), "prompt": prompt})
                        found_data = True
        if not found_data: print(f" CRITICAL: No data found for split '{split}'")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image"]).convert("RGB")
        mask = Image.open(item["mask"]).convert("L")
        inputs = self.processor(text=[item["prompt"]], images=[image], padding="max_length", return_tensors="pt")
        mask = mask.resize((352, 352), resample=Image.NEAREST)
        mask = np.array(mask) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        return inputs, torch.tensor(mask).unsqueeze(0)

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    
    for name, param in model.clip.named_parameters():
        param.requires_grad = False
    model.to(device)
    
    print("oading Datasets...")
    train_dataset = PromptedDrywallDataset(DATA_ROOT, processor, split='train')
    valid_dataset = PromptedDrywallDataset(DATA_ROOT, processor, split='valid')
    
    if len(train_dataset) == 0: return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.decoder.parameters(), lr=LEARNING_RATE)
    loss_fn = BCEWithLogitsLoss()

    start_time = time.time()
    writer = SummaryWriter(log_dir="runs/drywall_experiment_1")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, batch in progress_bar:
            inputs, gt_masks = batch
            input_ids = inputs["input_ids"].squeeze(1).to(device)
            pixel_values = inputs["pixel_values"].squeeze(1).to(device)
            attention_mask = inputs["attention_mask"].squeeze(1).to(device)
            gt_masks = gt_masks.to(device)
            
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
            preds = outputs.logits.unsqueeze(1)
            
            if preds.shape[-2:] != gt_masks.shape[-2:]:
                preds = torch.nn.functional.interpolate(preds, size=gt_masks.shape[-2:], mode='bilinear')

            loss = loss_fn(preds, gt_masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Loss/train", loss.item(), global_step)
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
        print(f"epoch {epoch+1} avg. loss: {train_loss / len(train_loader):.4f}")
        
        save_path = f"clipseg_drywall_finetuned_ep{epoch+1}"
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        print(f"   Saved model to {save_path}")
    
    total_train_time = time.time() - start_time
    print(f" Total Training Time: {total_train_time / 60:.2f} minutes")
    writer.close()

    print("\nMeasuring Inference Speed...")
    model.eval()
    timing_start = time.time()
    num_samples = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Benchmarking"):
            input_ids = batch[0]["input_ids"].squeeze(1).to(device)
            pixel_values = batch[0]["pixel_values"].squeeze(1).to(device)
            attention_mask = batch[0]["attention_mask"].squeeze(1).to(device)
            _ = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
            num_samples += input_ids.shape[0]
    
    avg_time = ((time.time() - timing_start) / num_samples) * 1000
    print(f"avg. inference time: {avg_time:.2f} ms/image")

    model_file = Path(save_path) / "pytorch_model.bin"
    if not model_file.exists():
         model_file = Path(save_path) / "model.safetensors"

    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"model size: {size_mb:.2f} MB")

    print("complete.")

if __name__ == "__main__":
    main()