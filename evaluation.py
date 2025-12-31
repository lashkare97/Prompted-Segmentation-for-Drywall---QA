import os
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from tqdm import tqdm

MODEL_PATH = "model/path/here" # IF TRAINED MODEL IS REQUIRED TO BE SUBMITTED, PLEASE LET ME KNOW 
DATA_ROOT = "Prompt_Segmentation"
OUTPUT_DIR = "predictions"
VISUAL_DIR = "report_examples" 

DATASET_MAP = [
    ("cracks-1", "segment crack"),
    ("Drywall-Join-Detect-1", "segment taping area")
]

def calculate_iou(pred_mask, gt_mask):
    pred = pred_mask.flatten()
    gt = gt_mask.flatten()
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0: return 1.0
    return intersection / union

def calculate_dice(pred_mask, gt_mask):
    pred = pred_mask.flatten()
    gt = gt_mask.flatten()
    intersection = np.logical_and(pred, gt).sum()
    sum_pixels = pred.sum() + gt.sum()
    if sum_pixels == 0: return 1.0 
    return (2.0 * intersection) / sum_pixels

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        processor = CLIPSegProcessor.from_pretrained(MODEL_PATH)
        model = CLIPSegForImageSegmentation.from_pretrained(MODEL_PATH)
    except OSError:
        print(f"could not find model")
        return

    model.to(device)
    model.eval()

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.exists(VISUAL_DIR): os.makedirs(VISUAL_DIR)

    results = {
        "segment crack": {"iou": [], "dice": []},
        "segment taping area": {"iou": [], "dice": []}
    }

    for folder_name, prompt in DATASET_MAP:
        base_path = Path(DATA_ROOT) / folder_name
        test_path = base_path / "test"
        if not test_path.exists():
            print(f"'test' folder not found")
            test_path = base_path / "valid"
        
        if not test_path.exists():
            print(f"skipping {folder_name}: Neither 'test' nor 'valid' found.")
            continue
            
        print(f"processing {folder_name} ({prompt})")
        
        image_files = [f for f in test_path.iterdir() if f.suffix.lower() in ['.jpg', '.png'] and "mask" not in f.name]
        saved_visuals = 0 
        
        for img_file in tqdm(image_files):
            original_image = Image.open(img_file).convert("RGB")
            original_size = original_image.size
            
            inputs = processor(
                text=[prompt], 
                images=[original_image], 
                padding="max_length", 
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits.unsqueeze(1)
            
            logits_resized = torch.nn.functional.interpolate(
                logits, size=(original_size[1], original_size[0]), mode='bilinear')
            preds = torch.sigmoid(logits_resized).cpu().numpy()[0, 0]
            
            binary_mask = (preds > 0.5).astype(np.uint8) * 255
            
            image_id = img_file.name.rsplit('.', 1)[0]
            sanitized_prompt = prompt.replace(" ", "_")
            save_name = f"{image_id}__{sanitized_prompt}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), binary_mask)
            
            gt_name = img_file.name.replace(".jpg", ".png").replace(".jpeg", ".png")
            gt_path = test_path / "masks" / gt_name
            
            if gt_path.exists():
                gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                if gt_mask is not None:
                    if gt_mask.shape != binary_mask.shape:
                        gt_mask = cv2.resize(gt_mask, (binary_mask.shape[1], binary_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

                    gt_bool = (gt_mask > 127).astype(bool)
                    pred_bool = (binary_mask > 127).astype(bool)
                    
                    iou = calculate_iou(pred_bool, gt_bool)
                    dice = calculate_dice(pred_bool, gt_bool)
                    
                    results[prompt]["iou"].append(iou)
                    results[prompt]["dice"].append(dice)

                    if saved_visuals < 3:
                        orig_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                        gt_color = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
                        pred_color = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
                        
                        combined = np.hstack([orig_cv, gt_color, pred_color])
                        
                        vis_save_path = os.path.join(VISUAL_DIR, f"{image_id}_report.jpg")
                        cv2.imwrite(vis_save_path, combined)
                        saved_visuals += 1

    for prompt, metrics in results.items():
        if len(metrics["iou"]) > 0:
            avg_iou = sum(metrics["iou"]) / len(metrics["iou"])
            avg_dice = sum(metrics["dice"]) / len(metrics["dice"])
            print(f"Prompt: '{prompt}'")
            print(f"mIoU: {avg_iou:.4f}")
            print(f"Dice: {avg_dice:.4f}")
        else:
            print(f"GT not found.")

if __name__ == "__main__":
    main()
