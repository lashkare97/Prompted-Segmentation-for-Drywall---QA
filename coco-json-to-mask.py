import os
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

def coco_to_masks(dataset_root_path):
    dataset_path = Path(dataset_root_path)
    
    if not dataset_path.exists():
        print(f"error: Could not find folder: {dataset_path}")
        print(f"current working directory is: {os.getcwd()}")
        return

    print(f"checking dataset: {dataset_path.name}")
    
    found_any = False
    for split in ['train', 'valid', 'test']:
        split_dir = dataset_path / split
        json_file = split_dir / "_annotations.coco.json"
        
        if not json_file.exists():
            print(f"skipping '{split}': No _annotations.coco.json found in {split_dir}")
            continue
            
        found_any = True
        print(f"processing '{split}'")
        
        with open(json_file, 'r') as f:
            coco = json.load(f)
            
        img_to_anns = {img['id']: [] for img in coco['images']}
        for ann in coco['annotations']:
            img_to_anns[ann['image_id']].append(ann)
            
        img_id_to_file = {img['id']: img['file_name'] for img in coco['images']}
        
        mask_dir = split_dir / "masks"
        mask_dir.mkdir(exist_ok=True)
        
        count = 0
        for img_id, filename in tqdm(img_id_to_file.items(), desc=f"converting {split}"):
            img_info = next((item for item in coco['images'] if item["id"] == img_id), None)
            if not img_info:
                continue
                
            h, w = img_info['height'], img_info['width']
            mask = np.zeros((h, w), dtype=np.uint8)
            
            anns = img_to_anns.get(img_id, [])
            for ann in anns:
                if 'segmentation' in ann and ann['segmentation']:
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                        cv2.fillPoly(mask, [poly], 255)
                elif 'bbox' in ann: # fallback
                    x, y, bw, bh = map(int, ann['bbox'])
                    cv2.rectangle(mask, (x, y), (x+bw, y+bh), 255, -1)
            
            mask_filename = filename.replace(".jpg", ".png").replace(".jpeg", ".png")
            cv2.imwrite(str(mask_dir / mask_filename), mask)
            count += 1
            
        print(f"created {count} masks in {mask_dir}")

    if not found_any:
        print("JSON files not found")

if __name__ == "__main__":
    base_folder = "Prompt_Segmentation"
    
    coco_to_masks(f"{base_folder}/cracks-1")
    coco_to_masks(f"{base_folder}/Drywall-Join-Detect-1")
    
    print("specific done")