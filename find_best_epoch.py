import argparse
import logging
import os
import torch
import glob
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from utils.data_loading import BasicDataset
from evaluate import evaluate

# --- CONFIGURATION ---
# Path to your checkpoints folder
CHECKPOINT_DIR = r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Code\Pytorch-UNet-master\checkpoints\Combined"

# Path to the dataset you want to use for selection (e.g., your Test Set)
#
VAL_IMG_DIR = r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Bacterial Leaf Blight\Infer_Ori"
VAL_MASK_DIR = r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Bacterial Leaf Blight\Infer_GT"
# ---------------------

def get_args():
    parser = argparse.ArgumentParser(description='Find the best epoch from saved checkpoints')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Scale factor for input images')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 1. Setup Dataset
    # We use batch_size=1 for accurate evaluation
    dataset = BasicDataset(VAL_IMG_DIR, VAL_MASK_DIR, scale=args.scale)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    logging.info(f'Evaluating on {len(dataset)} images')

    # 2. Find all checkpoints
    # Matches "checkpoint_epoch1.pth", "checkpoint_epoch2.pth", etc.
    checkpoint_paths = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch*.pth"))
    
    if not checkpoint_paths:
        logging.error(f"No checkpoints found in {CHECKPOINT_DIR}")
        exit()

    # Sort by epoch number to make the output readable
    # Assuming format "checkpoint_epochX.pth"
    try:
        checkpoint_paths.sort(key=lambda x: int(Path(x).stem.replace('checkpoint_epoch', '')))
    except ValueError:
        pass # Sort normally if naming is different

    logging.info(f"Found {len(checkpoint_paths)} checkpoints. Starting evaluation...")

    results = []
    best_score = 0
    best_epoch = ""

    # 3. Iterate and Evaluate
    for cp_path in tqdm(checkpoint_paths, desc="Scanning Epochs"):
        net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        net.to(device)
        
        # Load weight
        state_dict = torch.load(cp_path, map_location=device)
        state_dict.pop('mask_values', None) # Remove if present
        net.load_state_dict(state_dict)
        
        # Calculate Dice Score
        # evaluate() returns the average dice score for the loader
        #
        score = evaluate(net, loader, device, amp=False)
        
        filename = Path(cp_path).name
        results.append({'Checkpoint': filename, 'Dice Score': score})
        
        if score > best_score:
            best_score = score
            best_epoch = filename

    # 4. Report Results
    print("\n" + "="*40)
    print(f"🏆 BEST MODEL: {best_epoch}")
    print(f"💎 SCORE: {best_score:.5f}")
    print("="*40 + "\n")

    df = pd.DataFrame(results)
    df = df.sort_values(by='Dice Score', ascending=False)
    print(df.head(10).to_string(index=False))
    
    # Save to file
    df.to_csv("checkpoint_ranking.csv", index=False)
    print("\nFull ranking saved to 'checkpoint_ranking.csv'")