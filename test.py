import argparse
import logging
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your project modules
from unet import UNet
from utils.data_loading import BasicDataset

# --- USER CONFIGURATION SECTION ---
#
# SPECIFY THE PATH TO YOUR TESTING DATASET HERE
TEST_IMG_DIR = r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Tungro\Infer_Ori"
TEST_MASK_DIR = r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Tungro\Infer_GT"
# ----------------------------------

def calculate_complexity(model, input_size, device):
    """Calculates Params and FLOPs."""
    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return params, flops
    except ImportError:
        logging.warning("Library 'thop' not installed. Skipping FLOPs calculation.")
        return sum(p.numel() for p in model.parameters()), 0

def calculate_metrics(pred_mask, true_mask):
    """Calculates binary classification metrics."""
    pred = pred_mask.view(-1).cpu().numpy()
    true = true_mask.view(-1).cpu().numpy()

    tp = np.sum((pred == 1) & (true == 1))
    tn = np.sum((pred == 0) & (true == 0))
    fp = np.sum((pred == 1) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))

    epsilon = 1e-7
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": iou,
        "Dice": dice
    }

def save_visual_result_matplotlib(image_tensor, true_mask_tensor, pred_mask_tensor, filename, dice_score, output_dir):
    """
    Saves a figure with 3 subplots: Original, Ground Truth, Prediction.
    Matches the style of eval_BLAST_691.png
    """
    # 1. Prepare Original Image
    # BasicDataset scales images by 255.0, so we reverse it to get 0-255 uint8
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)

    # 2. Prepare Masks (Binary 0 or 1)
    true_np = true_mask_tensor.cpu().numpy()
    pred_np = pred_mask_tensor.cpu().numpy()

    # 3. Create Figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    ax[0].imshow(img_np)
    ax[0].set_title(f"Original: {filename}", fontsize=12)
    ax[0].axis("off")

    # Ground Truth (Gray/Black background, White foreground)
    ax[1].imshow(true_np, cmap='gray', interpolation='nearest')
    ax[1].set_title("Ground Truth", fontsize=12)
    ax[1].axis("off")

    # Prediction
    ax[2].imshow(pred_np, cmap='gray', interpolation='nearest')
    ax[2].set_title(f"Pred (Dice: {dice_score:.2f})", fontsize=12)
    ax[2].axis("off")

    # 4. Save
    plt.tight_layout()
    save_path = output_dir / f"{filename}_eval.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet and generate metrics/images')
    parser.add_argument('--model', '-m', default='checkpoints\Combined\checkpoint_epoch43.pth', metavar='FILE', help='Path to model .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Scale factor for input images')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='Threshold for binary masks')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Validate paths
    if not os.path.exists(TEST_IMG_DIR) or not os.path.exists(TEST_MASK_DIR):
        logging.error("Testing directories not found! Check TEST_IMG_DIR paths.")
        exit(1)

    # Load Model
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    net.to(device)
    
    if not os.path.exists(args.model):
        logging.error(f"Model file {args.model} not found!")
        exit(1)

    logging.info(f'Loading model from {args.model}')
    state_dict = torch.load(args.model, map_location=device)
    state_dict.pop('mask_values', None)
    net.load_state_dict(state_dict)
    net.eval()

    # Load Data
    # shuffle=False is critical to map filenames correctly
    dataset = BasicDataset(TEST_IMG_DIR, TEST_MASK_DIR, scale=args.scale)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    logging.info(f'Testing on {len(dataset)} images')

    # Setup Output Directories
    base_output = Path("testing_output")
    img_output = base_output / "predictions"
    base_output.mkdir(exist_ok=True)
    img_output.mkdir(exist_ok=True)

    # Complexity
    sample_img = dataset[0]['image']
    params, flops = calculate_complexity(net, (sample_img.shape[1], sample_img.shape[2]), device)
    logging.info(f'Params: {params:,} | FLOPs: {flops:,}')

    results = []

    logging.info('Starting inference...')
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(loader), total=len(loader), desc='Processing'):
            image = batch['image'].to(device, dtype=torch.float32)
            true_mask = batch['mask'].to(device, dtype=torch.long)

            # Inference
            output = net(image)
            
            if net.n_classes > 1:
                pred_mask = output.argmax(dim=1)
            else:
                pred_mask = (torch.sigmoid(output) > args.mask_threshold).float().squeeze(1)

            # Calculate Metrics
            metrics = calculate_metrics(pred_mask[0], true_mask[0])
            dice_score = metrics['Dice']
            
            # Retrieve Filename using the index (Safe because shuffle=False)
            img_name = dataset.ids[idx]
            metrics['Image Name'] = img_name
            metrics['Params'] = params
            metrics['FLOPs'] = flops
            
            results.append(metrics)

            # Generate and Save Visual Result (Matplotlib)
            save_visual_result_matplotlib(
                image[0], 
                true_mask[0], 
                pred_mask[0], 
                img_name, 
                dice_score, 
                img_output
            )

    # --- Save Excel Report with Summary ---
    df = pd.DataFrame(results)
    
    # Reorder columns to put Image Name first
    cols = ['Image Name'] + [c for c in df.columns if c != 'Image Name']
    df = df[cols]

    # Create Summary Dataframe
    summary_stats = df.describe().loc[['mean', 'min', 'max', 'std']]
    
    excel_path = base_output / 'performance_metrics.xlsx'
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Detailed Results', index=False)
        summary_stats.to_excel(writer, sheet_name='Summary Metrics')
    
    logging.info(f"Testing complete.")
    logging.info(f"Visualizations saved to: {img_output}")
    logging.info(f"Excel report saved to: {excel_path}")
    
    print("\n--- Average Metrics ---")
    print(summary_stats.loc['mean'])