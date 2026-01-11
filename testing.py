import argparse
import logging
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_coeff, multiclass_dice_coeff

def calculate_iou(pred_mask, true_mask, n_classes):
    """Calculates Intersection over Union (IoU) / Jaccard Index."""
    if n_classes > 1:
        pred_one_hot = F.one_hot(pred_mask.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
        true_one_hot = F.one_hot(true_mask, n_classes).permute(0, 3, 1, 2).float()
        inter = (pred_one_hot * true_one_hot).sum(dim=(2, 3))
        union = (pred_one_hot + true_one_hot).sum(dim=(2, 3)) - inter
        iou = (inter + 1e-6) / (union + 1e-6)
        return iou.mean().item()
    else:
        pred_bool = (pred_mask > 0.5).float()
        inter = (pred_bool * true_mask).sum()
        union = pred_bool.sum() + true_mask.sum() - inter
        return ((inter + 1e-6) / (union + 1e-6)).item()

def save_visual_result(img_tensor, true_mask, pred_tensor, output_path, n_classes):
    """
    Uses Matplotlib to plot and save the Input, Ground Truth, and Prediction side-by-side.
    """
    # 1. Prepare Image (CHW -> HWC)
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    # If image was normalized to 0-1, scale to display; if needed clip to valid range
    img = np.clip(img, 0, 1)

    # 2. Prepare Masks
    if n_classes > 1:
        pred_mask = pred_tensor.argmax(dim=0).cpu().numpy()
    else:
        pred_mask = (torch.sigmoid(pred_tensor) > 0.5).cpu().numpy().squeeze()
    
    true_mask = true_mask.cpu().numpy()

    # 3. Create Plot using Matplotlib
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input Image
    ax[0].imshow(img)
    ax[0].set_title("Input Image")
    ax[0].axis('off')

    # Ground Truth
    ax[1].imshow(true_mask, cmap='viridis', interpolation='nearest')
    ax[1].set_title("Ground Truth Mask")
    ax[1].axis('off')

    # Prediction
    ax[2].imshow(pred_mask, cmap='viridis', interpolation='nearest')
    ax[2].set_title("Predicted Mask")
    ax[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig) # Close figure to free memory

def plot_metrics_graph(dice_scores, iou_scores, output_dir):
    """
    Plots a graph of Dice Scores across the test set to identify outliers.
    """
    # Plot Dice Scores
    plt.figure(figsize=(12, 6))
    plt.plot(dice_scores, marker='o', linestyle='-', color='b', label='Dice Score', alpha=0.7)
    plt.plot(iou_scores, marker='x', linestyle='--', color='r', label='IoU', alpha=0.5)
    
    plt.title(f"Segmentation Performance per Image ({len(dice_scores)} Total)")
    plt.xlabel("Test Image Index")
    plt.ylabel("Score (0.0 - 1.0)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(0, 1.05)
    
    graph_path = os.path.join(output_dir, "test_metrics_graph.png")
    plt.savefig(graph_path)
    plt.close()
    logging.info(f"Metrics graph saved to {graph_path}")

def test_net(net, device, dir_img, dir_mask, output_dir, scale=1.0, mask_threshold=0.5):
    dataset = BasicDataset(dir_img, dir_mask, scale)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    
    n_val = len(loader)
    logging.info(f'Starting testing on {n_val} images...')
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dice_scores_list = []
    iou_scores_list = []
    
    net.eval()
    
    with tqdm(total=n_val, desc='Testing', unit='img') as pbar:
        for i, batch in enumerate(loader):
            image, true_mask = batch['image'], batch['mask']
            
            image = image.to(device=device, dtype=torch.float32)
            true_mask = true_mask.to(device=device, dtype=torch.long)

            with torch.no_grad():
                pred_mask = net(image)

                # --- Calculate Metrics ---
                if net.n_classes == 1:
                    pred_prob = torch.sigmoid(pred_mask)
                    pred_flat = (pred_prob > mask_threshold).float()
                    dice = dice_coeff(pred_flat, true_mask, reduce_batch_first=False)
                else:
                    true_one_hot = F.one_hot(true_mask, net.n_classes).permute(0, 3, 1, 2).float()
                    pred_one_hot = F.one_hot(pred_mask.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                    dice = multiclass_dice_coeff(pred_one_hot[:, 1:], true_one_hot[:, 1:], reduce_batch_first=False)

                iou = calculate_iou(pred_mask, true_mask, net.n_classes)

                # Store for graph
                dice_scores_list.append(dice.item())
                iou_scores_list.append(iou)

                # --- Save Visuals (Matplotlib) ---
                if output_dir:
                    save_path = os.path.join(output_dir, f"result_{i:03d}.png")
                    save_visual_result(image[0], true_mask[0], pred_mask[0], save_path, net.n_classes)

            pbar.update()

    # --- Summary ---
    avg_dice = sum(dice_scores_list) / n_val
    avg_iou = sum(iou_scores_list) / n_val
    
    logging.info(f'Testing Finished!')
    logging.info(f'Average Dice Score: {avg_dice:.4f}')
    logging.info(f'Average IoU (Jaccard): {avg_iou:.4f}')

    # --- Plot Metrics Graph ---
    if output_dir:
        plot_metrics_graph(dice_scores_list, iou_scores_list, output_dir)

def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet on images and target masks')
    parser.add_argument('--model', '-m', type=str, required=True, help='Path to the .pth model checkpoint')
    parser.add_argument('--images', '-i', type=str, required=True, help='Path to directory of test images')
    parser.add_argument('--masks', '-g', type=str, required=True, help='Path to directory of ground truth masks')
    parser.add_argument('--output', '-o', type=str, default='test_results', help='Directory to save visual results')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Load Model
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    
    logging.info(f'Loading model from {args.model}')
    try:
        state_dict = torch.load(args.model, map_location=device)
        if 'mask_values' in state_dict:
            del state_dict['mask_values']
        net.load_state_dict(state_dict)
        logging.info('Model loaded successfully!')
    except Exception as e:
        logging.error(f'Failed to load model: {e}')
        exit(1)

    net.to(device=device)

    # Run Test
    test_net(net, device, args.images, args.masks, args.output, args.scale)