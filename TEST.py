"""
Segmentation Validation Script
Converted from val_mask.ipynb
Evaluates a trained segmentation head on validation data and saves predictions
Aligned with train.py
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm

plt.switch_backend('Agg')


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion 
# ============================================================================

value_map = {
    0: 0,
    100: 1,
    200: 2,
    300: 3,
    500: 4,
    550: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9
}

n_classes = len(value_map)

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

color_palette = np.array([
    [0, 0, 0],
    [34, 139, 34],
    [0, 255, 0],
    [210, 180, 140],
    [139, 90, 43],
    [128, 128, 0],
    [139, 69, 19],
    [128, 128, 128],
    [160, 82, 45],
    [135, 206, 235],
], dtype=np.uint8)


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask


# ============================================================================
# Dataset 
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        image, mask = self.transform(image, mask)

        return image, mask, data_id


# ============================================================================
# Test Transform 
# ============================================================================

class TestTransform:
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, image, mask):

        image = TF.resize(image, (self.h, self.w))
        mask = TF.resize(mask, (self.h, self.w),
                         interpolation=transforms.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask


# Segmentation Head 

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, out_channels, 1)
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.net(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target):
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []

    for class_id in range(n_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class), iou_per_class


# ============================================================================
# Main Validation Function
# ============================================================================

def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Segmentation evaluation script')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(script_dir, 'segmentation_head.pth'))
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./predictions')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=5)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    
    h = w = 448

    transform = TestTransform(h, w)

    print(f"Loading dataset from {args.data_dir}...")
    valset = MaskDataset(args.data_dir, transform)
    val_loader = DataLoader(valset,
                            batch_size=args.batch_size,
                            shuffle=False)

    print(f"Loaded {len(valset)} samples")

    print("Loading DINOv2 backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval().to(device)

    imgs, _, _ = next(iter(val_loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        out = backbone.forward_features(imgs)["x_norm_patchtokens"]

    embed_dim = out.shape[2]

    print(f"Loading model from {args.model_path}...")
    classifier = SegmentationHead(
        embed_dim, n_classes, w // 14, h // 14
    ).to(device)

    classifier.load_state_dict(torch.load(args.model_path, map_location=device))
    classifier.eval()

    print("Model loaded successfully!")

    print("\nRunning evaluation...")

    iou_scores = []
    all_class_iou = []
    sample_count = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Processing", unit="batch")

        for imgs, labels, data_ids in pbar:

            imgs = imgs.to(device)
            labels = labels.to(device)

            features = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = classifier(features)

            outputs = F.interpolate(
                logits,
                size=imgs.shape[2:],
                mode="bilinear",
                align_corners=False
            )

            iou, class_iou = compute_iou(outputs, labels)

            iou_scores.append(iou)
            all_class_iou.append(class_iou)

            pbar.set_postfix(iou=f"{iou:.3f}")

    mean_iou = np.nanmean(iou_scores)
    avg_class_iou = np.nanmean(all_class_iou, axis=0)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean IoU: {mean_iou:.4f}")
    print("=" * 50)

    for name, iou in zip(class_names, avg_class_iou):
        print(f"{name:<20}: {iou:.4f}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
