"""
Improved Segmentation Training Script
- Fixed mask resizing
- Joint augmentation
- CE + Dice loss
- Cosine LR scheduler
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
from tqdm import tqdm

# ============================================================================
# Mask Mapping
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


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Joint Transform 
# ============================================================================

class JointTransform:
    def __init__(self, h, w):
        self.h = h
        self.w = w
        self.color = transforms.ColorJitter(0.2, 0.2, 0.2)

    def __call__(self, image, mask):
        image = TF.resize(image, (self.h, self.w))
        mask = TF.resize(mask, (self.h, self.w),
                         interpolation=transforms.InterpolationMode.NEAREST)

        if torch.rand(1) > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = self.color(image)

        image = TF.to_tensor(image)
        image = TF.normalize(image,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask


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
        return image, mask


# ============================================================================
# Segmentation Head
# ============================================================================

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
# Dice Loss
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, n_classes).permute(0, 3, 1, 2).float()

        intersection = (probs * one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


# ============================================================================
# Training
# ============================================================================

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    batch_size = 4
    h = w = 448
    lr = 3e-5
    n_epochs = 50

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..',
                            'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, '..',
                           'Offroad_Segmentation_Training_Dataset', 'val')

    transform = JointTransform(h, w)

    trainset = MaskDataset(data_dir, transform)
    valset = MaskDataset(val_dir, transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Load DINOv2 backbone
    backbone = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval().to(device)

    # Get embedding size
    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        out = backbone.forward_features(imgs)["x_norm_patchtokens"]

    embed_dim = out.shape[2]

    classifier = SegmentationHead(
        embed_dim, n_classes, w // 14, h // 14).to(device)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()

    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs)

    for epoch in range(n_epochs):

        classifier.train()
        total_loss = 0

        for imgs, labels in tqdm(train_loader):

            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                features = backbone.forward_features(imgs)[
                    "x_norm_patchtokens"]

            logits = classifier(features)
            logits = F.interpolate(
                logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            loss = ce_loss(logits, labels) + dice_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {total_loss/len(train_loader):.4f}")

    torch.save(classifier.state_dict(),
               os.path.join(script_dir, "segmentation_head.pth"))

    print("Training complete 🚀")


if __name__ == "__main__":
    main()

