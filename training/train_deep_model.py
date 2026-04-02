"""
Deep Learning CNN Training Pipeline for Steganalysis.
Trains a PyTorch CNN on SRM-residual features for robust image classification.
Supports mixed precision, early stopping, and checkpoint management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import config
from utils.feature_utils import iter_patches_with_coords
from utils.srm_filters import get_srm_filters, apply_srm_filters


# ============================================================================
# DATASET CLASS
# ============================================================================

class StegoDataset(Dataset):
    """
    PyTorch Dataset for steganalysis: loads images and extracts SRM residuals.
    """
    
    def __init__(self, image_paths, labels, srm_filters=None, patch_size=128):
        """
        Args:
            image_paths: list of Path objects pointing to images
            labels: list of labels (0=CLEAN, 1=STEGO)
            srm_filters: precomputed SRM filter kernels
            patch_size: size of patches to extract
        """
        self.image_paths = image_paths
        self.labels = labels
        self.srm_filters = srm_filters or get_srm_filters()
        self.patch_size = patch_size
        
        assert len(image_paths) == len(labels), "Paths and labels length mismatch"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            (residuals_tensor, label_tensor) where residuals is (num_patches, num_filters, H, W)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            # Fallback: return zeros
            return torch.zeros(1, 30, self.patch_size, self.patch_size), torch.tensor(label, dtype=torch.long)
        
        # Extract patches
        patches = []
        for patch, y, x in iter_patches_with_coords(img, self.patch_size):
            patches.append(patch)
        
        if not patches:
            return torch.zeros(1, 30, self.patch_size, self.patch_size), torch.tensor(label, dtype=torch.long)
        
        # FIX #1: Use ALL patches instead of just first patch (match inference)
        # Extract residuals for ALL patches
        residuals_list = []
        for patch in patches:
            residuals = apply_srm_filters(patch, self.srm_filters)  # (H, W, 30)
            residuals_list.append(residuals)
        
        # Aggregate: average residuals across patches (matches inference)
        residuals = np.mean(residuals_list, axis=0)  # (H, W, 30)
        
        # Convert to tensor: (num_filters, H, W)
        residuals_tensor = torch.from_numpy(residuals.transpose(2, 0, 1)).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return residuals_tensor, label_tensor


# ============================================================================
# CNN ARCHITECTURE
# ============================================================================

class SteganalysisNet(nn.Module):
    """
    Compact CNN for steganalysis on SRM residuals.
    Input: (batch, 30, 128, 128) — SRM residuals from 128x128 patches
    Output: (batch, 2) — logits for (CLEAN, STEGO)
    """
    
    def __init__(self, num_filters=30, num_classes=2):
        super(SteganalysisNet, self).__init__()
        
        # Conv block 1: Extract local patterns
        self.conv1 = nn.Conv2d(num_filters, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: (64, 64, 64)
        
        # Conv block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: (128, 32, 32)
        
        # Conv block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)  # Output: (256, 16, 16)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """x: (batch, 30, 128, 128)"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, use_mixed_precision=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        with torch.autocast(device, enabled=use_mixed_precision):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def train_deep_model(
    train_loader,
    val_loader,
    num_epochs=50,
    learning_rate=0.001,
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_mixed_precision=False,
    checkpoint_dir=None
):
    """
    Train the steganalysis CNN.
    
    Args:
        train_loader: PyTorch DataLoader for training
        val_loader: PyTorch DataLoader for validation
        num_epochs: number of training epochs
        learning_rate: initial learning rate
        device: "cuda" or "cpu"
        use_mixed_precision: enable mixed precision training
        checkpoint_dir: directory to save checkpoints
        
    Returns:
        dict with training history and model
    """
    
    model = SteganalysisNet(num_filters=30, num_classes=2).to(device)
    
    # Weighted loss to handle class imbalance (4,986 CLEAN vs 2,986 STEGO)
    # Weight ratio: 4986/2986 = 1.67
    # Give more importance to STEGO class
    class_weights = torch.tensor([1.0, 1.67], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    if checkpoint_dir is None:
        checkpoint_dir = config.PROJECT_ROOT / "checkpoints"
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_val_acc": 0.0,
        "best_epoch": -1
    }
    
    patience = 20  # Increased from 10 to allow longer training
    patience_counter = 0
    
    print(f"Training on device: {device}")
    print(f"Mixed precision: {use_mixed_precision}")
    print(f"=" * 60)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_mixed_precision)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:6.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:6.2f}%")
        
        if val_acc > history["best_val_acc"]:
            history["best_val_acc"] = val_acc
            history["best_epoch"] = epoch + 1
            patience_counter = 0
            
            # Save best checkpoint
            checkpoint_path = checkpoint_dir / f"best_model.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "history": history
            }, checkpoint_path)
            print(f"  -> Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"=" * 60)
    print(f"Best validation accuracy: {history['best_val_acc']:.2f}% at epoch {history['best_epoch']}")
    
    # Load best model
    best_checkpoint = checkpoint_dir / "best_model.pt"
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    history["model"] = model
    history["device"] = device
    
    return history


if __name__ == "__main__":
    print("CNN Training module ready.")
    print(f"Expected input shape: (batch, 30, 128, 128)")
    print(f"Expected output shape: (batch, 2)")
    
    # Quick test on dummy data
    dummy_input = torch.randn(4, 30, 128, 128)
    model = SteganalysisNet()
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print()
    
    # Start actual training
    print("=" * 60)
    print("PREPARING DATASET...")
    print("=" * 60)
    
    # Collect all images - USE AUGMENTED DATASET
    dataset_path = Path("DATASET_AUGMENTED")  # Changed from DATASET to DATASET_AUGMENTED
    clean_paths = []
    stego_paths = []
    
    # CLEAN images
    for folder in ["CLEAN/INTERNET_PNG", "CLEAN/INTERNET_BMP", "CLEAN/PNG", "CLEAN/BMP", "CLEAN/INTERNET"]:
        folder_path = dataset_path / folder
        if folder_path.exists():
            images = list(folder_path.glob("*.png")) + list(folder_path.glob("*.bmp"))
            clean_paths.extend(images)
            print(f"  ✓ {folder}: {len(images)} images")
    
    # STEGO images
    for folder in ["STEGO/INTERNET_PNG", "STEGO/INTERNET_BMP", "STEGO/PNG", "STEGO/BMP"]:
        folder_path = dataset_path / folder
        if folder_path.exists():
            images = list(folder_path.glob("*.png")) + list(folder_path.glob("*.bmp"))
            stego_paths.extend(images)
            print(f"  ✓ {folder}: {len(images)} images")
    
    print()
    print(f"Total CLEAN: {len(clean_paths)}")
    print(f"Total STEGO: {len(stego_paths)}")
    print(f"Total images: {len(clean_paths) + len(stego_paths)}")
    print()
    
    # Create labels (0=CLEAN, 1=STEGO)
    all_paths = clean_paths + stego_paths
    all_labels = [0] * len(clean_paths) + [1] * len(stego_paths)
    
    # Shuffle data
    from sklearn.model_selection import train_test_split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    print(f"Training set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    print()
    
    # Get SRM filters
    print("Loading SRM filters...")
    srm_filters = get_srm_filters()
    print(f"  ✓ {len(srm_filters)} SRM filters loaded")
    print()
    
    # Create datasets
    print("Creating PyTorch datasets...")
    train_dataset = StegoDataset(train_paths, train_labels, srm_filters, patch_size=config.PATCH_SIZE)
    val_dataset = StegoDataset(val_paths, val_labels, srm_filters, patch_size=config.PATCH_SIZE)
    print("  ✓ Datasets created")
    print()
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"  ✓ Train batches: {len(train_loader)}")
    print(f"  ✓ Val batches: {len(val_loader)}")
    print()
    
    # Start training
    print("=" * 60)
    print("STARTING TRAINING...")
    print("=" * 60)
    print()
    
    checkpoint_dir = Path("checkpoints/stego_detector_cnn")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    history = train_deep_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=300,
        learning_rate=0.0001,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_mixed_precision=True,
        checkpoint_dir=checkpoint_dir
    )
    
    print()
    print("=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Best validation accuracy: {history['best_val_acc']:.2f}%")
    print(f"Best epoch: {history['best_epoch']}")
    print()
    print("Model saved to: checkpoints/stego_detector_cnn/best_model.pt")
    print()
    
    # Copy to production location
    import shutil
    
    src = Path("checkpoints/stego_detector_cnn/best_model.pt")
    dst = Path("stego_detector_cnn_model.pt")
    
    if src.exists():
        # Extract just the model state dict
        checkpoint = torch.load(src, map_location='cpu')
        torch.save(checkpoint['model_state_dict'], dst)
        print(f"✓ Model copied to production: {dst}")
        print(f"  Validation accuracy: {checkpoint['val_acc']:.2f}%")
    else:
        print("⚠ Warning: Checkpoint not found!")
        print(f"  Looked for: {src}")

