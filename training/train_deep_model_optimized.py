"""
Optimized Deep Model Training - ResNet-style architecture
Includes:
- Deeper CNN (5 residual blocks instead of 3)
- Optimized hyperparameters
- Better regularization (dropout, batch norm)
- Learning rate scheduling
- Early stopping with patience

Usage:
    python train_deep_model_optimized.py --epochs 200 --batch-size 32 --lr 0.001
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import json
import logging
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import config
from utils.srm_filters import get_srm_filters, apply_srm_filters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class OptimizedSteganalysisNet(nn.Module):
    """Deeper CNN with residual connections - Optimized for production."""
    
    def __init__(self, num_filters=30, num_classes=2):
        super().__init__()
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(num_filters, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks (5 blocks = deeper architecture)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers with dropout
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 256)
        self.fc1_relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout1(out)
        out = self.fc1_relu(self.fc1(out))
        out = self.dropout2(out)
        out = self.fc2(out)
        return out


class StegoDataset(Dataset):
    """Dataset for steganography detection."""
    
    def __init__(self, clean_dirs, stego_dirs, srm_filters, patch_size=128, max_images=None):
        self.srm_filters = srm_filters
        self.patch_size = patch_size
        self.images = []
        self.labels = []
        
        # Load clean images
        for clean_dir in clean_dirs:
            clean_dir = Path(clean_dir)
            if not clean_dir.exists():
                continue
            
            for img_path in sorted(clean_dir.glob('*'))[:max_images]:
                if img_path.is_file() and img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    self.images.append(str(img_path))
                    self.labels.append(0)  # Clean
        
        # Load stego images
        for stego_dir in stego_dirs:
            stego_dir = Path(stego_dir)
            if not stego_dir.exists():
                continue
            
            for img_path in sorted(stego_dir.glob('*'))[:max_images]:
                if img_path.is_file() and img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    self.images.append(str(img_path))
                    self.labels.append(1)  # Stego
        
        logger.info(f"Loaded {len(self.images)} images ({sum(1 for l in self.labels if l==0)} clean, {sum(1 for l in self.labels if l==1)} stego)")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        img = cv2.imread(img_path)
        if img is None:
            # Return zero tensor if image fails to load
            return torch.zeros(30, self.patch_size, self.patch_size), label
        
        # Ensure correct size
        if img.shape[0] < self.patch_size or img.shape[1] < self.patch_size:
            img = cv2.resize(img, (self.patch_size, self.patch_size))
        
        # Extract random patch
        h, w = img.shape[:2]
        y = np.random.randint(0, max(1, h - self.patch_size + 1))
        x = np.random.randint(0, max(1, w - self.patch_size + 1))
        patch = img[y:y+self.patch_size, x:x+self.patch_size]
        
        # Apply SRM filters
        residuals = apply_srm_filters(patch, self.srm_filters)
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(residuals.transpose(2, 0, 1)).float()
        
        return tensor, label


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training (optional, for faster training)
        if scaler:
            with torch.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'}, refresh=False)
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train optimized deep model')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("OPTIMIZED DEEP CNN TRAINING")
    logger.info("="*70)
    logger.info(f"Hyperparameters:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  Early stopping patience: {args.patience}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load SRM filters
    srm_filters = get_srm_filters()
    logger.info(f"Loaded {len(srm_filters)} SRM filters")
    
    # Create datasets
    logger.info("Loading dataset...")
    train_dataset = StegoDataset(
        config.CLEAN_TRAIN_DIRS,
        config.STEGO_TRAIN_DIRS,
        srm_filters,
        max_images=None  # Use all images
    )
    
    val_dataset = StegoDataset(
        [config.TEST_CLEAN_DIR],
        [config.TEST_STEGO_DIR],
        srm_filters,
    )
    
    # Create dataloaders (optimized for GPU utilization)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,  # Multi-threaded data loading (was 0)
        pin_memory=True,  # Faster GPU transfer
        prefetch_factor=2  # Prefetch batches
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,  # Multi-threaded validation
        pin_memory=True
    )
    
    # Create model
    logger.info("Creating model...")
    model = OptimizedSteganalysisNet(num_filters=30, num_classes=2).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Training loop
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_epoch': 0,
        'best_val_acc': 0
    }
    
    best_model_path = config.PROJECT_ROOT / "stego_detector_cnn_model_optimized.pt"
    patience_counter = 0
    
    logger.info("="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70)
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            history['best_epoch'] = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), str(best_model_path))
            logger.info(f"✓ Saved best model: {val_acc:.2f}%")
        else:
            patience_counter += 1
            logger.info(f"No improvement ({patience_counter}/{args.patience})")
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"\n⚠ Early stopping at epoch {epoch+1}")
            break
    
    # Save final results
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Best epoch: {history['best_epoch']}")
    logger.info(f"Best validation accuracy: {history['best_val_acc']:.2f}%")
    logger.info(f"Model saved: {best_model_path}")
    
    # Save history
    history_path = config.PROJECT_ROOT / "training_history_optimized.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"History saved: {history_path}")
    
    logger.info("\n✅ Next steps:")
    logger.info("1. Copy the model: cp stego_detector_cnn_model_optimized.pt stego_detector_cnn_model.pt")
    logger.info("2. Test the model: python test_inference.py")
    logger.info("3. (Optional) Quantize for speed: python quantize_model.py")


if __name__ == '__main__':
    main()
