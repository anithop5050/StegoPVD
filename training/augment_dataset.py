"""
Data Augmentation Script - Expand training dataset 5K → 20K images
Run this to multiply your dataset with various transformations.

Usage:
    python augment_dataset.py --output-size 20000 --strategies all
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import random
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import config


class DataAugmenter:
    """Augment images with various transformations."""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
    
    def rotate(self, img, angle):
        """Rotate image."""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    def zoom(self, img, scale):
        """Zoom in/out."""
        h, w = img.shape[:2]
        if scale > 1:  # Zoom in
            new_h, new_w = int(h / scale), int(w / scale)
            x, y = (w - new_w) // 2, (h - new_h) // 2
            cropped = img[y:y+new_h, x:x+new_w]
            return cv2.resize(cropped, (w, h))
        else:  # Zoom out
            new_h, new_w = int(h / scale), int(w / scale)
            canvas = np.ones((new_h, new_w, 3), dtype=np.uint8) * 128
            x, y = (new_w - w) // 2, (new_h - h) // 2
            canvas[y:y+h, x:x+w] = img
            return cv2.resize(canvas, (w, h))
    
    def flip(self, img, direction):
        """Flip image horizontally or vertically."""
        if direction == 'h':
            return cv2.flip(img, 1)
        else:
            return cv2.flip(img, 0)
    
    def adjust_brightness(self, img, delta):
        """Adjust brightness."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + delta, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def adjust_contrast(self, img, factor):
        """Adjust contrast."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * factor, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    def add_gaussian_blur(self, img, kernel_size):
        """Add Gaussian blur."""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    def add_noise(self, img, noise_type='gaussian', intensity=0.01):
        """Add noise to image."""
        img_float = img.astype(np.float32) / 255.0
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, intensity, img_float.shape)
            noisy = np.clip(img_float + noise, 0, 1)
        else:  # salt_pepper
            noisy = img_float.copy()
            num_pixels = img.shape[0] * img.shape[1]
            num_salt = int(num_pixels * intensity)
            coords = [np.random.randint(0, i, num_salt) for i in img.shape[:2]]
            noisy[coords[0], coords[1], :] = 1
            
            num_pepper = int(num_pixels * intensity)
            coords = [np.random.randint(0, i, num_pepper) for i in img.shape[:2]]
            noisy[coords[0], coords[1], :] = 0
        
        return (noisy * 255).astype(np.uint8)
    
    def apply_random_augmentation(self, img):
        """Apply random augmentation."""
        augmentations = [
            lambda x: self.rotate(x, random.choice([-30, -15, -5, 5, 15, 30])),
            lambda x: self.zoom(x, random.choice([0.8, 0.9, 1.1, 1.2])),
            lambda x: self.flip(x, random.choice(['h', 'v'])),
            lambda x: self.adjust_brightness(x, random.choice([-20, -10, 10, 20])),
            lambda x: self.adjust_contrast(x, random.choice([0.8, 0.9, 1.1, 1.2])),
            lambda x: self.add_gaussian_blur(x, random.choice([3, 5])),
            lambda x: self.add_noise(x, 'gaussian', intensity=0.005),
        ]
        
        augmented = img.copy()
        # Apply 1-2 random augmentations
        num_augmentations = random.choice([1, 1, 1, 2])
        for _ in range(num_augmentations):
            augmented = random.choice(augmentations)(augmented)
        
        return augmented


def augment_directory(src_dir, dst_dir, output_per_image=3):
    """Augment all images in directory."""
    augmenter = DataAugmenter()
    os.makedirs(dst_dir, exist_ok=True)
    
    images = list(Path(src_dir).glob('*'))
    logger.info(f"Found {len(images)} images in {src_dir}")
    
    augmented_count = 0
    for img_path in tqdm(images, desc=f"Augmenting {Path(src_dir).name}"):
        if not img_path.is_file():
            continue
        
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Save original
            output_path = Path(dst_dir) / f"{img_path.stem}_orig{img_path.suffix}"
            cv2.imwrite(str(output_path), img)
            augmented_count += 1
            
            # Create augmented versions
            for aug_idx in range(output_per_image):
                augmented = augmenter.apply_random_augmentation(img)
                output_path = Path(dst_dir) / f"{img_path.stem}_aug{aug_idx}{img_path.suffix}"
                cv2.imwrite(str(output_path), augmented)
                augmented_count += 1
        
        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")
    
    return augmented_count


def main():
    parser = argparse.ArgumentParser(description='Augment dataset')
    parser.add_argument('--output-size', type=int, default=20000, help='Target dataset size')
    parser.add_argument('--strategies', default='all', help='Augmentation strategies (all or specific)')
    args = parser.parse_args()
    
    logger.info(f"Starting data augmentation...")
    logger.info(f"Target size: {args.output_size}")
    
    # Create augmented directories
    clean_dir = config.DATASET_DIR / "CLEAN"
    stego_dir = config.DATASET_DIR / "STEGO"
    clean_aug_dir = config.DATASET_DIR / "CLEAN_AUGMENTED"
    stego_aug_dir = config.DATASET_DIR / "STEGO_AUGMENTED"
    
    # Count existing images
    clean_count = len(list(clean_dir.glob('*')))
    stego_count = len(list(stego_dir.glob('*')))
    total = clean_count + stego_count
    
    logger.info(f"Current dataset: {clean_count} CLEAN + {stego_count} STEGO = {total} total")
    
    # Calculate augmentation factor
    if total > 0:
        target_per_class = args.output_size // 2
        aug_factor = max(1, (target_per_class // max(clean_count, stego_count)))
    else:
        aug_factor = 3
    
    logger.info(f"Augmentation factor: {aug_factor}x per image")
    
    # Augment
    logger.info("Augmenting CLEAN images...")
    clean_aug_count = augment_directory(str(clean_dir), str(clean_aug_dir), output_per_image=aug_factor)
    
    logger.info("Augmenting STEGO images...")
    stego_aug_count = augment_directory(str(stego_dir), str(stego_aug_dir), output_per_image=aug_factor)
    
    total_aug = clean_aug_count + stego_aug_count
    logger.info(f"\n{'='*60}")
    logger.info(f"Augmentation complete!")
    logger.info(f"  CLEAN_AUGMENTED: {clean_aug_count} images")
    logger.info(f"  STEGO_AUGMENTED: {stego_aug_count} images")
    logger.info(f"  TOTAL: {total_aug} images ({total_aug/total:.1f}x original)")
    logger.info(f"{'='*60}")
    
    logger.info("\n✅ Next steps:")
    logger.info("1. Update DATASET paths in config.py to use _AUGMENTED directories")
    logger.info("2. Run: python train_deep_model_optimized.py")


if __name__ == '__main__':
    main()
