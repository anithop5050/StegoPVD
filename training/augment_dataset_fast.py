"""
Fast data augmentation for all dataset images.
Generates augmented versions in DATASET_AUGMENTED/
"""
import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm

def augment_image(img):
    """Apply random augmentation to an image."""
    # Randomly select augmentation type
    aug_type = random.choice(['rotate', 'flip', 'brightness', 'noise', 'none'])
    
    if aug_type == 'rotate':
        angle = random.choice([90, 180, 270])
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    elif aug_type == 'flip':
        flip_code = random.choice([0, 1, -1])  # vertical, horizontal, both
        img = cv2.flip(img, flip_code)
    
    elif aug_type == 'brightness':
        factor = random.uniform(0.7, 1.3)
        img = np.clip(img * factor, 0, 255).astype(np.uint8)
    
    elif aug_type == 'noise':
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


if __name__ == "__main__":
    print('╔════════════════════════════════════════════════════════════╗')
    print('║          DATA AUGMENTATION - FAST MODE                     ║')
    print('╚════════════════════════════════════════════════════════════╝')
    print()
    
    dataset_dir = Path('DATASET')
    aug_dir = Path('DATASET_AUGMENTED')
    
    # Define source directories
    clean_sources = [
        'CLEAN/INTERNET_PNG',
        'CLEAN/INTERNET_BMP', 
        'CLEAN/PNG',
        'CLEAN/BMP',
        'CLEAN/INTERNET'
    ]
    
    stego_sources = [
        'STEGO/INTERNET_PNG',
        'STEGO/INTERNET_BMP',
        'STEGO/PNG',
        'STEGO/BMP'
    ]
    
    # Count original images
    total_original = 0
    for source in clean_sources + stego_sources:
        source_path = dataset_dir / source
        if source_path.exists():
            count = len(list(source_path.glob('*.png'))) + len(list(source_path.glob('*.bmp')))
            total_original += count
    
    print(f'Found {total_original} original images')
    print(f'Will generate 2x augmented dataset ({total_original * 2} images)')
    print()
    
    # Augmentation settings
    augmentations_per_image = 2  # Generate 2 augmented versions per original
    
    total_generated = 0
    
    # Process CLEAN images
    print('Augmenting CLEAN images...')
    for source in clean_sources:
        source_path = dataset_dir / source
        if not source_path.exists():
            continue
        
        dest_path = aug_dir / source
        dest_path.mkdir(parents=True, exist_ok=True)
        
        images = list(source_path.glob('*.png')) + list(source_path.glob('*.bmp'))
        
        for img_path in tqdm(images, desc=f'  {source}'):
            # Copy original
            img = cv2.imread(str(img_path))
            if img is not None:
                # Save original
                ext = img_path.suffix
                cv2.imwrite(str(dest_path / f'{img_path.stem}_orig{ext}'), img)
                total_generated += 1
                
                # Generate augmented versions
                for i in range(augmentations_per_image):
                    aug_img = augment_image(img.copy())
                    cv2.imwrite(str(dest_path / f'{img_path.stem}_aug{i}{ext}'), aug_img)
                    total_generated += 1
    
    print()
    print('Augmenting STEGO images...')
    for source in stego_sources:
        source_path = dataset_dir / source
        if not source_path.exists():
            continue
        
        dest_path = aug_dir / source
        dest_path.mkdir(parents=True, exist_ok=True)
        
        images = list(source_path.glob('*.png')) + list(source_path.glob('*.bmp'))
        
        for img_path in tqdm(images, desc=f'  {source}'):
            # Copy original
            img = cv2.imread(str(img_path))
            if img is not None:
                # Save original
                ext = img_path.suffix
                cv2.imwrite(str(dest_path / f'{img_path.stem}_orig{ext}'), img)
                total_generated += 1
                
                # Generate augmented versions
                for i in range(augmentations_per_image):
                    aug_img = augment_image(img.copy())
                    cv2.imwrite(str(dest_path / f'{img_path.stem}_aug{i}{ext}'), aug_img)
                    total_generated += 1
    
    print()
    print('═' * 70)
    print('✅ AUGMENTATION COMPLETE!')
    print('═' * 70)
    print(f'  Original images: {total_original}')
    print(f'  Generated images: {total_generated}')
    print(f'  Multiplier: {total_generated/total_original:.1f}x')
    print()
    print('Next: Update train_deep_model.py to use DATASET_AUGMENTED/')
