import os
import random
import math
from PIL import Image, ImageDraw

# NOTE: This script requires the Pillow library.
# If you run into an error, install it using:
# pip install Pillow

def add_noise(img, intensity=10):
    """Adds subtle random noise/grain to the image pixels."""
    pixels = img.load()
    width, height = img.size
    
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            
            # Generate small random shift for each color channel
            r_shift = random.randint(-intensity, intensity)
            g_shift = random.randint(-intensity, intensity)
            b_shift = random.randint(-intensity, intensity)
            
            # Apply shift and clamp values between 0 and 255
            r = max(0, min(255, r + r_shift))
            g = max(0, min(255, g + g_shift))
            b = max(0, min(255, b + b_shift))
            
            pixels[x, y] = (r, g, b)
    return img


def generate_png_images():
    """
    Prompts the user for a number and generates that many colorful PNG images
    with smooth radial gradients and a subtle noise texture, similar to the example image.
    Each image will be 200x200.
    """
    print("--- Abstract PNG Generator (Smooth Gradient & Noise) ---")

    # 1. Get user input and handle errors
    while True:
        try:
            num_images = int(input("How many abstract gradient images do you need to generate? "))
            if num_images <= 0:
                print("Please enter a positive number greater than zero.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    print(f"\nGenerating {num_images} smooth, colorful images...")
    
    # Define image properties
    width = 200
    height = 200
    center_x = width // 2
    center_y = height // 2
    max_distance = math.sqrt(center_x**2 + center_y**2) # Max distance from center to corner
    
    # 2. Loop and generate images
    for i in range(1, num_images + 1):
        # Generate two random vibrant colors for the radial gradient
        # Using a wide range for diverse, vibrant colors
        color_center = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color_edge = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # Create a new blank image and load pixel data
        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        pixels = img.load()

        # Apply a radial gradient for smooth color blending
        for x in range(width):
            for y in range(height):
                # Calculate distance from the center
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                # Normalize distance to a factor between 0.0 (center) and 1.0 (edge)
                # We use factor to control the blend
                factor = min(1.0, distance / max_distance) 
                
                # Interpolate colors: color_center * (1 - factor) + color_edge * factor
                r = int(color_center[0] * (1 - factor) + color_edge[0] * factor)
                g = int(color_center[1] * (1 - factor) + color_edge[1] * factor)
                b = int(color_center[2] * (1 - factor) + color_edge[2] * factor)
                
                pixels[x, y] = (r, g, b)
        
        # 3. Apply the noise/grain texture
        img = add_noise(img, intensity=10)

        # 4. Save the image
        filename = f"abstract_noise_gradient_{i}.bmp"
        try:
            img.save(filename, "bmp")
            print(f"  > Successfully created: {filename}")
        except Exception as e:
            print(f"  > Error saving {filename}: {e}")

    print("\nGeneration complete!")

if __name__ == "__main__":
    generate_png_images()
