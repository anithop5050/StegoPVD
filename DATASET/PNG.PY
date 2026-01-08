import math
import random
import string
import os
from PIL import Image

# --- PVD Range Table (Must match JS version exactly) ---
# [Lower_Bound, Upper_Bound, Bits_Per_Pair (t)]
PVD_RANGES = [
    (0, 7, 3), 
    (8, 15, 3), 
    (16, 31, 4), 
    (32, 63, 5), 
    (64, 127, 6), 
    (128, 255, 7) 
]
HEADER_SIZE_BITS = 32

def get_pvd_range(diff):
    """Finds the capacity (t) and range (L, U) for a given pixel difference."""
    for L, U, t in PVD_RANGES:
        if L <= diff <= U:
            return L, U, t
    return None, None, None

def generate_random_message(length_bytes=100): # Increased default length for better testing
    """Generates a random string of specified length for embedding."""
    chars = string.ascii_letters + string.digits + ' '
    return ''.join(random.choice(chars) for _ in range(length_bytes))

def message_to_bits(message):
    """Converts a message string into a binary string including the 32-bit length header."""
    message_bytes = message.encode('utf-8')
    message_len = len(message_bytes)
    
    # 1. Build 32-bit header (Big-Endian)
    len_bits = bin(message_len)[2:].zfill(HEADER_SIZE_BITS)
    
    # 2. Build message bits
    msg_bits = ''.join(bin(b)[2:].zfill(8) for b in message_bytes)
    
    return len_bits + msg_bits, message_len

def embed_message(input_image_path, output_image_path, message, filename):
    """
    Embeds the message into the image using the defined PVD rules on the Red channel.
    """
    print(f"\n--- Processing: {filename} ---")
    try:
        # Use full path for PIL to find the file
        img = Image.open(input_image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_image_path}. Skipping.")
        return False
    except Exception as e:
        print(f"Error opening image {filename}: {e}. Skipping.")
        return False

    # Get list of pixel tuples: [(R1, G1, B1), (R2, G2, B2), ...]
    pixels = list(img.getdata())
    
    bits, message_len = message_to_bits(message)
    total_message_bits = len(bits)
    
    # Check capacity (simple estimate)
    max_capacity = 0
    # Iterate over pixel pairs (i and i+1)
    for i in range(0, len(pixels), 2): 
        if i + 1 >= len(pixels): break
        
        # Access Red channel (index 0) of the tuples
        diff = abs(pixels[i][0] - pixels[i+1][0])
        _, _, t = get_pvd_range(diff)
        if t:
            max_capacity += t

    if total_message_bits > max_capacity:
        print(f"Error: Message is too large ({total_message_bits} bits) for image capacity ({max_capacity} bits). Skipping.")
        return False

    new_pixels = []
    bit_index = 0
    
    for i in range(0, len(pixels), 2): # Iterate over Pixel 1 and Pixel 2
        if bit_index >= total_message_bits:
            new_pixels.extend(pixels[i:])
            break

        if i + 1 >= len(pixels): # Handle stray last pixel
            new_pixels.append(pixels[i])
            break

        # Get R, G, B values for P1 and P2
        p1_r, p1_g, p1_b = pixels[i]
        p2_r, p2_g, p2_b = pixels[i+1]
        
        L, U, t = get_pvd_range(abs(p2_r - p1_r))

        if L is not None:
            
            # --- PVD Encoding Logic (Compatibility Match) ---
            
            # 1. Get bits to hide (t bits)
            remaining_message_bits = total_message_bits - bit_index
            actual_bits_to_take = min(t, remaining_message_bits)
            
            bits_to_hide = bits[bit_index : bit_index + actual_bits_to_take]
            
            # CRITICAL FIX: Pad the message bits to the full capacity 't' 
            padded_bits = bits_to_hide.ljust(t, '0')
            
            b = int(padded_bits, 2)
            bit_index += actual_bits_to_take # ONLY advance by actual message bits taken
            
            # 2. Calculate new difference d'
            d_prime = L + b
            
            # 3. Calculate change amount (delta)
            delta = d_prime - abs(p2_r - p1_r)
            
            # 4. Adjust pixels p1' and p2'
            sign = 1 if p2_r >= p1_r else -1
            
            if sign == 1: # p2 >= p1
                p1_r_prime = p1_r - math.floor(delta / 2)
                p2_r_prime = p2_r + math.ceil(delta / 2)
            else: # p1 > p2
                p1_r_prime = p1_r + math.ceil(delta / 2)
                p2_r_prime = p2_r - math.floor(delta / 2)
                
            # 5. Apply clipping (0-255)
            p1_r_prime = max(0, min(255, p1_r_prime))
            p2_r_prime = max(0, min(255, p2_r_prime))

            # Update the R values in the new pixel list, keep G and B untouched
            new_pixels.append((p1_r_prime, p1_g, p1_b))
            new_pixels.append((p2_r_prime, p2_g, p2_b))

        else:
            # Cannot use this pair, append original pixels
            new_pixels.append(pixels[i])
            new_pixels.append(pixels[i+1])
    
    # Ensure all pixels are accounted for if loop was broken early due to bit_index
    if len(new_pixels) < len(pixels):
        new_pixels.extend(pixels[len(new_pixels):])

    # Save the new image data
    stego_img = Image.new(img.mode, img.size)
    stego_img.putdata(new_pixels) # Pillow expects list of tuples here
    
    # Save as PNG to ensure LSB/PVD changes are preserved (no lossy compression)
    stego_img.save(output_image_path, "PNG")
    
    print(f"SUCCESS: Message ({message_len} bytes) embedded and saved to {output_image_path}")
    # Save the message for later decoding reference (optional)
    message_log_path = os.path.join(os.path.dirname(output_image_path), "stego_messages.txt")
    with open(message_log_path, 'a') as f:
        f.write(f"{filename},{message_len},\"{message}\"\n")
    return True


if __name__ == "__main__":
    # --- Configuration ---
    
    # Folder containing all your clean images
    IMAGE_DIRECTORY = r"C:\Users\Anith\OneDrive\Desktop\New folder\CLEAN\PNG"
    
    # Length of the random message to generate for EACH image (in bytes)
    MESSAGE_LENGTH = 100 
    
    # Output Subfolder: Stego images will be saved here (e.g., in a new folder called 'Stego_Output')
    OUTPUT_SUBFOLDER_NAME = "Stego_Output"
    
    # --- Setup ---
    
    # Create output directory path
    OUTPUT_DIRECTORY = os.path.join(IMAGE_DIRECTORY, OUTPUT_SUBFOLDER_NAME)
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # Define acceptable image extensions
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')
    
    # Get all image files in the directory
    all_files = os.listdir(IMAGE_DIRECTORY)
    image_files = [f for f in all_files if f.lower().endswith(IMAGE_EXTENSIONS)]
    
    print(f"--- Starting Batch PVD Embedding ---")
    print(f"Source Folder: {IMAGE_DIRECTORY}")
    print(f"Found {len(image_files)} images to process.")
    
    if not image_files:
        print("No image files found. Please check your folder path and image extensions.")
    else:
        # Clear the message log file before starting a new batch
        message_log_path = os.path.join(OUTPUT_DIRECTORY, "stego_messages.txt")
        if os.path.exists(message_log_path):
            os.remove(message_log_path)
            
        # --- Batch Execution ---
        for filename in image_files:
            input_path = os.path.join(IMAGE_DIRECTORY, filename)
            
            # Create a unique output name: originalName_stego.png
            name_parts = os.path.splitext(filename)
            output_filename = f"{name_parts[0]}_stego.png"
            output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)

            # 1. Generate a unique random message for this image
            secret_message = generate_random_message(MESSAGE_LENGTH)

            # 2. Embed the message
            embed_message(
                input_image_path=input_path, 
                output_image_path=output_path, 
                message=secret_message, 
                filename=filename
            )
            
        print(f"\n✅ Batch Processing Complete. All stego images saved in: {OUTPUT_DIRECTORY}")
        print("Note: All original messages are logged in 'stego_messages.txt' in the output directory for reference.")
