import torch
import tifffile
import numpy as np
from tqdm import tqdm
import os
import cv2
from main_MCAN import Net, input_matrix_wpn

# Configuration
checkpoint_path = "checkpoint/De_happy_model_epoch_8500.pth"
input_dir = "/home/llawson/MCAN/Mosaic-Convolution-Attention-Network-for-Demosaicing-Multispectral-Filter-Array-Images-main/CAVE_dataset/real_images"
output_dir = "visual_results/"
os.makedirs(output_dir, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = Net().to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model"].state_dict())
model.eval()

def load_tiff(path):
    """Load TIFF file with proper axis handling"""
    try:
        img = tifffile.imread(path)
        
        # Handle different TIFF formats
        if img.ndim == 2:  # Grayscale
            img = np.expand_dims(img, axis=0)  # Add channel dimension
        elif img.ndim == 3:  # Multi-channel
            if img.shape[0] < img.shape[-1]:  # CHW format
                img = np.transpose(img, (2, 0, 1))  # Convert HWC to CHW
        return img.astype(np.float32) / 65535.0  # Normalize 16-bit to [0,1]
    except Exception as e:
        print(f"Error loading {os.path.basename(path)}: {str(e)}")
        return None

def process_image(img_path, output_dir):
    try:
        # Load and preprocess
        img = load_tiff(img_path)
        if img is None:
            return
            
        # Convert to tensor
        input_tensor = torch.from_numpy(img).unsqueeze(0).to(device)  # Add batch dim
        
        # Generate coordinate map
        _, _, H, W = input_tensor.shape
        scale_coord_map = input_matrix_wpn(H, W).to(device)
        
        # Inference
        with torch.no_grad():
            output = model([input_tensor, input_tensor], scale_coord_map)
        
        # Save output
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        output_np = output.squeeze().cpu().numpy()
        
        # Save as 16-bit TIFF
        output_np = (output_np * 65535).astype(np.uint16)
        tifffile.imwrite(output_path, output_np)
        
    except Exception as e:
        print(f"Error processing {os.path.basename(img_path)}: {str(e)}")

# Process all TIFFs
tiff_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
for filename in tqdm(tiff_files):
    process_image(os.path.join(input_dir, filename), output_dir)

print(f"\nProcessing complete. Check {output_dir} for results")
