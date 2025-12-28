import os
from PIL import Image
from pathlib import Path

# Update path to your images
dir_img = Path(r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Combined\Training_GT")

print(f"Scanning {dir_img}...")

sizes = {}
files_by_size = {}

for f in dir_img.iterdir():
    if f.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']:
        img = Image.open(f)
        w, h = img.size
        size_str = f"{w}x{h}"
        
        # Count how many images have this size
        sizes[size_str] = sizes.get(size_str, 0) + 1
        
        # Store first filename of this size for reference
        if size_str not in files_by_size:
            files_by_size[size_str] = f.name

print("\n--- Image Dimension Report ---")
for size, count in sizes.items():
    print(f"Size {size}: {count} images (Example: {files_by_size[size]})")

if len(sizes) > 1:
    print("\n❌ PROBLEM FOUND: Your dataset has mixed sizes.")
    print("PyTorch U-Net requires ALL input images to be the exact same size during batching.")
else:
    print("\n✅ All images are the same size.")