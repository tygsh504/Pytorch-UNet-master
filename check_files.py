import os
from pathlib import Path

# 1. SETUP: Use Path(r"...") to ensure Windows paths work correctly
dir_img = Path(r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Combined\Training_Ori")
dir_mask = Path(r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Combined\Training_GT")

print(f"Scanning images in: {dir_img}")
print(f"Scanning masks in:  {dir_mask}")

# 2. Get list of all images
img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

try:
    images = [f for f in dir_img.iterdir() if f.suffix.lower() in img_extensions]
except FileNotFoundError:
    print(f"\n❌ ERROR: The folder '{dir_img}' does not exist.")
    print("Please check the path spelling.")
    exit()

print(f"Found {len(images)} images. Checking for matching masks...\n")

missing_masks = []
mismatched_names = []

for img_path in images:
    img_id = img_path.stem  # Filename without extension
    
    # 3. Check logic: Look for mask with SAME name
    found = list(dir_mask.glob(f"{img_id}.*"))
    
    # 4. Check secondary logic: Look for '_mask' suffix
    found_suffix = list(dir_mask.glob(f"{img_id}_mask.*"))
    
    if not found and not found_suffix:
        missing_masks.append(img_path.name)
    elif not found and found_suffix:
        mismatched_names.append(img_path.name)

# 5. Report results
if missing_masks:
    print(f"❌ CRITICAL ERROR: Found {len(missing_masks)} images with NO matching mask:")
    for name in missing_masks[:10]:
        print(f"   - {name}")
    if len(missing_masks) > 10: print(f"   ... and {len(missing_masks)-10} more.")
    print("\nSOLUTION: You must create masks for these images or delete the images.")

elif mismatched_names:
    print(f"⚠️ WARNING: Found {len(mismatched_names)} images where masks have '_mask' suffix.")
    print("Example: Image 'abc.jpg' has mask 'abc_mask.png'.")
    print("The default loader expects 'abc.png'. You may need to rename your masks.")

else:
    print("✅ SUCCESS: All images appear to have matching masks.")