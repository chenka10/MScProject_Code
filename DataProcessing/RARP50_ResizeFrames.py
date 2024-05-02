import os
from PIL import Image, ImageFilter
from tqdm import tqdm

# Source directory
src_dir = "/media/HDD1/chen/rarp50/frames_from_selected_videos"

# Destination directory
dest_dir = "/home/chen/MScProject/data/RARP50_selected_64"

# Function to resize images
def resize_image(input_path, output_path):
    with Image.open(input_path) as img:
        # Apply Gaussian blur
        # img = img.filter(ImageFilter.GaussianBlur(radius=2))

        # Resize image
        img_resized = img.resize((128, 128), Image.LANCZOS)        
        img_resized.save(output_path)       


# Traverse the directory structure
for root, dirs, files in os.walk(src_dir):
    for dir_name in dirs:
        src_subdir = os.path.join(root, dir_name)
        dest_subdir = os.path.join(dest_dir, os.path.relpath(src_subdir, src_dir))

        # Create corresponding directory structure in destination folder
        os.makedirs(dest_subdir, exist_ok=True)

        # Loop through image files in source directory
        for file_name in tqdm(os.listdir(src_subdir)):
            if file_name.endswith(".jpg"):
                src_image_path = os.path.join(src_subdir, file_name)
                dest_image_path = os.path.join(dest_subdir, file_name)

                # Resize image and save to destination folder
                resize_image(src_image_path, dest_image_path)
