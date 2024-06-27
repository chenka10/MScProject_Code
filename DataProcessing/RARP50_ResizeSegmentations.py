import os
from PIL import Image, ImageFilter
from tqdm import tqdm
import numpy as np

# in rarp50 segmentations:
# 0 - background
# 1 - claw "fingers"
# 2 - claw "base"
# 3 - arm
# 4 - needle
# 5 - string

# Source directory
src_dir = "/home/chen/MScProject/data/rarp50_segmentations_orig"

FRAME_SIZE = 64

# Destination directory
dest_dir = f"/home/chen/MScProject/data/rarp50_segmentations_{FRAME_SIZE}"

# Function to resize images
def resize_image(input_path, output_path):
    with Image.open(input_path) as img:
        # Resize image
        img_resized = img.resize((FRAME_SIZE, FRAME_SIZE), Image.NEAREST) 

        img_resized.save(output_path)   


# Traverse the directory structure
for root, dirs, files in os.walk(src_dir):
    for dir_name in dirs:
        src_subdir = os.path.join(root, dir_name, 'segmentation')
        # src_subdir = os.path.join(root, dir_name)
        dest_subdir = os.path.join(dest_dir, os.path.relpath(src_subdir, src_dir))

        # Create corresponding directory structure in destination folder
        os.makedirs(dest_subdir, exist_ok=True)

        print('doing ',src_subdir)
        print('saving in ',dest_subdir)

        # Loop through image files in source directory
        for file_name in tqdm(os.listdir(src_subdir)):
            if file_name.endswith(".png"):
                src_image_path = os.path.join(src_subdir, file_name)
                dest_image_path = os.path.join(dest_subdir, file_name)

                # Resize image and save to destination folder
                resize_image(src_image_path, dest_image_path)
