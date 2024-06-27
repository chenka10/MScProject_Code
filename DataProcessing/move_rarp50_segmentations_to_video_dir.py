import os
import shutil
from glob import glob

# Define the path to the directory containing the video_x folders
base_path = '/home/chen/MScProject/data/rarp50_segmentations_128'

# Find all video_x directories within the base path
video_dirs = glob(os.path.join(base_path, 'video_*'))

for video_dir in video_dirs:
    seg_dir = os.path.join(video_dir, 'segmentation')
    
    # Check if the segmentations subfolder exists
    if os.path.isdir(seg_dir):
        # Move all PNG files from the segmentations subfolder to the parent video_x directory
        for png_file in glob(os.path.join(seg_dir, '*.png')):
            shutil.move(png_file, video_dir)

        # Remove the empty segmentations subfolder
        shutil.rmtree(seg_dir)