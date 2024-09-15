import os
from PIL import Image
import re
from tqdm import tqdm

def natural_sort_key(s):
    """Sort helper function that extracts numbers from a string for natural sorting."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def aggregate_frames(input_dir, output_dir, window_size=50, stride=10):
    """
    Aggregate video frames into horizontal strips using a sliding window approach.
    
    Args:
    - input_dir (str): The directory containing subdirectories of video frames.
    - output_dir (str): The directory where output horizontal strips will be saved.
    - window_size (int): Number of frames per strip.
    - stride (int): The step size for the sliding window.
    """
    
    # Iterate through each subdirectory (each video frames directory)
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        
        if os.path.isdir(subdir_path):
            print(f'Processing directory: {subdir_path}')
            
            # Create the same subdirectory structure in the output directory
            subdir_output_path = os.path.join(output_dir, subdir)
            os.makedirs(subdir_output_path, exist_ok=True)
            
            # List all frame files and sort them by the numeric part of their filename
            frame_files = [f for f in os.listdir(subdir_path) if f.endswith('.png') or f.endswith('.jpg')]
            frame_files.sort(key=natural_sort_key)
            
            num_frames = len(frame_files)
            if num_frames < window_size:
                print(f'Skipping directory {subdir}, not enough frames.')
                continue  # Skip if not enough frames to form a strip
            
            # Progress bar for the sliding window processing
            for start_idx in tqdm(range(0, num_frames - window_size + 1, stride), desc=f"Processing {subdir}"):
                # Load the frames in the current window
                window_frames = []
                for i in range(start_idx, start_idx + window_size):
                    frame_path = os.path.join(subdir_path, frame_files[i])
                    frame = Image.open(frame_path)
                    window_frames.append(frame)
                
                # Concatenate the frames horizontally
                widths, heights = zip(*(frame.size for frame in window_frames))
                total_width = sum(widths)
                max_height = max(heights)
                
                # Create an empty image to paste frames into
                strip_image = Image.new('RGB', (total_width, max_height))
                
                # Paste each frame next to each other
                x_offset = 0
                for frame in window_frames:
                    strip_image.paste(frame, (x_offset, 0))
                    x_offset += frame.width
                
                # Save the result in the corresponding subdirectory
                output_path = os.path.join(subdir_output_path, f'{subdir}_strip_{start_idx}.png')
                strip_image.save(output_path)
            
            print(f'Finished processing directory: {subdir_path}')

# Example usage:
input_dir = '/home/chen/MScProject/data/jigsaws_extracted_frames_64'
output_dir = '/home/chen/MScProject/data_mocoGAN/jigsaws_extracted_frames_64'
aggregate_frames(input_dir, output_dir, window_size=50, stride=10)
