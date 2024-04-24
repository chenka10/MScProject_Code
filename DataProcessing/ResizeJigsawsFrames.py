from PIL import Image
import os
from tqdm import tqdm

def resize_images_in_folder(input_folder, output_folder, target_size=(128, 128)):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each subfolder in the input folder
    for root, dirs, files in os.walk(input_folder):
        # Get the relative path of the subfolder within the input folder
        rel_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, rel_path)

        # Create the corresponding subfolder in the output folder
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        # Process each file in the subfolder
        for file in tqdm(files):
            # Check if the file is a PNG image
            if file.lower().endswith('.png'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_subfolder, file)

                # Open the image
                img = Image.open(input_path)

                # Resize the image
                img_resized = img.resize(target_size, resample=Image.LANCZOS)

                # Save the resized image
                img_resized.save(output_path)                

# Example usage
input_folder = '/home/chen/MScProject/data/jigsaws_extracted_frames'
output_folder = '/home/chen/MScProject/data/jigsaws_extracted_frames_128'

resize_images_in_folder(input_folder, output_folder)
