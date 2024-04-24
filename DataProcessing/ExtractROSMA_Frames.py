import cv2
import os
import glob
from tqdm import tqdm

# Function to extract frames from a video file and save them as PNG images
def extract_frames(video_file, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Extract and save each frame
    for frame_num in tqdm(range(total_frames)):
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        min_dim = min(h, w)
        center_x = w // 2
        center_y = (h // 2) + 150
        half_size = min_dim // 2
        crop_img = frame[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]
        
        blurred_img = cv2.GaussianBlur(crop_img, (11, 11), 0)

        # Resize the frame to 64x64 pixels
        resized_frame = cv2.resize(blurred_img, (128, 128),interpolation=cv2.INTER_AREA)
        
        # Save the frame as a PNG image
        frame_filename = os.path.join(output_folder, f"frame_{frame_num}.png")
        cv2.imwrite(frame_filename, resized_frame)

    # Release the video capture object
    cap.release()

# Function to process all MP4 files in a folder
def process_mp4_files(folder, output_folder):
    # Get a list of all MP4 files in the folder
    mp4_files = glob.glob(os.path.join(folder, "*Pea*.mp4"))

    # Process each MP4 file
    for mp4_file in mp4_files:
        # Extract frames from the MP4 file and save them as PNG images
        video_name = os.path.splitext(os.path.basename(mp4_file))[0]
        output_folder_vid = os.path.join(output_folder, video_name)
        extract_frames(mp4_file, output_folder_vid)

# Main function
def main():
    # Input folder containing MP4 files
    input_folder = "/home/chen/MScProject/data/ROSMA"
    output_folder = "/home/chen/MScProject/data/ROSMA_frames_128"

    # Process MP4 files in the input folder
    process_mp4_files(input_folder, output_folder)

if __name__ == "__main__":
    main()
