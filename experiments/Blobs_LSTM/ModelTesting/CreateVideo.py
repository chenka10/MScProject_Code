import imageio
import os

# Directory containing the PNG images
images_dir = "/home/chen/MScProject/Code/experiments/rarp50/Blobs_LSTM/ModelTesting/2_blobs_framesize_128_leave_37_models_20240629_174022_i1mrwoir_new1/"

# List to store image paths
image_paths = []

# Iterate over the files in the directory
i = 20
while True:
    filename = f'test_{i}.png'
    filepath = os.path.join(images_dir, filename)

    if os.path.exists(filepath) is False:
        break

    image_paths.append(filepath)
    i+=1

# List to store images
images = []

# Read each image and append it to the images list
for image_path in image_paths:
    images.append(imageio.imread(image_path))

# Path to save the video
video_path = os.path.join(images_dir,"000_video.mp4")

# Save images as a video with a frame rate of 5 Hz
imageio.mimsave(video_path, images, fps=5)

print(f"Video saved to: {video_path}")