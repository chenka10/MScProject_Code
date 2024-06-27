import os
import csv

def get_frame_info(directory):
    """ Get start frame, end frame, and number of frames in a given directory. """
    frames = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            try:
                # Extract the frame number assuming the filename is a number followed by .jpg
                frame_number = int(filename.split('.')[0])
                frames.append(frame_number)
            except ValueError:
                continue
    
    if not frames:
        return None, None, 0
    
    start_frame = min(frames)
    end_frame = max(frames)
    number_of_frames = len(frames)
    
    return start_frame, end_frame, number_of_frames

def create_csv(directory, output_file):
    """ Create a CSV file with video information. """
    with open(output_file, mode='w', newline='') as csv_file:
        fieldnames = ['videoName', 'startFrame', 'endFrame', 'numberOfFrames']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()

        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir,'segmentation')
            if os.path.isdir(subdir_path) and subdir.startswith("video"):
                start_frame, end_frame, number_of_frames = get_frame_info(subdir_path)
                if number_of_frames > 0:
                    writer.writerow({
                        'videoName': subdir,
                        'startFrame': start_frame,
                        'endFrame': end_frame,
                        'numberOfFrames': number_of_frames
                    })

# Usage
main_directory = '/home/chen/MScProject/data/rarp50_segmentations_64'
output_csv = 'rarp50_segmentations_data_detailed.csv'
create_csv(main_directory, output_csv)
