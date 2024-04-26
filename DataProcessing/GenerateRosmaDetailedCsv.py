import sys
sys.path.append('/home/chen/MScProject/Code/')

import os
import pandas as pd
from moviepy.editor import VideoFileClip
from ROSMA.RosmaConfig import config
from tqdm import tqdm


def get_usable_frame_count(sync_init_frame, sync_init_row, video_frames_count, rows_count, target_frame_rate=5):
   video_sample_rate = 15 # [Hz]
   kinematic_sample_Rate = 50 # [Hz]
   usable_video_frames = (video_frames_count - sync_init_frame)//(video_sample_rate/target_frame_rate) + 1
   usable_kinematic_frames = (rows_count - sync_init_row)//(kinematic_sample_Rate/target_frame_rate) + 1

   return (usable_video_frames, usable_kinematic_frames)

# Function to count lines in CSV file
def count_csv_lines(csv_file):
    with open(csv_file, 'r') as file:
        return sum(1 for line in file)

# Function to get video information
def get_video_info(file_path):
    clip = VideoFileClip(file_path)
    return (clip.reader.nframes, clip.fps)
    

# Directory containing the mp4 files
folder_path = "/home/chen/MScProject/data/ROSMA"

# Initialize list to store data
data = []

synchronization_dataframe = pd.read_csv("/home/chen/MScProject/data/ROSMA/synchronizationData.csv", delimiter=' ',header=None,names=['name','start_frame','start_row','idk'])

# Iterate through files in the folder
for file_name in tqdm(os.listdir(folder_path)):
    if file_name.endswith(".mp4"):
        # Extract information from file name
        parts = file_name[:-4].split("_")
        subject_id = parts[0]
        task_name = '_'.join(parts[1:-1])
        repetition_number = parts[-1]

        # Get full file path
        file_path = os.path.join(folder_path, file_name)
        csv_file_path = os.path.join(folder_path, file_name[:-4] + ".csv")

        # Get video information
        (video_frame_count, video_fps) = get_video_info(file_path)

        sync_info = synchronization_dataframe[synchronization_dataframe['name']=='_'.join(parts)]
        sync_init_frame = sync_info['start_frame'].values[0]
        sync_init_row = sync_info['start_row'].values[0]

        kinematic_rows_num = count_csv_lines(csv_file_path)-2

        (usable_video_frames, usable_kinematic_frames) = get_usable_frame_count(sync_init_frame, sync_init_row, video_frame_count, kinematic_rows_num, 5)


        # Append data to list
        data.append({
            "TaskName": task_name,
            "TaskID": config.get_task_id(task_name),
            "SubjectID": subject_id,
            "RepetitionNumber": repetition_number,
            "FrameCount":video_frame_count,
            "VideoSampleRate[Hz]":video_fps,            
            "Kinematics Count": kinematic_rows_num,
            "KinematicsSampleRate[Hz]":50,
            "sync_start_frame": sync_init_frame,
            "sync_start_row": sync_init_row,
            "video_usable_sync_frames[5Hz]": usable_video_frames,
            "kinematics_usable_sync_frames[5Hz]": usable_kinematic_frames
        })

# Create DataFrame
df = pd.DataFrame(data)

# Display DataFrame
df.to_csv('/home/chen/MScProject/rosma_all_data_detailed.csv',index=False)