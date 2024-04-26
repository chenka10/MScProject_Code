import os
import numpy as np
from PIL import Image
from ROSMA.RosmaConfig import RosmaConfig
import pandas as pd


def get_extracted_frame(config, frame_index_orig, task_id, subject, repetition):
  task_name = config.get_task_name(task_id)
  frame_path = os.path.join(config.extracted_frames_dir,f'{subject}_{task_name}_0{repetition}',f'frame_{frame_index_orig}.png')
  frame = np.array(Image.open(frame_path))
  return frame

def get_kinematics(config: RosmaConfig, kinematic_row_orig, task_id, subject, repetition):
  task_name = config.get_task_name(task_id)
  
  kinematics_df = None
  if config.kinematics_df_storage.entry_exists(task_id, subject,repetition):
    kinematics_df = config.kinematics_df_storage.get_entry(task_id, subject, repetition)
  else:
    kinematics_file_path = os.path.join(config.kinematics_files_dir,f'{subject}_{task_name}_0{repetition}.csv')
    kinematics_df = pd.read_csv(kinematics_file_path) 
    config.kinematics_df_storage.add_entry(kinematics_df, task_id, subject, repetition)

  return kinematics_df.iloc[kinematic_row_orig].values


