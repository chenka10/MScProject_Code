from JigsawsConfig import JigsawsConfig
from torchvision.io import read_image
import os
import numpy as np
from utils import utils_find_series_in_arr
import pandas as pd
from PIL import Image

def jigsaws_get_extracted_frame(config: JigsawsConfig, frame_index_orig, task_id, subject, repetition):
  task_name = config.get_task_name(task_id)
  frame_path = os.path.join(config.extracted_frames_dir,f'{task_name}_{subject}00{repetition}_capture1',f'frame_{frame_index_orig}.png')

  frame = np.array(Image.open(frame_path))

  return frame

def jigsaws_get_kinematics(config: JigsawsConfig ,task_id,subject,repetition):
  if (config.kinematics_df_storage.entry_exists(task_id,subject,repetition)):
    return config.kinematics_df_storage.get_entry(task_id,subject,repetition)

  task_name = config.get_task_name(task_id)
  kinematics_file_path = os.path.join(config.get_metadata_dir(),task_name,'kinematics','AllGestures','{}_{}00{}.txt'.format(task_name,subject,repetition))
  df = pd.read_csv(kinematics_file_path,delimiter='\s+',header=None)
  config.kinematics_df_storage.add_entry(df,task_id,subject,repetition)

  return df

def jigsaws_get_kinematics_means_stds(config: JigsawsConfig):
  kinematics_stats_csv_path = os.path.join(config.get_project_dir(),'kinematics_stats.csv')
  kinematics_stats = np.loadtxt(kinematics_stats_csv_path, delimiter=',')

  means = kinematics_stats[0,:]
  stds = kinematics_stats[1,:]

  return means, stds

def jigsaws_normalize_kinematic_vecs(config: JigsawsConfig,vecs_to_normalize):
  means, stds = jigsaws_get_kinematics_means_stds(config)
  normalized = (vecs_to_normalize - means)/stds
  return normalized

def jigsaws_unnormalize_kinematic_vecs(config: JigsawsConfig,vecs_to_unnormalize):
  means, stds = jigsaws_get_kinematics_means_stds(config)
  unnormalized = vecs_to_unnormalize*stds + means
  return unnormalized


def jigsaws_filter_kinematic_vecs(config: JigsawsConfig, vecs_to_filter, kin_unit_name,kin_var_name):
  kin_slice = config.get_kinematic_slices(kin_unit_name,kin_var_name)
  return vecs_to_filter[:,kin_slice]

def jigsaws_get_gestures(config: JigsawsConfig, task_id,subject,repetition):
  if config.gestures_storage.entry_exists(task_id,subject,repetition):
      return config.gestures_storage.get_entry(task_id,subject,repetition)
  else:
    gestures_file_path =\
    os.path.join(config.get_metadata_dir(),'Suturing','transcriptions','{}_{}00{}.txt'\
                  .format(config.get_task_name(task_id),subject,repetition))

    df = pd.read_csv(gestures_file_path,header=None,sep=' ')

    last_frame = df.iloc[-1,1]

    frames = np.arange(last_frame+500) # the +500 thing is a buffer since sometimes kinematics anotations end before video

    gestures_frames = np.array(df.iloc[:,1].values)

    gestures = np.array([int(gesture_name.split('G')[1]) for gesture_name in np.array(df.iloc[:,2].values)])   
    gestures = np.append(gestures,gestures[-1]) # I add the last gestures so that frames after it will also be classified as it
 
    gestures_per_frame = gestures[np.searchsorted(gestures_frames,frames+1)]
    config.gestures_storage.add_entry(gestures_per_frame,task_id,subject,repetition)
    return gestures_per_frame