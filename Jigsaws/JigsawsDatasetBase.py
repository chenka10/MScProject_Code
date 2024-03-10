from torch.utils.data import Dataset
import numpy as np
import JigsawsUtils as ju
import bisect

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class JigsawsDatasetBase(Dataset):
  def __init__(self,data_csv_df,config,frames_to_retrieve,sample_rate):

        self.frames_to_retrieve = frames_to_retrieve
        self.config = config
        self.df = data_csv_df
        self.sample_rate = sample_rate
        self.num_frames_per_video = (self.df['Frame Count (kinematics)']/self.sample_rate).to_numpy().astype(int) - self.frames_to_retrieve + 1     
        self.cumulative_frames = [0] + list(np.cumsum(self.num_frames_per_video))
        

  def __len__(self):
      return sum(self.num_frames_per_video)

  def get_reward(self,frame_index_resolution,task_id,subject,repetition):
      gestures_for_full_video = ju.jigsaws_get_gestures(self.config, task_id, subject, repetition)
      gestures_by_time_resolution = gestures_for_full_video[np.array(range(0,gestures_for_full_video.shape[0],self.sample_rate))]

      if frame_index_resolution > gestures_by_time_resolution.shape[0]:
         return 0

      rewards_by_time_resolution = ju.jigsaws_get_gesture_based_rewards(self.config, gestures_by_time_resolution)
      return rewards_by_time_resolution[frame_index_resolution]


  def get_kinematics(self,frame_index_orig,task_id,subject,repetition):    
    return ju.jigsaws_get_kinematics(self.config ,task_id,subject,repetition).iloc[frame_index_orig,:].values

  def get_frame_indexes_and_jigsaws_meta_by_item_index(self,index):
      # Find the corresponding video
      video_index = bisect.bisect_right(self.cumulative_frames, index) - 1
      video_start_frame = self.cumulative_frames[video_index]

      # Find frame in the video resnet features (orig frame indexes divided by the sample resolution)
      frame_index_resolution = index - video_start_frame

      # Frame index in the original video
      frame_index_orig = int((frame_index_resolution) * self.sample_rate)

      task_id = self.df['Task ID'][video_index]
      subject = self.df['Subject'][video_index]
      repetition = self.df['Repetition'][video_index]

      return frame_index_resolution, frame_index_orig, task_id, subject, repetition

  def __getitem__(self, index):      
      pass

class JigsawsMetaDataset(JigsawsDatasetBase):
  def __init__(self, data_csv_df, config, frames_to_retrieve,sample_rate=1):
      super().__init__(data_csv_df, config, frames_to_retrieve,sample_rate)        

  def __len__(self):
    return super().__len__()

  def __getitem__(self, index):        
    frame_index_resolution, frame_index_orig, task_id, subject, repetition = super().get_frame_indexes_and_jigsaws_meta_by_item_index(index)
    return frame_index_resolution, frame_index_orig, task_id, subject, repetition