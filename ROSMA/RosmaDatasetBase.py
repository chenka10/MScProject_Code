from torch.utils.data import Dataset
import numpy as np
import bisect

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
   

def get_usable_frame_count(sync_init_frame, sync_init_row, video_frames_count, rows_count, target_frame_rate=5):
   video_sample_rate = 15 # [Hz]
   kinematic_sample_Rate = 50 # [Hz]
   usable_video_frames = (video_frames_count - sync_init_frame)//(video_sample_rate/target_frame_rate) + 1
   usable_kinematic_frames = (rows_count - sync_init_row)//(kinematic_sample_Rate/target_frame_rate) + 1

   return (usable_video_frames, usable_kinematic_frames)

class RosmaDatasetBase(Dataset):
  def __init__(self,data_csv_df,config,frames_to_retrieve,sample_rate):

        self.frames_to_retrieve = frames_to_retrieve
        self.config = config
        self.df = data_csv_df
        self.sample_rate = sample_rate

        self.num_frames_per_video = []

        for index, row in self.df.iterrows():
            sync_init_frame = row['sync_start_frame']
            sync_init_row = row['sync_start_row']
            video_frames_count = row['FrameCount']
            rows_count = row['Kinematics Count']
            (usable_video_frames, usable_kinematic_frames) = get_usable_frame_count(sync_init_frame, sync_init_row, video_frames_count, rows_count, sample_rate)
            self.num_frames_per_video.append(min(usable_video_frames,usable_kinematic_frames))

        self.num_frames_per_video = np.array(self.num_frames_per_video) - self.frames_to_retrieve        
        self.cumulative_frames = [0] + list(np.cumsum(self.num_frames_per_video))
        

  def __len__(self):
      return int(sum(self.num_frames_per_video))

  def get_frame_indexes_and_rosma_meta_by_item_index(self,index):
      # Find the corresponding video
      video_index = bisect.bisect_right(self.cumulative_frames, index) - 1
      video_start_frame = self.cumulative_frames[video_index]

      # Find frame in the video resnet features (orig frame indexes divided by the sample resolution)
      frame_index_resolution = index - video_start_frame

      curr_video_sample_rate = self.df['VideoSampleRate[Hz]'][video_index]
      curr_kinematic_sample_rate = self.df['KinematicsSampleRate[Hz]'][video_index]

      # Frame index in the original video
      frame_index_orig = int((frame_index_resolution) * (curr_video_sample_rate/self.sample_rate)) + self.df['sync_start_frame'][video_index] - 1
      kinematic_row_orig = int((frame_index_resolution) * (curr_kinematic_sample_rate/self.sample_rate)) + self.df['sync_start_row'][video_index] - 1

      task_id = self.df['TaskID'][video_index]
      subject = self.df['Subject'][video_index]
      repetition = self.df['Repetition'][video_index]

      return frame_index_orig, kinematic_row_orig, task_id, subject, repetition

  def __getitem__(self, index):      
    #   frame_index_orig, kinematic_row_orig, task_id, subject, repetition = self.get_frame_indexes_and_rosma_meta_by_item_index(index)
    #   a = 5

    #   return frame_index_orig, kinematic_row_orig, task_id, subject, repetition
    pass