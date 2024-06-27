import bisect
from torch.utils.data import Dataset
import numpy as np

from Code.rarp50.rarp50Config import rarp50Config


class rarp50DatasetBase(Dataset):
    def __init__(self,data_csv_df,config,frames_to_retrieve, frame_increments = 12):

        self.frames_to_retrieve = frames_to_retrieve
        self.config: rarp50Config = config
        self.df = data_csv_df        
        self.num_frames_per_video = (self.df['numberOfFrames']).to_numpy().astype(int) - self.frames_to_retrieve + 1     
        self.cumulative_frames = [0] + list(np.cumsum(self.num_frames_per_video))    
        self.frame_increments = frame_increments    

    def get_videoStartFrame_and_videoIndex_by_dataset_index(self,dataset_index):
        video_index = bisect.bisect_right(self.cumulative_frames, dataset_index) - 1
        video_start_frame = (dataset_index - self.cumulative_frames[video_index])*self.frame_increments + self.df.at[video_index,'startFrame']

        return video_start_frame, video_index



    def __len__(self):
        return sum(self.num_frames_per_video) 

    def __getitem__(self, index):      
        pass