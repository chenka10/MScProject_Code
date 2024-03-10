from JigsawsDatasetBase import JigsawsDatasetBase
import torch
import JigsawsUtils as ju
import torchvision.transforms as T

# Define a custom transform to change the data type of a tensor
class ChangeTypeTransform:
    def __init__(self, new_dtype):
        self.new_dtype = new_dtype

    def __call__(self, tensor):
        return tensor.to(dtype=self.new_dtype)

class JigsawsImageDataset(JigsawsDatasetBase):
  def __init__(self, data_csv_df, config, frames_to_retrieve=1, transforms = None, sample_rate=6):
      super().__init__(data_csv_df, config, frames_to_retrieve,sample_rate)

      if transforms is None:
        self.transforms = T.Compose([
            T.ToTensor(),            
            T.Resize((224,224)),            
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
      else:
        self.transforms = transforms


  def __len__(self):
    return super().__len__()

  def __getitem__(self, index):    
    # kinematicVec_norm, kinematicVec_orig, reward = super().__getitem__(index)
    frame_index_resolution, frame_index_orig, task_id, subject, repetition = super().get_frame_indexes_and_jigsaws_meta_by_item_index(index)

    frames = torch.tensor([])

    for i in range(self.frames_to_retrieve):
      frame = ju.jigsaws_get_extracted_frame(self.config, frame_index_orig + i*self.sample_rate, task_id, subject, repetition)
      frame = self.transforms(frame)
      min = frame.min()
      max = frame.max()
      frame = ((frame - min)/(max - min))      
      frames = torch.cat((frames, frame.unsqueeze(0)), dim=0)

    return frames
