import sys
sys.path.append('/home/chen/MScProject/Code')
sys.path.append('/home/chen/MScProject/Code/Jigsaws')
sys.path.append('/home/chen/MScProject/Code/experiments')

import pandas as pd
import os
from Jigsaws.JigsawsConfig import main_config as config
from DataUtils import ConcatDataset
from Jigsaws.JigsawsKinematicsDataset import JigsawsKinematicsDataset
from Jigsaws.JigsawsImageDataset import JigsawsImageDataset
from Jigsaws.JigsawsGestureDataset import JigsawsGestureDataset
from torchvision.transforms import transforms
import torch
from utils import torch_to_numpy
from modelUtils import load_models, iterate_on_images

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



params = {
   'frame_size':64,
   'batch_size': 8,
   'num_epochs':100,
   'img_compressed_size': 256,
   'prior_size': 32,
   'subjects_num': 8,
   'past_count': 10,
   'future_count': 20,
   'num_gestures': 16,   
   'lr': 0.0005,
   'beta': 0.001,
   'gamma': 1000, 
   'conditioning':'gesture', #'gesture'   
   'dataset':'JIGSAWS'
}
params['seq_len'] = params['past_count'] + params['future_count']
if params['conditioning'] == 'position':
   params['added_vec_size'] = 32
elif params['conditioning'] == 'gesture':
   params['added_vec_size'] = 16
else:
   raise ValueError()


runid = 'd4im5xx1'

taskId = 2
subject = 'C'
repetition = 1

frame = 450

past_count = params['past_count']
future_count = params['future_count']

transform = transforms.Compose([
    transforms.ToTensor()
])

df = pd.read_csv(os.path.join(config.get_project_dir(),'jigsaws_all_data_detailed.csv'))
df_test = df[(df['Subject']=='C')].reset_index(drop=True)

jigsaws_sample_rate = 6
dataset_test = ConcatDataset(JigsawsImageDataset(df_test,config,past_count + future_count,transform,sample_rate=jigsaws_sample_rate),
                        JigsawsGestureDataset(df_test,config,past_count + future_count,sample_rate=jigsaws_sample_rate),
                        JigsawsKinematicsDataset(df_test,config,past_count + future_count,sample_rate=jigsaws_sample_rate))

models = load_models(runid, params, device)

frames_dir = f'/home/chen/MScProject/Code/experiments/LSTM_AutoEncoder_Clean/ModelTesting/{runid}_{frame}_{taskId}_{subject}_{repetition}'
os.makedirs(frames_dir, exist_ok=True)

for condition_gesture in range(1):

    batch = [b.unsqueeze(0) for b in dataset_test[frame]]
    orig_gestures = batch[1].detach().clone().cpu()
    # batch[1][0,past_count:] = condition_gesture
    generated_seq = iterate_on_images(models, params, batch, config, device)

    past_images_to_show = 3

    
    fig, axes = plt.subplots(2,past_count + future_count, figsize=(15,4))

    for i in range(future_count):
        axes[0,past_count+ i].imshow(torch_to_numpy(generated_seq[past_count-1+i][0,:,:,:].detach()))
        axes[0,past_count+ i].set_xticks([])
        axes[0,past_count+ i].set_yticks([])
        axes[0,past_count+ i].set_title(batch[1][0,past_count-1+i].item())

        axes[0,i].set_axis_off()        

        axes[1,i].imshow(torch_to_numpy(batch[0][0,i,:,:,:].detach()))
        axes[1,i].set_xticks([])
        axes[1,i].set_yticks([])
        axes[1,i].set_title(orig_gestures[0,i].item())

        axes[1,past_count+ i].imshow(torch_to_numpy(batch[0][0,past_count-1+i,:,:,:].detach()))
        axes[1,past_count+ i].set_xticks([])
        axes[1,past_count+ i].set_yticks([])
        axes[1,past_count+ i].set_title(orig_gestures[0,past_count-1+i].item())
        

    plt.tight_layout()
    plt.savefig(os.path.join(frames_dir,f'test_{condition_gesture}.png'))
    plt.close()