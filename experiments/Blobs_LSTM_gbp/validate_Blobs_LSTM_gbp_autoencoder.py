import sys
sys.path.append('/home/chen/MScProject/Code/')
sys.path.append('/home/chen/MScProject/Code/Jigsaws/')
sys.path.append('/home/chen/MScProject/Code/models/')

import wandb
wandb.login(key = '33514858884adc0292c3f8be3706845a1db35d3a')

from JigsawsConfig import main_config
from Jigsaws.JigsawsUtils import jigsaws_to_quaternions
from pytorch_msssim import ssim
from tqdm import tqdm
from models.blobReconstructor import combine_blob_maps

from losses import kl_criterion_normal
from utils import get_distance, expand_positions

from Code.experiments.Blobs_LSTM.Blobs_LSTM_DataSetup import unpack_batch
position_indices = main_config.kinematic_slave_position_indexes
rotation_indices = main_config.kinematic_slave_rotation_indexes

import torch
import torch.nn as nn

mse = nn.MSELoss(reduce=False)

import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg')

def validate(models, position_to_blobs, dataloader_valid, params, config, device):
  worst_batch_mse = 0.0
  best_batch_mse = 9999999.0

  # touples that will contain batch, generated_seq, and batch_index of the worst and best batches (based on mse of specific entry in the batch)
  worst_batch_seq_ind = None
  best_batch_seq_ind = None

  loss_fn_vgg.to(device)

  frame_encoder, frame_decoder, generation_lstm, blobs_to_maps = models

  # storages for training metrics
  loss = torch.tensor([0.0,0.0,0.0,0.0])
  ssim_per_future_frame = torch.zeros((params['future_count']))
  for batch in tqdm(dataloader_valid):        

    frames, gestures, gestures_onehot, positions, rotations, kinematics, batch_size = unpack_batch(params, config, batch, device)      

    # set models to eval
    for model in models:
      model.eval()
      model.batch_size = batch_size

    # prepare lstms for batch      
    generation_lstm.batch_size = batch_size
    generation_lstm.hidden = generation_lstm.init_hidden()

    # encode all frames (past and future)
    seq = [frame_encoder(frames[:,i,:,:,:]) for i in range(params['past_count'])]

    # storage for genrated frames
    generated_seq = []

    # storage for computed losses
    loss_MSE = torch.tensor(0.0).to(device)
    loss_MSE_per_batch = torch.zeros(batch_size)
    loss_PER = torch.tensor(0.0).to(device)
    loss_KLD = torch.tensor(0.0).to(device)          

    # get avg. position diffrences for every sequence in the batch
    distance_weight = get_distance(positions[:,:-1,:],positions[:,1:,:])      
    batch_mover = distance_weight[:,params['past_count']:].sum(dim=1).argmax().item()
    batch_least_mover = distance_weight[:,params['past_count']:].sum(dim=1).argmin().item()
    distance_weight = distance_weight.sum(dim=1)/params['seq_len']

    for t in range(1,params['seq_len']):

      blob_datas = position_to_blobs(kinematics[:,t,:])
      feature_maps = []
      grayscale_maps = []

      num_blobs = 4

      for i in range(len(blobs_to_maps)):
        f,g = blobs_to_maps[i](blob_datas[i%num_blobs])
        feature_maps.append(f)
        grayscale_maps.append(g)
      
      combined_blobs_feature_maps = []
      for i in range(len(blobs_to_maps)//num_blobs):
        combined_blobs_feature_maps.append(combine_blob_maps(torch.zeros_like(feature_maps[i*num_blobs]),
                                                        [feature_maps[i*num_blobs],feature_maps[i*num_blobs+1],feature_maps[i*num_blobs+2],feature_maps[i*num_blobs+3]],
                                                        [grayscale_maps[i*num_blobs],grayscale_maps[i*num_blobs+1],grayscale_maps[i*num_blobs+2],grayscale_maps[i*num_blobs+3]]))


      # keep loading past frames (for conditioning), once conditioning is over, load previously encoded frames
      if t <= params['past_count']:          
        frames_t_minus_one, skips = seq[t-1]
        skips[0] = torch.concat([skips[0],combined_blobs_feature_maps[0]],1)
        skips[1] = torch.concat([skips[1],combined_blobs_feature_maps[1]],1)
        skips[2] = torch.concat([skips[2],combined_blobs_feature_maps[2]],1)
      else:          
        frames_t_minus_one = frame_encoder(decoded_frames)[0]   
        skips[0][:,-combined_blobs_feature_maps[0].size(1):,:,:] = combined_blobs_feature_maps[0]
        skips[1][:,-combined_blobs_feature_maps[1].size(1):,:,:] = combined_blobs_feature_maps[1]
        skips[2][:,-combined_blobs_feature_maps[2].size(1):,:,:] = combined_blobs_feature_maps[2]  

      # predict next frame latent, decode next frame, store next frame
      frames_to_decode = generation_lstm(frames_t_minus_one.float())
      decoded_frames = frame_decoder([frames_to_decode,skips])
      generated_seq.append(decoded_frames.detach().cpu())
      
      # compute losses for generated frame
      mse_per_batch = mse(decoded_frames, frames[:,t,:,:,:]).sum(-1).sum(-1).sum(-1)
      loss_MSE_per_batch += mse_per_batch.cpu()
      loss_MSE += (distance_weight*mse_per_batch).mean()
      loss_PER += (distance_weight*loss_fn_vgg((decoded_frames*2)-1, (frames[:,t,:,:,:]*2)-1).sum(-1).sum(-1).sum(-1)).mean()      

      # for all predicted future frames compute SSIM with real future frames
      if t>=params['past_count']:
        ssim_per_batch = ssim(decoded_frames, frames[:,t,:,:,:],data_range=1, size_average=False)
        ssim_per_future_frame[t-params['past_count']] += (ssim_per_batch.mean().item())

    # save worst and best batch in terms of mse for qualitative display later
    worst_mse_batch_index = loss_MSE_per_batch.argmax().item()
    best_mse_batch_index = loss_MSE_per_batch.argmin().item()

    if loss_MSE_per_batch[worst_mse_batch_index]>worst_batch_mse: 
      worst_batch_mse = loss_MSE_per_batch[worst_mse_batch_index]
      worst_batch_seq_ind = ([frames.detach().cpu(), gestures.detach().cpu()],generated_seq,worst_mse_batch_index)

    if loss_MSE_per_batch[best_mse_batch_index]<best_batch_mse:
      best_batch_mse = loss_MSE_per_batch[best_mse_batch_index]
      best_batch_seq_ind = ([frames.detach().cpu(), gestures.detach().cpu()],generated_seq,best_mse_batch_index)

    loss_tot = loss_MSE + loss_KLD
    loss += torch.tensor([loss_tot.item(),loss_MSE.item(),loss_PER.item(),loss_KLD.item()])  

  loss /= len(dataloader_valid)
  ssim_per_future_frame /= len(dataloader_valid)

  mover_batch_seq_ind = ([frames.detach().cpu(), gestures.detach().cpu()], generated_seq, batch_mover)
  non_mover_batch_seq_ind = ([frames.detach().cpu(), gestures.detach().cpu()], generated_seq, batch_least_mover)

  return loss, ssim_per_future_frame, mover_batch_seq_ind, non_mover_batch_seq_ind, best_batch_seq_ind, worst_batch_seq_ind


