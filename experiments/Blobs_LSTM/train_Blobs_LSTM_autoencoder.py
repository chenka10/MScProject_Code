from JigsawsConfig import main_config
from pytorch_msssim import ssim
from tqdm import tqdm
from losses import kl_criterion_normal
from utils import get_distance
from experiments.Blobs_LSTM.Blobs_LSTM_DataSetup import unpack_batch
from models.blobReconstructor import combine_blob_maps


position_indices = main_config.kinematic_slave_position_indexes
rotation_indices = main_config.kinematic_slave_rotation_indexes

import torch
import torch.nn as nn

import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg')

mse = nn.MSELoss(reduce=False)

def train(models, position_to_blobs, dataloader_train, optimizer, params, config, device):

  loss_fn_vgg.to(device)

  frame_encoder, frame_decoder, generation_lstm, blobs_to_maps = models

  # set all models to train 
  for model in models:
    model.train()

  # storages for training metrics
  loss = torch.tensor([0.0,0.0,0.0,0.0]) # [MSE + beta*KLD, MSE, KLD]
  ssim_per_future_frame = torch.zeros((params['future_count']))

  for batch in tqdm(dataloader_train):

    frames, gestures, gestures_onehot, positions, rotations, kinematics, batch_size = unpack_batch(params, config, batch, device) 

    # prepare lstms for batch (set internal batch-size and set default hidden state)
    generation_lstm.batch_size = batch_size
    generation_lstm.hidden = generation_lstm.init_hidden()

    # clear optimizer
    optimizer.zero_grad()

    # encode all frames (past and future)    
    seq = [frame_encoder(frames[:,i,:,:,:]) for i in range(params['seq_len'])]    

    # storage for genrated frames
    generated_seq = []

    # storage for computed losses
    loss_MSE = torch.tensor(0.0).to(device)
    loss_PER = torch.tensor(0.0).to(device)
    loss_KLD = torch.tensor(0.0).to(device)         

    # get avg. position diffrences for every sequence in the batch
    distance_weight = get_distance(positions[:,:-1,:],positions[:,1:,:]).sum(dim=1)/params['seq_len']        
    
    for t in range(1,params['seq_len']):

      blob_datas = position_to_blobs(positions[:,t,:])
      feature_maps_0, grayscale_maps_0 = blobs_to_maps[0](blob_datas[0])
      feature_maps_1, grayscale_maps_1 = blobs_to_maps[1](blob_datas[1])
      combined_blobs_feature_maps = combine_blob_maps(torch.zeros_like(feature_maps_0),
                                                      [feature_maps_0,feature_maps_1],
                                                      [grayscale_maps_0,grayscale_maps_1])

      # frames_t = seq[i][0]
      frames_t_minus_one = seq[t-1][0]

      # load skip connections from the frame encoder for all conditioned frames
      # once conditioned frames are over kuup using skips from the last conditioned frame
      if t <= params['past_count']:
        skips = seq[t-1][1] 
        skips[0] = torch.concat([skips[0],combined_blobs_feature_maps],1)
      else:
        skips[0][:,-combined_blobs_feature_maps.size(1):,:,:] = combined_blobs_feature_maps

      # predict next frame latent, decode next frame, store next frame
      frames_to_decode = generation_lstm(frames_t_minus_one.float())

      contains_nan = torch.isnan(frames_to_decode).any().item()
      if contains_nan:
        raise ValueError('frames to decode contains NaN')

      decoded_frames = frame_decoder([frames_to_decode,skips])
      generated_seq.append(decoded_frames)      

      # compute losses for generated frame
      mse_per_batch = mse(decoded_frames, frames[:,t,:,:,:]).sum(-1).sum(-1).sum(-1)
      loss_MSE += (distance_weight*mse_per_batch).mean()
      loss_PER += (distance_weight*loss_fn_vgg((decoded_frames*2)-1, (frames[:,t,:,:,:]*2)-1).sum(-1).sum(-1).sum(-1)).mean()      

      # for all predicted future frames compute SSIM with real future frames
      if t>=params['past_count']:
        ssim_per_batch = ssim(decoded_frames, frames[:,t,:,:,:],data_range=1, size_average=False)
        ssim_per_future_frame[t-params['past_count']] += (ssim_per_batch.mean().item())          

    loss_tot = loss_MSE + params['gamma']*loss_PER + params['beta']*loss_KLD

    loss_tot.backward()
    optimizer.step()

    loss += torch.tensor([loss_tot.item(),loss_MSE.item(),loss_PER.item(),loss_KLD.item()])     

  loss /= len(dataloader_train)
  ssim_per_future_frame /= len(dataloader_train)

  return loss, ssim_per_future_frame

