from JigsawsConfig import main_config
from pytorch_msssim import ssim
from tqdm import tqdm
from losses import kl_criterion_normal
from utils import get_distance, expand_positions

position_indices = main_config.kinematic_slave_position_indexes

import torch
import torch.nn as nn

mse = nn.MSELoss(reduce=False)

def train(models, dataloader_train, optimizer, params, device):

  frame_encoder, frame_decoder, prior_lstm, generation_lstm = models

  # set all models to train 
  for model in models:
    model.train()

  # storages for training metrics
  loss = torch.tensor([0.0,0.0,0.0]) # [MSE + beta*KLD, MSE, KLD]
  ssim_per_future_frame = torch.zeros((params['future_count']))

  for batch in tqdm(dataloader_train):

    # load batch data
    frames = batch[0].to(device)
    gestures = torch.nn.functional.one_hot(batch[1],params['num_gestures']).to(device)
    positions = batch[2][:,:,position_indices].to(device)
    positions_expanded = expand_positions(positions)
    batch_size = frames.size(0)    

    # prepare lstms for batch (set internal batch-size and set default hidden state)
    prior_lstm.batch_size = batch_size
    prior_lstm.hidden = prior_lstm.init_hidden()
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
    loss_KLD = torch.tensor(0.0).to(device)         

    # get avg. position diffrences for every sequence in the batch
    distance_weight = get_distance(positions[:,:-1,:],positions[:,1:,:]).sum(dim=1)/params['seq_len']        
    
    for t in range(1,params['seq_len']):
      # frames_t = seq[i][0]
      frames_t_minus_one = seq[t-1][0]

      # load skip connections from the frame encoder for all conditioned frames
      # once conditioned frames are over kuup using skips from the last conditioned frame
      if t <= params['past_count']:
        skips = seq[t-1][1]            

      # compute prior (z) using prior lstm
      z,mu,logvar = prior_lstm(frames_t_minus_one)

      # load condition data of current frame
      if params['conditioning'] == 'position':
        conditioning_vec = positions_expanded[:,t,:]
      elif params['conditioning'] == 'gesture':
        conditioning_vec = gestures[:,t,:]

      # predict next frame latent, decode next frame, store next frame
      frames_to_decode = generation_lstm(torch.cat([frames_t_minus_one,z,conditioning_vec],dim=-1))
      decoded_frames = frame_decoder([frames_to_decode,skips])
      generated_seq.append(decoded_frames)      

      # compute losses for generated frame
      mse_per_batch = mse(decoded_frames, frames[:,t,:,:,:]).sum(-1).sum(-1).sum(-1)
      loss_MSE += (distance_weight*mse_per_batch).mean()
      loss_KLD += params['beta']*kl_criterion_normal(mu,logvar) 

      # for all predicted future frames compute SSIM with real future frames
      if t>=params['past_count']:
        ssim_per_batch = ssim(decoded_frames, frames[:,t,:,:,:],data_range=1, size_average=False)
        ssim_per_future_frame[t-params['past_count']] += (ssim_per_batch.mean().item())

    loss_tot = loss_MSE + loss_KLD

    loss_tot.backward()
    optimizer.step()

    loss += torch.tensor([loss_tot.item(),loss_MSE.item(),loss_KLD.item()])  

  loss /= len(dataloader_train)
  ssim_per_future_frame /= len(dataloader_train)

  return loss, ssim_per_future_frame

