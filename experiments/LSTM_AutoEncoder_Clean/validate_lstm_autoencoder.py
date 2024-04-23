import sys
sys.path.append('/home/chen/MScProject/Code/')
sys.path.append('/home/chen/MScProject/Code/Jigsaws/')
sys.path.append('/home/chen/MScProject/Code/models/')

import wandb
wandb.login(key = '33514858884adc0292c3f8be3706845a1db35d3a')

from JigsawsKinematicsDataset import JigsawsKinematicsDataset
from JigsawsImageDataset import JigsawsImageDataset
from JigsawsGestureDataset import JigsawsGestureDataset
from JigsawsDatasetBase import JigsawsMetaDataset, ConcatDataset
from JigsawsConfig import main_config
from utils import torch_to_numpy
import matplotlib.pyplot as plt

from pytorch_msssim import ssim

import torch.optim as optim
from tqdm import tqdm
from torchModules import BasicFc
from torchvision import models as tv_models
from vgg import Encoder, Decoder
from lstm import gaussian_lstm, lstm
from losses import kl_criterion_normal
from torch.utils.data import DataLoader
import os
import torchvision.transforms as transforms
import pandas as pd
from utils import get_distance, expand_positions
import io
from PIL import Image

import statistics


position_indices = main_config.kinematic_slave_position_indexes

import torch
import torch.nn as nn

mse = nn.MSELoss(reduce=False)

def validate(models, dataloader_valid, params, device):

  frame_encoder, frame_decoder, prior_lstm, generation_lstm = models

  # storages for training metrics
  loss = torch.tensor([0.0,0.0,0.0])
  ssim_per_future_frame = torch.zeros((params['future_count']))
  for batch in tqdm(dataloader_valid):        

    # load batch data
    frames = batch[0].to(device)
    gestures = torch.nn.functional.one_hot(batch[1],params['num_gestures']).to(device)
    positions = batch[2][:,:,position_indices].to(device)
    positions_expanded = expand_positions(positions)
    batch_size = frames.size(0)

    # set models to eval
    for model in models:
      model.eval()
      model.batch_size = batch_size

    # prepare lstms for batch      
    prior_lstm.batch_size = batch_size
    prior_lstm.hidden = prior_lstm.init_hidden()
    generation_lstm.batch_size = batch_size
    generation_lstm.hidden = generation_lstm.init_hidden()

    # encode all frames (past and future)
    seq = [frame_encoder(frames[:,i,:,:,:]) for i in range(params['past_count'])]

    # storage for genrated frames
    generated_seq = []

    # storage for computed losses
    loss_MSE = torch.tensor(0.0).to(device)
    loss_KLD = torch.tensor(0.0).to(device)          

    # get avg. position diffrences for every sequence in the batch
    distance_weight = get_distance(positions[:,:-1,:],positions[:,1:,:])      
    batch_mover = distance_weight[:,params['past_count']:].sum(dim=1).argmax().item()
    batch_least_mover = distance_weight[:,params['past_count']:].sum(dim=1).argmin().item()
    distance_weight = distance_weight.sum(dim=1)/params['seq_len']

    for t in range(1,params['seq_len']):

      # keep loading past frames (for conditioning), once conditioning is over, load previously encoded frames
      if t <= params['past_count']:          
        frames_t_minus_one, skips = seq[t-1]
      else:          
        frames_t_minus_one = frame_encoder(decoded_frames)[0]       
      
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
    valid_loss += torch.tensor([loss_tot.item(),loss_MSE.item(),loss_KLD.item()])     
    
  loss /= len(dataloader_valid)
  ssim_per_future_frame /= len(dataloader_valid)

  return loss, ssim_per_future_frame, batch_mover, batch_least_mover, generated_seq, frames, batch


