import sys
sys.path.append('/home/chen/MScProject/Code')
sys.path.append('/home/chen/MScProject/Code/Jigsaws')
sys.path.append('/home/chen/MScProject/Code/experiments')

import os
import torch
from models.lstm import gaussian_lstm, lstm
from models.vgg import Encoder, Decoder
from models.vgg128 import Encoder128, Decoder128
from experiments.LSTM_AutoEncoder_Clean.DataSetup import unpack_batch


def load_models(runid,params,device):
    path = f'/home/chen/MScProject/Code/experiments/LSTM_AutoEncoder_Clean/models_{params['conditioning']}/'

    frame_encoder = None
    frame_decoder = None
    prior_lstm = None
    generation_lstm = None

    for dir_name in os.listdir(path):
        if runid in dir_name:
            models_dir = os.path.join(path, dir_name)  

            if params['frame_size'] == 128:
                frame_encoder = Encoder128(params['img_compressed_size'],3).to(device)
                frame_decoder = Decoder128(params['img_compressed_size'],3).to(device)                
            else:   
                frame_encoder = Encoder(params['img_compressed_size'],3).to(device)                
                frame_decoder = Decoder(params['img_compressed_size'],3).to(device)

            frame_encoder.load_state_dict(torch.load(os.path.join(models_dir,'frame_encoder.pth')))
            frame_decoder.load_state_dict(torch.load(os.path.join(models_dir,'frame_decoder.pth')))

            prior_lstm = gaussian_lstm(params['img_compressed_size'],params['prior_size'],256,1,params['batch_size'],device).to(device)
            generation_lstm = lstm(params['img_compressed_size'] + params['prior_size'] + params['added_vec_size'],params['img_compressed_size'],256,2,params['batch_size'],device).to(device)          

            prior_lstm.load_state_dict(torch.load(os.path.join(models_dir,'prior_lstm.pth')))
            generation_lstm.load_state_dict(torch.load(os.path.join(models_dir,'generation_lstm.pth')))

    return frame_encoder, frame_decoder, prior_lstm, generation_lstm

def iterate_on_images(models, params, batch, config, device):
     
    frames, gestures, gestures_onehot, positions, rotations, kinematics, batch_size = unpack_batch(params, config, batch, device)      

    # set models to eval
    for model in models:
      model.eval()
      model.batch_size = batch_size

    frame_encoder, frame_decoder, prior_lstm, generation_lstm = models

    # prepare lstms for batch      
    prior_lstm.batch_size = batch_size
    prior_lstm.hidden = prior_lstm.init_hidden()
    generation_lstm.batch_size = batch_size
    generation_lstm.hidden = generation_lstm.init_hidden()

    # encode all frames (past and future)
    seq = [frame_encoder(frames[:,i,:,:,:]) for i in range(params['past_count'])]
    generated_seq = []

    for t in range(1,params['seq_len']):

        # keep loading past frames (for conditioning), once conditioning is over, load previously encoded frames
        if t <= params['past_count']:          
            frames_t_minus_one, skips = seq[t-1]
        else:          
            frames_t_minus_one = frame_encoder(decoded_frames)[0]       
        
        z,mu,logvar = prior_lstm(frames_t_minus_one)        

        # load condition data of current frame
        if params['conditioning'] == 'position':
            conditioning_vec = kinematics[:,t,:]
        elif params['conditioning'] == 'gesture':
            conditioning_vec = gestures_onehot[:,t,:]

        # predict next frame latent, decode next frame, store next frame
        frames_to_decode = generation_lstm(torch.cat([frames_t_minus_one,mu,conditioning_vec],dim=-1).float())
        decoded_frames = frame_decoder([frames_to_decode,skips])
        generated_seq.append(decoded_frames.detach().cpu())

    return generated_seq