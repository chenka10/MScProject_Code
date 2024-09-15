import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from utils import torch_to_numpy

def visualize_frame_diff(images_dir, title, index_to_show, orig_frames, generated_seq, generated_grayscale_blob_maps, display_past_count, actual_past_count, display_future_count, epoch, orig_gestures=None):
  fig = plt.figure(figsize=(10,4))    
  for i in range(display_future_count):
    plt.subplot(3,display_future_count+display_past_count,display_past_count+i+1)
    plt.imshow(torch_to_numpy(generated_seq[actual_past_count-1+i][index_to_show,:,:,:].detach()))
    plt.xticks([])
    plt.yticks([])
  for i in range(display_future_count+display_past_count):
    plt.subplot(3,display_future_count+display_past_count,i+1+display_future_count+display_past_count)
    plt.imshow(torch_to_numpy(orig_frames[index_to_show,actual_past_count-display_past_count+i,:,:,:].detach()))
    if orig_gestures is not None:
      plt.title(orig_gestures[index_to_show,actual_past_count-display_past_count+i].item())

    if i>=display_past_count and generated_grayscale_blob_maps is not None:
      plt.imshow(torch_to_numpy(generated_grayscale_blob_maps[actual_past_count-display_past_count+i-1][index_to_show,:,:,:]), cmap='jet', alpha=0.15)

    plt.xticks([])
    plt.yticks([])

  plt.tight_layout()
  fig.savefig(os.path.join(images_dir,'epoch_{}_{}.png').format(epoch,title))
  plt.close()

def visualize_frame_diff_for_test(images_dir, title, index_to_show, orig_frames, generated_seq, generated_grayscale_blob_maps, display_past_count, actual_past_count, display_future_count, epoch, orig_gestures=None):
  fig = plt.figure(figsize=(13,2))   
  display_future_count=int(display_future_count/2)
  for i in range(display_future_count):    

    # display generated future frames
    plt.subplot(2,display_future_count+display_past_count,display_past_count+i+1)
    plt.imshow(torch_to_numpy(generated_seq[actual_past_count-1+i][index_to_show,:,:,:].detach()))
    plt.xticks([])
    plt.yticks([])

    # display ground-truth future frames
    plt.subplot(2,display_future_count+display_past_count,i+1+display_future_count+display_past_count*2)
    plt.imshow(torch_to_numpy(orig_frames[index_to_show,actual_past_count+i,:,:,:].detach())+0*torch_to_numpy(generated_grayscale_blob_maps[actual_past_count+i-1][index_to_show,:,:,:]))
    plt.xticks([])
    plt.yticks([])

    # if generated_grayscale_blob_maps is not None:
    #   plt.imshow(torch_to_numpy(generated_grayscale_blob_maps[actual_past_count+i-1][index_to_show,:,:,:]), cmap='gray', alpha=0.15)


  # display past ground-truth frames
  for i in range(display_past_count):
    plt.subplot(2,display_future_count+display_past_count,i+1+display_future_count+display_past_count)
    plt.imshow(torch_to_numpy(orig_frames[index_to_show,actual_past_count-display_past_count+i,:,:,:].detach()))
    plt.xticks([])
    plt.yticks([])

  plt.subplots_adjust(wspace=0, hspace=0)  
  
  fig.savefig(os.path.join(images_dir,'epoch_{}_{}.png').format(epoch,title))
  plt.close()