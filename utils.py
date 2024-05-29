import numpy as np
import torch
import math

def utils_find_series_in_arr(arr,series):
    series_len = series.shape[0]
    arr_len = arr.shape[0]

    indexes = np.array(range(arr_len - series_len))

    result_array = np.array([np.all(arr[i:i+series_len]==series) for i in range(len(arr)-series_len+1)])
    return result_array

def utils_save_pytorch_model(model,file_path):
  torch.save(model, file_path)

def torch_to_numpy(image_tensor):
    """
    Convert a PyTorch tensor representing a 3-channel image to a NumPy array.

    Args:
        image_tensor (torch.Tensor): PyTorch tensor representing a 3-channel image.

    Returns:
        numpy.ndarray: NumPy array representing the same image.
    """
    # Ensure the input tensor is 3-dimensional with three channels
    if len(image_tensor.shape) != 3:
        raise ValueError("Input tensor must have shape (3, H, W)")

    # Convert PyTorch tensor to NumPy array
    numpy_array = image_tensor.cpu().numpy()

    # Reorder dimensions from (C, H, W) to (H, W, C) for NumPy
    numpy_array = np.moveaxis(numpy_array, 0, -1)

    return numpy_array

def get_distance(input, target):
        
        input_size = len(input.shape)

        if input_size==3:
            # Reshape input and target tensors to separate left and right coordinates
            input_left = input[:, :, :3]  # Extract left coordinates (first 3 elements)
            input_right = input[:, :, 3:]  # Extract right coordinates (last 3 elements)
            target_left = target[:, :, :3]  # Extract left coordinates (first 3 elements)
            target_right = target[:, :, 3:]  # Extract right coordinates (last 3 elements)

        if input_size==2:
            # Reshape input and target tensors to separate left and right coordinates
            input_left = input[:,:3]  # Extract left coordinates (first 3 elements)
            input_right = input[:,3:]  # Extract right coordinates (last 3 elements)
            target_left = target[:,:3]  # Extract left coordinates (first 3 elements)
            target_right = target[:,3:]  # Extract right coordinates (last 3 elements)
        


        # Compute Euclidean distance between predicted and target coordinates
        loss_left = torch.sqrt(torch.sum((input_left - target_left) ** 2,dim=input_size-1))
        loss_right = torch.sqrt(torch.sum((input_right - target_right) ** 2,dim=input_size-1))

        # Total loss is the sum of left and right losses
        total_loss = loss_left + loss_right

        return total_loss/2

def expand_positions(positions):
   positions = positions.repeat(1,1,4)
   pos_multiplier = torch.tensor([1,1,1,1,1,1,10,10,10,10,10,10,100,100,100,100,100,100,1000,1000,1000,1000,1000,1000]).to(positions.device)

   positions = positions*pos_multiplier

   return positions

def flatRotMat_to_quaternion(flat_rot_mat: torch.Tensor):
  '''
  This function is intended to work with batch of sequences.  
  So, input size is [batch_size, seq_len, 9]

  Output size is [batch_size, seq_len, 4]
  '''
  R = flat_rot_mat.reshape(flat_rot_mat.shape[0],flat_rot_mat.shape[1],3,3)

  epsilon = 10**-6

  q0 = 0.5*torch.sqrt(1+ R[:,:,0,0] + R[:,:,1,1] + R[:,:,2,2] + epsilon)
  q1 = (R[:,:,2, 1] - R[:,:,1, 2]) / (4 * q0)
  q2 = (R[:,:,0, 2] - R[:,:,2, 0]) / (4 * q0)
  q3 = (R[:,:,1, 0] - R[:,:,0, 1]) / (4 * q0)

  if torch.isinf(q1).any().item():
      raise ValueError('q1 contains infinite values')


  return torch.stack([q0, q1, q2, q3],dim=-1)