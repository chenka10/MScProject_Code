import numpy as np
import torch

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
    if len(image_tensor.shape) != 3 or image_tensor.shape[0] != 3:
        raise ValueError("Input tensor must have shape (3, H, W)")

    # Convert PyTorch tensor to NumPy array
    numpy_array = image_tensor.cpu().numpy()

    # Reorder dimensions from (C, H, W) to (H, W, C) for NumPy
    numpy_array = np.moveaxis(numpy_array, 0, -1)

    return numpy_array

def get_distance(input, target):
        # Reshape input and target tensors to separate left and right coordinates
        input_left = input[:, :, :3]  # Extract left coordinates (first 3 elements)
        input_right = input[:, :, 3:]  # Extract right coordinates (last 3 elements)
        target_left = target[:, :, :3]  # Extract left coordinates (first 3 elements)
        target_right = target[:, :, 3:]  # Extract right coordinates (last 3 elements)

        # Compute Euclidean distance between predicted and target coordinates
        loss_left = torch.sqrt(torch.sum((input_left - target_left) ** 2,dim=2))
        loss_right = torch.sqrt(torch.sum((input_right - target_right) ** 2,dim=2))

        # Total loss is the sum of left and right losses
        total_loss = loss_left + loss_right

        return total_loss/2