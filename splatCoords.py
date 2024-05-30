import sys
sys.path.append('/home/chen/MScProject/Code')
import torch

def rotation_matrices(thetas):
    # Calculate cosines and sines of thetas
    cos_thetas = torch.cos(thetas)
    sin_thetas = torch.sin(thetas)
    
    # Construct rotation matrices
    rotation_matrices = torch.stack([
        torch.stack([cos_thetas, -sin_thetas], dim=-1),
        torch.stack([sin_thetas, cos_thetas], dim=-1)
    ], dim=-2)
    
    return rotation_matrices

def splat_coord(blob_data,size=64,c=500):   

    device = blob_data.device 

    x_inputs = blob_data[:,:2]
    s_inputs = blob_data[:,2]
    a_inputs = blob_data[:,3]
    theta_inputs = blob_data[:,4]

    # Create 1D tensors representing the x and y coordinates
    x = torch.arange(size)
    y = torch.arange(size)

    # Create meshgrid using torch.meshgrid
    X, Y = torch.meshgrid(x, y)
    X = X/size
    Y = Y/size
    Xgrid = torch.stack([X,Y]).to(x_inputs.device)

    d = Xgrid.unsqueeze(0) - x_inputs.unsqueeze(-1).unsqueeze(-1)

    aspec_mats = torch.stack([
        torch.stack([a_inputs**2,torch.zeros_like(a_inputs)],dim=-1),
        torch.stack([torch.zeros_like(a_inputs),1/(a_inputs**2)],dim=-1)
    ],dim=-2).to(device)
    
    rotation_mats = rotation_matrices(theta_inputs).to(device)
    rotation_mats_T = rotation_matrices(-1*theta_inputs).to(device)

    mat = torch.matmul(torch.matmul(rotation_mats,aspec_mats),rotation_mats_T)
    mat = torch.linalg.inv(mat)
    
    matmul = torch.matmul(mat[:,None,None,:,:],d.permute(0,2,3,1).unsqueeze(-1))
    matmul = matmul.squeeze(-1).permute(0,3,1,2)

    matmul = (matmul*d).sum(1)        
    res = torch.sigmoid(s_inputs[:,None,None] - c*matmul)

    return res.unsqueeze(1)

def generate_blobs_image(blobs,img_size=64):

    res_image = torch.zeros([1,img_size,img_size])

    for blob_data in blobs:
        curr_splat = splat_coord(blob_data,img_size)
        res_image = torch.mul(res_image,1-curr_splat)
        res_image = torch.add(res_image,curr_splat)

    return res_image
