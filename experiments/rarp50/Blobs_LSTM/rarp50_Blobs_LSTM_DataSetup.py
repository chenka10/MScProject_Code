import torch

def unpack_batch_rarp50(batch, device):

    # load batch data
    frames = batch[0].to(device)    

    psm1_position, psm1_rotation = batch[1][0].to(device)*100, batch[1][1].to(device)
    psm2_position, psm2_rotation = batch[1][2].to(device)*100, batch[1][3].to(device)
    ecm_position, ecm_rotation = batch[1][4].to(device)*100, batch[1][5].to(device)
    
    kinematics = torch.cat([psm1_position, psm1_rotation, psm2_position, psm2_rotation],dim=-1)
    ecm_kinematics = torch.cat([ecm_position, ecm_rotation],dim=-1)
    batch_size = frames.size(0)  

    positions = torch.cat([psm1_position,psm2_position],dim=1)
   

    return (frames, kinematics,ecm_kinematics, positions, batch_size) 
        

