
import torch

from Code.models.blobReconstructor import combine_blob_maps


def generate_blob_maps(blobs_to_maps, blob_datas):
    feature_maps = []
    grayscale_maps = []

    num_blobs = 2

    for i in range(len(blobs_to_maps)):
        f,g = blobs_to_maps[i](blob_datas[i%num_blobs])
        feature_maps.append(f)
        grayscale_maps.append(g)

    combined_blobs_feature_maps = []
    for i in range(len(blobs_to_maps)//num_blobs):
        combined_blobs_feature_maps.append(combine_blob_maps(torch.zeros_like(feature_maps[i*num_blobs]),
                                                    [feature_maps[i*num_blobs],feature_maps[i*num_blobs+1],feature_maps[i*num_blobs+2],feature_maps[i*num_blobs+3]],
                                                    [grayscale_maps[i*num_blobs],grayscale_maps[i*num_blobs+1],grayscale_maps[i*num_blobs+2],grayscale_maps[i*num_blobs+3]]))
        
    return combined_blobs_feature_maps