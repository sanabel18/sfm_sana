import os, sys

def get_image_file_dict(image_dir: str):
    # image_dir can be with
    #   1. [image id].jpg [image id]_mask.png  => key_idx = [image id]
    #   2. [video id]_[frame id].jpg [video id]_[frame id]_mask.png  => key_idx = [video id]
    #   3. [frame id]_[camera id].jpg [frame id]_[camera id]_mask.png  => key_idx = [frame id]
    # e.g. image_dir with [frame id]_[camera id].jpg [frame id]_[camera id]_mask.png
    #       00001_1.jpg         
    #       00001_1_mask.png  
    #       00001_2.jpg     
    #       00001_2_mask.png  
    #       00002_1.jpg         
    #       00002_1_mask.png  
    #       00002_2.jpg     
    #       00002_2_mask.png 
    #  return image_file_dict
    #           {1: [00001_1.jpg, 00001_1_mask.png, 00001_2.jpg, 00001_2_mask.png], 
    #            2: [00002_1.jpg, 00002_1_mask.png, 00002_2.jpg, 00002_2_mask.png]}
    #         key_idx_list
    #           [1, 2]
    image_file_dict = dict()
    image_file_mask_dict = dict()
    image_files = os.listdir(image_dir)
    for image_file in image_files:
        key_idx = int(image_file.split('.')[0].split('_')[0]) 
        if key_idx not in image_file_dict:
            image_file_dict[key_idx] = []
            image_file_mask_dict[key_idx] = []
        if '_mask' in image_file:
            image_file_mask_dict[key_idx].append(image_file)
        else:
            image_file_dict[key_idx].append(image_file)

    for key_idx in image_file_dict:
        image_file_dict[key_idx].sort()
        image_file_mask_dict[key_idx].sort()

    key_idx_list = sorted(image_file_dict.keys())

    return image_file_dict, image_file_mask_dict, key_idx_list