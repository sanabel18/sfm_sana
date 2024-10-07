#!/opt/conda/bin/python

import os, sys
import json

from colorsys import hls_to_rgb

def rainbow_color_stops(n, end=2/3):
    return [ hls_to_rgb(end * i/(n-1), 0.5, 1) for i in range(n) ]

if __name__ == "__main__":
    input_sfm_data = sys.argv[1]
    output_ply_folder = os.path.split(input_sfm_data)[0]
    poses_ply_path = os.path.join(output_ply_folder, 'sfm_data_poses.ply')
    footprints_ply_path = os.path.join(output_ply_folder, 'sfm_data_footprints.ply')
    
    with open(input_sfm_data, 'r') as f:
        sfmdata_dict = json.load(f)

    frames = sfmdata_dict['extrinsics']
    
    pose_vertex_line_list = []
    footprint_vertex_line_list = []
    color_list = rainbow_color_stops(len(frames))
    for frame, color in zip(frames, color_list):
        lense_position = frame['value']['center']
        pose_vertex_line_list.append('{} {} {} {} {} {}\n'.format(
            lense_position[0], lense_position[1], lense_position[2],
            int(255*color[0]), int(255*color[1]), int(255*color[2])))
        lense_footprint = frame['value']['footprint']
        footprint_vertex_line_list.append('{} {} {} {} {} {}\n'.format(
            lense_footprint[0], lense_footprint[1], lense_footprint[2],
            int(255*color[0]), int(255*color[1]), int(255*color[2])))

    for filepath, vertex_line_list in \
        [[poses_ply_path, pose_vertex_line_list],
         [footprints_ply_path, footprint_vertex_line_list]]:
        
        with open(filepath, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex {}\n'.format(len(vertex_line_list)))
            f.write('property double x\n')
            f.write('property double y\n')
            f.write('property double z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            for vertex_line in vertex_line_list:
                f.write(vertex_line)    
