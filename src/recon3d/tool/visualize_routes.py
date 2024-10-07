#!/opt/conda/bin/python

import os, sys
import json
from colorsys import hls_to_rgb

def rainbow_color_stops(n, end=2/3):
    return [ hls_to_rgb(end * i/(n-1), 0.5, 1) for i in range(n) ]

if __name__ == "__main__":
    input_routes = sys.argv[1]
    output_ply_folder = os.path.split(input_routes)[0]
    poses_ply_path = os.path.join(output_ply_folder, 'route_poses.ply')
    footprints_ply_path = os.path.join(output_ply_folder, 'route_footprints.ply')
    
    with open(input_routes, 'r') as f:
        route = json.load(f)

    frames = route['frames']
    
    for i in range(1, len(frames)):
        assert frames[i]['timestamp'] > frames[i-1]['timestamp']

    pose_vertex_line_list = []
    footprint_vertex_line_list = []
    color_list = rainbow_color_stops(len(frames))
    for frame, color in zip(frames, color_list):
        for lense_position in frame['lense_positions']:
            pose_vertex_line_list.append('{} {} {} {} {} {}\n'.format(
                lense_position[0], lense_position[1], lense_position[2],
                int(255*color[0]), int(255*color[1]), int(255*color[2])))
        for lense_footprint in frame['lense_footprints']:
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
    # {
    #                 'dir': img_upsmpl_dir, 
    #                 'slice': slice_idx, 
    #                 'frame_idx': frame_upsmpl_idx,
    #                 'timestamp': frame_upsmpl_idx / fps + ss, # frame_upsmpl_idx start from 0 !!
    #                 'filenames': self.metadata['image_upsmpl_file_dict'][frame_upsmpl_idx],
    #                 'lense_positions': [None] * n_lenses,
    #                 'lense_rotations': [None] * n_lenses,
    #                 'lense_footprints': [None] * n_lenses,
    #                 'position': None,
    #                 'rotation': None,
    #                 'footprint': None
    #             }
    #         )
    # ply
    # format ascii 1.0
    # element vertex 282955
    # property double x
    # property double y
    # property double z
    # property uchar red
    # property uchar green
    # property uchar blue
    # end_header
    # 0.0000339905635032 0.2888815937141413 0.7621399018830718 91 71 72
