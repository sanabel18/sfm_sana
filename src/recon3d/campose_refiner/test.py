

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json

from campose_refiner.rig_mean_campose_refiner import RigMeanCamposeRefiner
from campose_refiner.rig_center_campose_refiner import RigCenterCamposeRefiner

from colorsys import hls_to_rgb

def rainbow_color_stops(n, end=2/3):
    return [ hls_to_rgb(end * i/(n-1), 0.5, 1) for i in range(n) ]

if __name__ == "__main__":
    input_sfm_data = sys.argv[1]

    # refiner = RigMeanCamposeRefiner({'num_rig_views': 6})
    refiner = RigCenterCamposeRefiner({'num_rig_views': 6})

    with open(input_sfm_data, 'r') as f:
        sfm_data = json.load(f)

    frames = sfm_data['extrinsics']
    
    vertex_line_list = []
    color_list = rainbow_color_stops(len(frames))
    for frame, color in zip(frames, color_list):
        center = frame['value']['center']
        vertex_line_list.append('{} {} {} {} {} {}\n'.format(
            center[0], center[1], center[2],
            int(255*color[0]), int(255*color[1]), int(255*color[2])))

    with open('/volume/cpl-dev/sfm/hsien/debug/sfm_o.ply', 'w') as f:
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


    refiner.refine(sfm_data)


    frames = sfm_data['extrinsics']
    
    vertex_line_list = []
    color_list = rainbow_color_stops(len(frames))
    for frame, color in zip(frames, color_list):
        center = frame['value']['center']
        vertex_line_list.append('{} {} {} {} {} {}\n'.format(
            center[0], center[1], center[2],
            int(255*color[0]), int(255*color[1]), int(255*color[2])))

    with open('/volume/cpl-dev/sfm/hsien/debug/sfm_r.ply', 'w') as f:
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
