#!/opt/conda/bin/python

import os, sys
import pickle 
import shutil
from project import PreprocProj, ReconProj, LocProj, ExportProj
import json, yaml
from utils.safe_yaml import safe_yaml_load, safe_yaml_dump
from utils.exportutil import copy_2_tower
import cProfile
import pstats
import io
from super_project import SuperProject, SuperProjectSrc, find_filename_w_label



if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--tower_dir", required=True, help='tower repo directory')
    parser.add_argument("-f", "--template_file", required=True, help='template file')
    parser.add_argument("-p", "--proj_dir_root", required=True, help='root directory of project')
    input_vals = parser.parse_args()
    
    tower_dir = input_vals.tower_dir
    template_file = input_vals.template_file
    proj_dir_root = input_vals.proj_dir_root

    profiler = cProfile.Profile()
    profiler.enable()
    video_path = find_filename_w_label(tower_dir, '@stitched-video')
    src_marker = find_filename_w_label(tower_dir, '@marker')
    gps_track  = find_filename_w_label(tower_dir, '@gpstrack')
    
    super_src = SuperProjectSrc(tower_dir, src_marker, gps_track, video_path, proj_dir_root)
    super_cfg_src_file = super_src.gen_super_src_yaml()
    SuperProject(super_cfg_src_file, template_file).run()
    profiler.disable()
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
    stats.print_stats(.1)
    outpath = os.path.join(tower_dir,'speed_test.prof')
    with open(outpath, 'w+') as f:
        f.write(s.getvalue())
    f.close()
