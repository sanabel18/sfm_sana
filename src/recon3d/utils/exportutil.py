import copy
import json
import numpy as np
import os
import shutil
import tarfile

from colorsys import hls_to_rgb
from os.path import join
from scipy.spatial.transform import Rotation as R

from utils.genutil import safely_create_dir, get_current_time_str, decompose_trf_mat
from utils.meshlabutil import GenMesh
from utils.safe_json import safe_json_dump

TAR_FILE_IMG_PRJ = 'img_prj.tar.gz'
TAR_FILE_MATCHES = 'matches.tar.gz'

def rainbow_color_stops(n, end=2/3):
    return [ hls_to_rgb(end * i/(n-1), 0.5, 1) for i in range(n) ]
    

def export_route_visualization(route_json_path: str, logger) -> int:
    """
    export routes.json to *ply files for visualization
    camera poses will be exported as route_poses.ply
    if footprint exists in routes.json, then footprints will be exported as route_footprints.ply
    Args: route_json_path: str, path of routes.json file
    """
    route_ply_folder = os.path.split(route_json_path)[0]
    poses_ply_path = join(route_ply_folder, 'route_poses.ply')
    footprints_ply_path = join(route_ply_folder, 'route_footprints.ply')
    
    with open(route_json_path, 'r') as f:
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
        # only export footprint when it exists in  frame
        if (frame['lense_footprints'][0]):
            if_footprint = True
        else:
            if_footprint = False
        if if_footprint:
            for lense_footprint in frame['lense_footprints']:
                footprint_vertex_line_list.append('{} {} {} {} {} {}\n'.format(
                    lense_footprint[0], lense_footprint[1], lense_footprint[2],
                    int(255*color[0]), int(255*color[1]), int(255*color[2])))

    file_path_vertex_list = [[poses_ply_path, pose_vertex_line_list]]
    if len(footprint_vertex_line_list) > 0:
        file_path_vertex_list.append([footprints_ply_path, footprint_vertex_line_list])

    for filepath, vertex_line_list in file_path_vertex_list: 
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
        logger.info(f'Successfully exported {filepath}.')

    return 0

def export_sfm_data_visualization(sfm_data_json_path: str, logger) -> int:
    """
    structure of sfm_data model dict
    "sfm_data_version": str
    "root_path": str: path of input images
    "views": list of dict{}
        {
            "key": int,
            "value": {
                "polymorphic_id": int,
                "ptr_wrapper": {
                    "id": int,
                    "data": {
                        "local_path": str
                        "filename": str, image file name
                        "width": int, image width
                        "height": int, image height
                        "id_view": int,
                        "id_intrinsic": int,
                        "id_pose": int
                        }
                    }
                }
            }
    "extrinsics": camera extrinsics parameters
    list of dict{}
    {
        "key": int, corresponds to the key defined in views
        "value": 
        {
            "rotation": [3x3] list, rotation matrix of camera pose
            "center": [3] list, camera position of camera pose
        }
 
    }


    export sfm_data in sfm_data_json_path to *ply files for visualization
    camera poses will be exported as 'original_sfm_data_file_name'_poses.ply
    if footprint exists in routes.json, then footprints will be exported as 'original_sfm_data_file_name'_footprints.ply
 
    Args:
    sfm_data_json_path: str, original sfm_data file path
    """
    output_ply_folder = os.path.split(sfm_data_json_path)[0]
    sfm_data_file_name = os.path.basename(sfm_data_json_path)
    sfm_data_file_pose_name = sfm_data_file_name.replace('.json','_poses.ply')
    sfm_data_file_footprint_name = sfm_data_file_name.replace('.json','_footprints.ply')
    poses_ply_path = os.path.join(output_ply_folder, sfm_data_file_pose_name)
    footprints_ply_path = os.path.join(output_ply_folder, sfm_data_file_footprint_name)
    
    with open(sfm_data_json_path, 'r') as f:
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

        if 'footprint' in frame['value']:
            lense_footprint = frame['value']['footprint']
            footprint_vertex_line_list.append('{} {} {} {} {} {}\n'.format(
                lense_footprint[0], lense_footprint[1], lense_footprint[2],
                int(255*color[0]), int(255*color[1]), int(255*color[2])))
    
    file_path_vertex_list = [[poses_ply_path, pose_vertex_line_list]]
    # only export footprint to file if footprint data exists
    if len(footprint_vertex_line_list) > 0:
        file_path_vertex_list.append([footprints_ply_path, footprint_vertex_line_list])

    for filepath, vertex_line_list in file_path_vertex_list: 
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
    
    return 0


def export_route_obj_visualizations(data_dir: str, \
                                    empty_route_file_path: str,\
                                    n_lense_for_rot: int, \
                                    logger) -> int:
    """

    1. from sfm_data_transformed.json, parse sfm data 
    2. use sfm data to create routes.json
    3. export to visulization (.ply files) for both sfm data and routes.json

    structure of sfm_data model dict
    "sfm_data_version": str
    "root_path": str: path of input images
    "views": list of dict{}
        {
            "key": int,
            "value": {
                "polymorphic_id": int,
                "ptr_wrapper": {
                    "id": int,
                    "data": {
                        "local_path": str
                        "filename": str, image file name
                        "width": int, image width
                        "height": int, image height
                        "id_view": int,
                        "id_intrinsic": int,
                        "id_pose": int
                        }
                    }
                }
            }
    "extrinsics": camera extrinsics parameters
    list of dict{}
    {
        "key": int, corresponds to the key defined in views
        "value": 
        {
            "rotation": [3x3] list, rotation matrix of camera pose
            "center": [3] list, camera position of camera pose
        }
 
    }



    Args:
    data_dir:str , path that contains routes.json
    empty_route_file_path: str, file path of empty routes.json that contains template of routes.json
    n_lense_for_rot: int, which camera to be use as assigning orientation. if equrectagular, should be zero.
    Returns:
    routes.json file written as filled_route_file_path
    """
    # Definition of names & paths
    input_sfm_data_path = join(data_dir, 'sfm_data_transformed.json') # TODO: is using this right?
    trf_path = join(data_dir, 'transformation.json')
    filled_route_file_path = join(data_dir, 'routes.json')
    # Load route file
    with open(empty_route_file_path, 'r') as f:
        route_dict = json.load(f)

    # Fill route dict
    route_dict['last_overwritten'] = get_current_time_str()
    if route_dict['is_scene'] == True:
        # Load sfm data
        with open(input_sfm_data_path, 'r') as f:
            camera_dict = json.load(f)

        def parse_sfm_data(sfm_data):
            id_view_list = []
            filename_to_id_view_dict = dict()
            id_view_to_id_pose_dict = dict()
            pose_dict = dict()
            for view in sfm_data['views']:
                id_view = view['value']['ptr_wrapper']['data']['id_view']
                id_pose = view['value']['ptr_wrapper']['data']['id_pose']
                filename = view['value']['ptr_wrapper']['data']['filename']
                id_view_list.append(id_view)
                filename_to_id_view_dict[filename] = id_view
                id_view_to_id_pose_dict[id_view] = id_pose
            for extrinsic in sfm_data['extrinsics']:
                pose_dict[extrinsic['key']] = extrinsic
            
            id_view_list = np.asarray(id_view_list)
            
            return id_view_list, filename_to_id_view_dict, id_view_to_id_pose_dict, pose_dict

        id_view_list, filename_to_id_view_dict, id_view_to_id_pose_dict, pose_dict = parse_sfm_data(camera_dict)

        for frame in route_dict['frames']:
            for i_lense, filename in enumerate(frame['filenames']):
                if filename in filename_to_id_view_dict:
                    pose = pose_dict[id_view_to_id_pose_dict[filename_to_id_view_dict[filename]]]['value']
                    frame['lense_positions'][i_lense] = pose['center']
                    frame['lense_rotations'][i_lense] = pose['rotation']
                    #frame['lense_footprints'][i_lense] = pose['footprint']

        # Merge positions of different lenses
        invalid_frame_ids = []
        for idx, frame in enumerate(route_dict['frames']):
            if ((None in frame['lense_positions']) or
                (None in frame['lense_rotations'])):
                #(None in frame['lense_footprints'])):
                logger.warning(f'Incomplete camera poses for frame {idx}. Put None as merged pose.')
                invalid_frame_ids.append(idx)
                # Do nothing, since default value of position, rotation & footprint is already None.
            else:
                frame['position'] = np.mean(np.array(frame['lense_positions']), axis=0)
                frame['rotation'] = frame['lense_rotations'][n_lense_for_rot]
                if not (None in frame['lense_footprints']):
                    frame['footprint'] = np.mean(np.array(frame['lense_footprints']), axis=0)
        # Remove invalid frames (Frontend doesn't want to see them)
        for idx in reversed(invalid_frame_ids):
            # Pop invalid frame from the end so that the indexing works
            route_dict['frames'].pop(idx)     
    else:
        # Load sfm data
        with open(trf_path, 'r') as f:
            trf_mat = json.load(f)['concatenated']

        trf_dict = decompose_trf_mat(trf_mat)
        # Sequence: Rotation -> offset
        # TODO (hkazami): Clarify the sequence of the front-end (should be the same)
        route_dict['position'] = trf_dict['offset']  # Offset is defined in the target coordinate so no scale needed
        if np.abs(trf_dict['angle_deg']) > np.finfo(np.float32).eps:
            route_dict['rotation'] = R.from_rotvec(
                np.deg2rad(trf_dict['angle_deg']) * np.array(trf_dict['axis'])).as_matrix().tolist()
        else:  # No rotation at all
            route_dict['rotation'] = [[1,0,0], [0,1,0], [0,0,1]]

    # Write route file
    with open(filled_route_file_path, 'w') as f:
        safe_json_dump(route_dict, f)
    logger.info(f'Route file {filled_route_file_path} generated successfully.')

    
    # Export visualizations
    returncode = export_route_visualization(filled_route_file_path, logger)
    returncode = export_sfm_data_visualization(input_sfm_data_path, logger)

    return returncode



def copy_2_tower(loc_prj_dir, tower_src_dir, prj_name) -> int:
    '''
    Find and copy (content, not symlinks) following things:
        1) Data folders that has project name in its name
        1) Loc folders that has project name in its name
        1) Any files that starts with the project name
    If a thing already exists in tower src dir, then skip it. (??????)
    '''
    
    data_match_string = 'data_' + prj_name
    loc_match_string = 'loc__' + prj_name

    for item in os.listdir(loc_prj_dir):
        src = join(loc_prj_dir, item)

        if os.path.isdir(src) and (item[:len(data_match_string)]==data_match_string):
            dst = safely_create_dir(tower_src_dir, item)

            for item in os.listdir(src):
                if (item == 'img_prj') or (item == 'matches'):
                    current_dir = os.getcwd()
                    # go to src dir
                    os.chdir(src)
                    #symlink_dir = join(src, item)
                    #tar_filepath = join(dst, item + '.tar.gz')
                    tar_file = item+'.tar.gz'
                    with tarfile.open(tar_file, mode='w:gz', dereference=True) as tar:
                        print(f'Adding {item} to tar.gz file {tar_file}...')
                        tar.add(item)
                    # copy tar file
                    tar_dst_path = join(dst, tar_file)
                    print(f'Copying {tar_file} from {src} to {tar_dst_path}...')
                    shutil.copy(tar_file, tar_dst_path)
                    # go back to current_dir
                    os.chdir(current_dir)
                else:
                    try:
                        print(f'Copying {item} from {src} to {dst}...')
                        shutil.copy(join(src, item), join(dst, item))  # Except for img and matches, everything is file not folder
                    except:
                        print(f'Unexpected folder {item} is in {src}. Ignored.')

        elif os.path.isdir(src) and (item[:len(loc_match_string)]==loc_match_string):
            dst = safely_create_dir(tower_src_dir, item)

            for item in os.listdir(src):
                # Everything in loc dir should be file not folder (matches & query should be deleted)
                try:
                    print(f'Copying {item} from {src} to {dst}...')
                    shutil.copy(join(src, item), join(dst, item))  # Except for img and matches, everything is file not folder
                except:
                    print(f'Unexpected folder {item} is in {src}. Ignored.')

        elif (not os.path.isdir(src)) and (item[:len(prj_name)]==prj_name) and (item[-5:]=='.yaml'):
            try:
                print(f'Copying {src} to {tower_src_dir}...')
                shutil.copy(src, tower_src_dir)
            except Exception as e:
                raise Exception(e)

    return 0

def copy_from_tower(loc_prj_dir, tower_src_dir) -> int:
    '''
    Find and copy following things:
        1) Data folders that contains string of project name
           i.e.: data_{project_name}_000 or data_{project_name}_001...etc.
        2) Extract tar files
    Args:
    loc_prj_dir: path of local workspace
    tower_src_dir: path of tower repo directory
    '''
    
    data_match_string = 'data_'

    for item_in_tower in os.listdir(tower_src_dir):
        src = join(tower_src_dir, item_in_tower)

        if os.path.isdir(src) and (item_in_tower[:len(data_match_string)]==data_match_string):
            dst = safely_create_dir(loc_prj_dir, item_in_tower)
            for item in os.listdir(src):
                copy_item_to_tower(src, dst, item)
    return 0

def copy_item_to_tower(src, dst, item):
    '''
    Copy items in data_* folders to tower repo
    Args:
    src: str, local path
    dst: str, tower repo path
    item: str, name of item to be copied
    '''
    try:
        print(f'Copying {item} from {src} to {dst}...')
        shutil.copy(join(src, item), join(dst, item))  # Except for img and matches, everything is file not folder
    except:
        print(f'Unexpected folder {item} is in {src}. Ignored.')

    if (item == TAR_FILE_IMG_PRJ) or (item == TAR_FILE_MATCHES):
        current_dir = os.getcwd()
        # go to dst dir and extract tar file
        os.chdir(dst)
        tar_file = item
        with tarfile.open(tar_file, mode='r:gz', dereference=True) as tar:
            print(f'Extracting {item} from tar.gz file {tar_file}...')
            tar.extractall('.')
        # go back to current_dir
        os.chdir(current_dir)


