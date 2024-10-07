import json
import os
import numpy as np
import subprocess
import traceback
import trimesh
import re
import copy

from datetime import datetime
from numpy import concatenate as concat
from os.path import join
from pytz import timezone, utc
from scipy.spatial.transform import Rotation as R
from point_filter.poly_fit_filter import PolyFitFilter

def get_current_time_str() -> str:
    utc_dt_tz = utc.localize(datetime.utcnow()).astimezone(timezone('Asia/Taipei'))
    return '-'.join([f'{n:02d}' for n in utc_dt_tz.timetuple()][:6])
   
    
def safely_create_dir(parent_dir: str, sub_dir: str) -> str:
    ''' Make directory and return path as string. '''

    path = join(parent_dir, sub_dir)
    if not os.path.exists(path): os.makedirs(path)
    return path

def run_cmd_w_retry(logger, cmd, cmd_name, retry_max = 5):
    retry = 0
    while retry < retry_max:
        proc = run_log_cmd(logger, cmd)
        if proc.returncode !=0:
            retry += 1
            logger.info(f'Retry {cmd_name}: {retry}')
            if (retry >= retry_max):
                logger.error('Stop running futher commands.')
                return proc.returncode
        else:
            logger.info(f'{cmd_name} ended with return code {proc.returncode}.')
            return proc.returncode

def run_log_cmd(logger, cmd):
    ''' Run subprocess and log to logger until finish. '''

    logger.info('========== Started running command. ==========')
    # Run asynchronously
    logger.info(f'Command: {" ".join(cmd)}')
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        logger.info('Command successfully started, now running...')
    # except Exception as e:
    except Exception:
        e = traceback.format_exc()
        logger.error(f'Command error occured: {e}')

    # Log in while loop
    while True:
        output = proc.stdout.readline()
        if output == b'' and proc.poll() is not None:
            break
        elif output:
            try:
                # If use ascii: can't decode path including traditional chinese......
                logger.info(output.strip().decode('utf-8'))
            except Exception as e:
                logger.error(f'Log error occured: {e}')
    logger.info('========== Ended running command. ==========')
    return proc


class TransData(object):
    '''
    The transformation data here is defined to transform the mesh (model), not the camera orientation in sfm data json! 
    I.e. If the camera orientation (key 'rotation' in sfm data json) should be transformed, the inverse of transformation
    here should be applied in back of the 'rotation' key.    
    '''
    
    def __init__(self):
        self._trf_list = None
        self._concat_trf_mat = None
    
    @property
    def concat_trf_mat(self):
        return self._concat_trf_mat
    
    @property
    def trf_list(self):
        return self._trf_list
    
    def set_from_json(self, trans_json_path: str):
        ''' Fill empty object with data read from json. '''
        
        if self._trf_list or self._concat_trf_mat:
            raise RuntimeError('Overwriting the existing content of TransData object is forbidden!')
        with open(trans_json_path) as f:
            json_data = json.load(f)
            self._concat_trf_mat = json_data['concatenated']
            self._trf_list = json_data['list']
        return self
            
    def set_from_input(self, trf_mat: list, note: str = None):
        ''' Fill empty object with input data and note. '''
        
        if self._trf_list or self._concat_trf_mat:
            raise RuntimeError('Overwriting the existing content of TransData object is forbidden!')
        self._trf_list = [{'matrix': trf_mat,'timestamp': get_current_time_str(), 'note': note}]
        self._concat_trf_mat = trf_mat
        return self
        
    def append_trf(self, trf_mat: list, note: str = None):
        ''' Append non-empty object with input data, then update the concatenated transformation matrix.'''
        
        # Append to list
        if (not self._trf_list) and (not self._concat_trf_mat):
            raise RuntimeError('TransData object is empty. Please use from_input or from_json to initialize.')
        self._trf_list.append({'matrix': trf_mat,'timestamp': get_current_time_str(), 'note': note})
        # Update concatenated transformation matrix
        self._concat_trf_mat = np.matmul(np.array(trf_mat), np.array(self._concat_trf_mat)).tolist()
        return
        
    def write2json(self, output_json_path: str):
        ''' Write both concatenated transformation and the list of all transformations to a json file. '''
        
        output_dict = {'concatenated': self._concat_trf_mat, 'list': self._trf_list}
        with open(output_json_path, 'w') as f:  # TODO (hkazami): Is that clean? Can it overwrite the whole file?
            json.dump(output_dict, f)
        return
    

def compose_trf_mat(scale: float, angle_deg: float, axis: list, offset: list) -> list:
    ''' Make transformation matrix with the order: rotation -> scale -> translation.'''

    # Rotation
    r = R.from_rotvec(np.deg2rad(angle_deg) * np.array(axis)).as_matrix()
    rotation_4x4 = concat((concat((r, [[0,0,0]]), axis=0), [[0],[0],[0],[1]]), axis=1)
    # Scale
    s = scale
    scale_4x4 = np.array([[s,0,0,0],[0,s,0,0],[0,0,s,0],[0,0,0,1]])
    # Translation
    o = np.array(offset)[:,np.newaxis]
    offset_4x4 = concat((concat((np.identity(3), o), axis=1), [[0,0,0,1]]), axis=0)

    # Concatenate
    matrix_4x4 = np.matmul(offset_4x4, np.matmul(scale_4x4, rotation_4x4))

    return matrix_4x4.tolist()


def decompose_trf_mat(trf_mat: list) -> dict:
    ''' From 4x4 transformation matrix, extract: scale, axis, angle in deg, and offset. '''

    # Offset
    offset = np.array(trf_mat)[0:3,3].tolist()
    
    # Scale
    trf_3x3 = np.array(trf_mat)[0:3, 0:3]
    det = np.linalg.det(trf_3x3)
    scale = np.cbrt(det)
    
    # Axis & angle
    rot_3x3 = (1 / scale) * trf_3x3
    rot_vec = R.from_matrix(rot_3x3).as_rotvec()
    angle_deg = np.rad2deg(np.linalg.norm(rot_vec))
    axis = rot_vec / np.deg2rad(angle_deg)
    
    return {'scale': scale, 'angle_deg': angle_deg, 'axis': axis, 'offset': offset}
    

def transform_sfm_data(input_sfm_data_path: str, output_sfm_data_path: str, trf_mat: list, logger):
    '''
    Given translation vector & rotation matrix which rotates the 3D model / camera poses & camera body frames,
    read original sfm data, rotate every extrinsics and save as new json.

    Steps of transformation: 1. Rotate 2. scale 3. translate
    Mode can only be "model_cor" or "loc", they have different equations for rotation (see CPL doc).
    If this function should be used for new cases, check the doc first to clarify which mode it should be.
    '''
    def transform_per_view(trf_mat_np: np.ndarray, view: dict) -> np.ndarray:
        def transform_rot(rot, rot_mat):
            '''
            return (np.matmul(np.array(view['rotation']), rotation.transpose())  # Original, should be for localization
                    if (mode == 'loc') else
                    np.matmul(rotation.transpose(), np.array(view['rotation'])))  # Should be for level correction & initial pose zeroing (20201014: almost certain this is wrong...)
            '''
            return np.matmul(rot, rot_mat.transpose())
        def transform_pos(pos, trf_mat):
            pos_np = np.append(np.array(pos), 1)
            return np.matmul(trf_mat, pos_np)[:3]

        rot_mat_np = trf_mat_np[:3,:3] / np.cbrt(np.linalg.det(trf_mat_np[:3,:3]))
        rot_transformed = transform_rot(np.array(view['rotation']), rot_mat_np).tolist()
        pos_transformed = transform_pos(np.array(view['center']), trf_mat_np).tolist()
        footprint_transformed = (transform_pos(np.array(view['footprint']), trf_mat_np).tolist()
                                 if 'footprint' in view else None)
        return pos_transformed, rot_transformed, footprint_transformed

    logger.info('Start generating transformed sfm data file...')
    with open(input_sfm_data_path, 'r') as f:
        camera_dict = json.load(f)
    trf_mat_np = np.array(trf_mat)

    # Correct center and rotation of views 1 by 1
    for i_view in range(len(camera_dict['extrinsics'])):
        pos_trans, rot_trans, footprint_trans = transform_per_view(trf_mat_np, camera_dict['extrinsics'][i_view]['value'])
        camera_dict['extrinsics'][i_view]['value']['center'] = pos_trans
        camera_dict['extrinsics'][i_view]['value']['rotation'] = rot_trans
        if footprint_trans:
            camera_dict['extrinsics'][i_view]['value']['footprint'] = footprint_trans

    # Write new sfm data file
    with open(output_sfm_data_path, 'w') as file:
        json.dump(camera_dict, file)
    logger.info(f'Generation of transformed sfm data file {output_sfm_data_path} done.')

    
def extend_sfm_data_with_footprints(sfm_data_path: str, mesh_path: str, logger) -> dict:
    """ Use trimesh to find the intersection of a ray (bottom of the camera in our case) with a mesh. """
    
    with open(sfm_data_path, 'r') as f:
        camera_dict = json.load(f)
    logger.info(f'Camera poses from {sfm_data_path} loaded. Start calculating footprints...')

    # Iterate through all views to calculate footprint
    center_list = []
    direction_list = []
    for extrinsic in camera_dict['extrinsics']:
        center_list.append(extrinsic['value']['center'])
        direction_list.append(np.array(extrinsic['value']['rotation'])[1, :])
    center_list = np.array(center_list)
    direction = np.mean(direction_list, axis=0)

    mesh = trimesh.load(mesh_path)
    locations, index_ray_list, _ = mesh.ray.intersects_location(
                                ray_origins=center_list,
                                ray_directions=[direction]*len(center_list),
                                multiple_hits=False)

    footprint_list = [None] * len(center_list)
    if len(locations) > 0:
        distance_list= np.linalg.norm(np.array(center_list)[index_ray_list] - locations, axis=1)
        min_dist = np.percentile(distance_list, 30, interpolation='nearest')
        median_dist = np.percentile(distance_list, 50, interpolation='nearest')
        max_dist = np.percentile(distance_list, 70, interpolation='nearest')

        for i, location, distance in zip(index_ray_list, locations, distance_list):
            if distance < min_dist:
                location = center_list[i] + min_dist * direction
            elif distance > max_dist:
                location = center_list[i] + max_dist * direction
            footprint_list[i] = location

        for i in range(len(footprint_list)):
            if footprint_list[i] is None:
                footprint_list[i] = center_list[i] + median_dist * direction
    else:
        logger.warn('Bad Model Cause Bad Footprint')
        for i in range(len(footprint_list)):
            if footprint_list[i] is None:
                footprint_list[i] = center_list[i]

    for extrinsic, footprint in zip(camera_dict['extrinsics'], footprint_list):
        extrinsic['value']['footprint'] = footprint.tolist()

    logger.info('Footprints calculation done.')
    return camera_dict


def extend_sfm_data_with_stepsize_anchor(sfm_data_path: str, logger) -> dict:
    '''
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

    in this function another key "footprint" will be added in dict in extrinsics
    so it will look like:
    {
        "key": int, corresponds to the key defined in views
        "value": 
        {
            "rotation": [3x3] list, rotation matrix of camera pose
            "center": [3] list, camera position of camera pose
            "footprint": [3] list, camera footprint of camera pose \
                                   generated with stepsize 
        }
 
    }


    Arguments: 
    str: filepaht of sfm_data model
    logger: Logging object
    '''
    with open(sfm_data_path, 'r') as f:
        camera_dict = json.load(f)
    logger.info(f'Camera poses from {sfm_data_path} loaded. Start calculating footprints...')

    # Iterate through all views to calculate footprint
    center_list = []
    direction_list = []
    
    for extrinsic in sorted(camera_dict['extrinsics'], key=lambda d: d['key']):
        center_list.append(extrinsic['value']['center'])
        direction_list.append(np.array(extrinsic['value']['rotation'])[1, :])
    center_list = np.array(center_list)
    center_step_length_list = calc_step_length(center_list)
    
    center_step_length_list = [step*4 for step in center_step_length_list]

    footprint_list = [None] * len(center_list)
    for idx, (direction, step_length) in enumerate(zip(direction_list, center_step_length_list)):
        footprint_list[idx] = center_list[idx] +step_length*direction
    for extrinsic, footprint in zip(camera_dict['extrinsics'], footprint_list):
        extrinsic['value']['footprint'] = footprint.tolist()
    logger.info('Footprints calculation done.')
    return camera_dict

def calc_step_length(cam_pose_array):
    '''
    calculate step length by finding difference between camera pose 
    since difference has one row less than original camera pose array,
    so we repeat the last row to make dimension consistant

    Arguments:
    cam_pose_array: numpy array [N, 3] N is the number of camera poses
    Returns:
    step_lenght: [N] list 
    '''
    cam_pose_diff = cam_pose_array[1:,:] - cam_pose_array[0:-1,:]
    last_row = np.array([cam_pose_diff[-1,:]])
    extended_cam_pose_diff = np.append(cam_pose_diff, last_row,0)
    step_length = np.linalg.norm(extended_cam_pose_diff, axis=1)
    return step_length.tolist()

def reorder_sfm_data(input_json_path, output_json_path, logger):
    '''
    create a sfm_data model with image name. If a view does not has corresponding 
    extrinsics within sfm_data model from input_json_path, this view will not be 
    included in the newly created model. 
    the sfm_data model will be written to output_json_path
    
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

    Arguments: 
    intput_json_path: str, json filepath of sfm_data model that to be ordered.
    output_json_path: str, json filepath of the new ordered sfm_data model.
    '''
    # reorder with image file name and also remove view that do not 
    # have extrinsics
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    # === Make a image file numbering list with original order ===
    # Image name: imgNr_lenseNr_hash.xxx
    img_name_str = [
        re.split(r'[.]', view['value']['ptr_wrapper']['data']['filename'])[0]
        for view in data['views']]
    # Sort according to the above list and extract the new order
    seq = [element[0] for element in sorted(enumerate(img_name_str), key=lambda x:x[1])]

    # === Rearrange the views and extrincs according to the new order ===
    # Make a extrinsics dict with ['key'] as key
    extrinsics_dict = {}
    for extr in data['extrinsics']:
        extrinsics_dict[extr['key']] = extr['value']
    # Place the view / extr pair into a new dict w/ the new order, if no pair then skip the view
    reordered_data = {'views':[], 'extrinsics':[]}
    for new_key, i in enumerate(seq):
        view_orig = data['views'][i]
        view = copy.deepcopy(data['views'][i])
        #set to new key
        view['key'] = new_key
        try:
            reordered_data['extrinsics'].append(
                {'key': new_key, 'value': extrinsics_dict[view_orig['key']]})
            reordered_data['views'].append(view)
        except Exception as e:
            logger.warning(f'Failed to find view / extr pair using key nr.{view["key"]}. Skip it. Original error message: {e}')
    
    # Put reordered things back and dump 
    data['views'] = reordered_data['views']
    data['extrinsics'] = reordered_data['extrinsics']

    json.dump(data, open(output_json_path, 'w'), indent=4)

def filter_sfm_data(input_sfm_json, output_sfm_json, logger):
    """
    use polyFit to fit the camera routes in input_sfm_json and remove the outliers
    export the filter sfm_data in output_sfm_json

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
    input_sfm_json: str, path of the sfm_data file before filtered
    output_sfm_json: str, path of the filtered sfm_data file
    logger: logging obj
    """
    with open(input_sfm_json, 'r') as f:
        sfm_data = json.load(f)
    # Iterate through all views to calculate footprint
    center_list = []
    rotation_list = [] 
    for extrinsic in sorted(sfm_data['extrinsics'], key=lambda d: d['key']):
        center_list.append(extrinsic['value']['center'])
        rotation_list.append(extrinsic['value']['rotation'])
    valid_center_idx = rm_outlier(center_list)
    valid_rot_idx = filter_rotation(np.array(rotation_list)[valid_center_idx])
    valid_data_idx = np.array(valid_center_idx)[valid_rot_idx]
    data_xyz = np.array(center_list)[valid_data_idx]
    # this setting here represent linear fit, which is a pretty strong assumption
    poly_degree  = 5
    max_iter = 10
    polyfit = PolyFitFilter(poly_degree, max_iter) 
    valid_polyfit_idx = polyfit.filter(data_xyz, num_sigma = 3)
    valid_idx = np.array(valid_data_idx)[valid_polyfit_idx]
    # Make a extrinsics dict with ['key'] as key
    extrinsics_dict = {}
    for extr in sfm_data['extrinsics']:
        extrinsics_dict[extr['key']] = extr['value']
 
    # create new dict with only valid index 
    filtered_data = {'views':[], 'extrinsics':[]}
    for new_key, idx in enumerate(valid_idx):
        view_orig = sfm_data['views'][idx]
        view = copy.deepcopy(sfm_data['views'][idx])
        #set to new key
        view['key'] = new_key
        try:
            filtered_data['extrinsics'].append(
                {'key': new_key, 'value': extrinsics_dict[view_orig['key']]})
            filtered_data['views'].append(view)
        except Exception as e:
            logger.warning(f'Failed to find view / extr pair using key nr.{view_orig["key"]}. Skip it. Original error message: {e}')
    # Put filtered things back and dump 
    sfm_data['views'] = filtered_data['views']
    sfm_data['extrinsics'] = filtered_data['extrinsics']

    json.dump(sfm_data, open(output_sfm_json, 'w'), indent=4)

   
 
def reduce_views(input_json_ref, input_json_match, output_json, logger):
    '''
    remove the views in sfm_data model in input_json_match that is not existing 
    in sfm_data model in input_json_ref, and write to output_json
    
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

    
    
    Arguments:
    input_json_ref: str, json filepath of reference sfm_data model
    input_json_match: str, json filepath of sfm_data model to match reference
    output_jsonL str, json filepath of output sfm_data model
    logger: Logging object
    '''
    with open(input_json_ref, 'r') as f:
        data_ref = json.load(f)
    with open(input_json_match, 'r') as f:
        data_match = json.load(f)
    # Image name: imgNr_lenseNr_hash.xxx
    img_name_ref = [
        re.split(r'[.]', view['value']['ptr_wrapper']['data']['filename'])[0]
        for view in data_ref['views']]
    
    # Make a extrinsics dict with ['key'] as key
    extrinsics_dict = {}
    for extr in data_match['extrinsics']:
        extrinsics_dict[extr['key']] = extr['value']
 
    reduced_data = {'views':[], 'extrinsics':[]}
    for view in data_match['views']:
        filename = view['value']['ptr_wrapper']['data']['filename']
        filename_base, _ = os.path.splitext(filename) 
        if filename_base in img_name_ref:
            print(f'view in image list')
            try:
                reduced_data['extrinsics'].append(
                {'key': view['key'], 'value': extrinsics_dict[view['key']]})
                reduced_data['views'].append(view)
            except Exception as e:
                logger.warning(f'Failed to find view / extr pair using key nr.{view["key"]}. Skip it. Original error message: {e}')
            
    # Put reordered things back and dump 
    data_match['views'] = reduced_data['views']
    data_match['extrinsics'] = reduced_data['extrinsics']

    json.dump(data_match, open(output_json, 'w'), indent=4)

def rm_outlier(center_list):
    """
    remove outliers if the distantace of points is too far
    from mean position
    Args:
    center_list, list of [1,3] , positions of 3D points
    Returns:
    valid_idx: list of int
    """

    center = np.array(center_list)
    center_mean = np.mean(center, axis=0)
    center_diff = []
    for c in center:
        diff = np.linalg.norm(c-center_mean)
        center_diff.append(diff)
    center_diff_mean = np.mean(center_diff)
    center_diff_std = np.std(center_diff)
    thr = center_diff_mean+center_diff_std
    valid_idx = []
    for idx, cd in enumerate(center_diff):
        if cd > thr:
            pass
        else:
            valid_idx.append(idx)
    return valid_idx

def filter_rotation(rotation_list, thr=1.0):
    """
    filter camera orientation if the downward direction is 
    away from mean value by thr(degree)
    Args:
    rotation_list: list of [3X3], camera orientation
    Returns:
    valid_idx: list of int
    """
    direction_list = []
    for rot_matrix in rotation_list:
        direction_list.append(np.array(rot_matrix)[1, :])
        #rot_obj = R.from_matrix(rot_matrix)
        #rot_vec = rot_obj.as_rotvec()
        #rot_axis = rot_vec / np.linalg.norm(rot_vec)
        #rot_angle_deg = np.rad2deg(np.linalg.norm(rot_vec))
    dir_arr = np.array(direction_list)
    mean_dir = np.mean(dir_arr, axis=0)
    deg_list = []
    for direction in dir_arr:
        deg = np.arccos(np.dot(direction, mean_dir))
        deg_list.append(deg)
    valid_idx = []
    for idx, deg in enumerate(deg_list):
        if deg < thr:
            valid_idx.append(idx)
    return valid_idx

