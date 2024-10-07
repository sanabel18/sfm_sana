import json
import os
from fnmatch import fnmatch
import numpy as np

from os.path import join
from scipy.spatial.transform import Rotation as R
from utils.genutil import run_log_cmd
from utils.meshdata import MeshData
from xml.etree import cElementTree as ET


filters_dir = join(os.path.dirname(__file__), 'filters')
meshlab_cli = ['xvfb-run', '-a', '-s', '-screen 0 800x600x24', '/opt/meshlab/distrib/meshlabserver']

    
class ProcMlx(object):
    def __init__(self):
        pass

    @staticmethod
    def _print_bounding_box_inequality(box: dict) -> str:
        ''' Return the bounding box inequality string to be written in mlx file. '''
        s_list = []
        pts = {'x': ['x0','x1','x2'], 'y':['y0','y1','y2'], 'z': ['z0','z1','z2']}
        for axis, pts_axis_list in pts.items():
            for pt_axis in pts_axis_list:
                s = f'{pt_axis} > {box[axis]["min"]} && {pt_axis} < {box[axis]["max"]}'
                s_list.append(s)
        s = ' && '.join(s_list)
        return s   

    @staticmethod
    def _write_cfs_mlx(condition: str, filepath: str):
        ''' Generate conditional face selection mlx filter script. '''

        root = ET.Element('FilterScript')
        tr = ET.SubElement(root, 'filter', {'name': 'Conditional Face Selection'})
        param = ET.SubElement(tr, 'Param')
        param.set('name', 'condSelect')
        param.set('value' , condition)
        param.set('description' , 'boolean function') 
        param.set('type', 'RichString') 
        tree = ET.ElementTree(root)
        with open(filepath, 'wb') as f:
            f.write('<!DOCTYPE FilterScript>'.encode('utf8'))
            tree.write(f,'utf-8')        

    @classmethod
    def write_seg_mlx(cls, data, seg_label: int, normal_seed: np.ndarray, bounding_box_buffer: float, min_seg_size: int, output_dir: str, logger):
        '''
        Given segment label, find faces with that label and output two types of mlx:
            - bounding box vertices
            - face id's
        Also the metadata of the 2 mlx's will be saved as json.
        '''
 
        idx_array = data.seg_labels[data.seg_labels == seg_label].index.values
        if (idx_array.size > min_seg_size):
            metadata = data.get_bounding_box(idx_array, bounding_box_buffer)
            metadata['group'] = seg_label
            metadata['size'] = idx_array.size
            metadata['normal_seed'] = normal_seed.tolist()  # Type casting to list to be able to write in json

            s_list = []
            for idx in idx_array:
                s_list.append('(fi == {})'.format(idx))
            face_ids = ' || '.join(s_list)
            file_name_face = f'group_{metadata["group"]}_size_{metadata["size"]}_face.mlx'
            cls._write_cfs_mlx(face_ids, join(output_dir, file_name_face))
            logger.info(f'Finish executing write {file_name_face}.')

            box_list = cls._print_bounding_box_inequality(metadata)
            file_name_box = f'group_{metadata["group"]}_size_{metadata["size"]}_box.mlx'
            cls._write_cfs_mlx(box_list, join(output_dir, file_name_box) + '.mlx')
            logger.info(f'Finish executing write {file_name_box}.')

            file_name_json = f'group_{metadata["group"]}_size_{metadata["size"]}.json'
            with open(join(output_dir, file_name_json), 'w') as f:
                json.dump(metadata, f)            
            logger.info(f'Finish executing write {file_name_json}.')
            
        else:
            logger.warning(f'Segment size is too small, no mlx is written: {idx_array.size} <= {min_seg_size}')

    @staticmethod
    def write_trans_mlx(output_mlx_path: str, axis: list, angle: float, scale: float, offset: list):
        ''' Given rotation axis and angle (deg), write rotation mlx file. '''
        
        # Definitions of file paths
        template_path = join(filters_dir, 'transform.mlx')
        
        root = ET.parse(template_path).getroot()
        # Find rotation axis and angle fields and replace with input arguments
        for child in root:
            if (child.attrib['name'] == 'Transform: Rotate'):
                for grand_child in child:
                    # Type cast to str, since ET cannot serialize float64......
                    if (grand_child.attrib['name'] == 'customAxis'):
                        grand_child.attrib['x'] = str(axis[0])
                        grand_child.attrib['y'] = str(axis[1])
                        grand_child.attrib['z'] = str(axis[2])
                    if (grand_child.attrib['description'] == 'Rotation Angle'):
                        grand_child.attrib['value'] = str(angle)
            if (child.attrib['name'] == 'Transform: Scale, Normalize'):
                for grand_child in child:
                    # Type cast to str, since ET cannot serialize float64......
                    if ((grand_child.attrib['name'] == "axisX") or
                        (grand_child.attrib['name'] == "axisY") or
                        (grand_child.attrib['name'] == "axisZ")):
                        grand_child.attrib['value'] = str(scale)
            if (child.attrib['name'] == 'Transform: Translate, Center, set Origin'):
                for grand_child in child:
                    # Type cast to str, since ET cannot serialize float64......
                    if (grand_child.attrib['name'] == "axisX"):
                        grand_child.attrib['value'] = str(offset[0])
                    if (grand_child.attrib['name'] == "axisY"):
                        grand_child.attrib['value'] = str(offset[1])
                    if (grand_child.attrib['name'] == "axisZ"):
                        grand_child.attrib['value'] = str(offset[2])
        
        # Write new mlx file
        new_tree = ET.ElementTree(root)
        with open(output_mlx_path, 'wb') as f:
            f.write('<!DOCTYPE FilterScript>'.encode('utf8'))
            f.write(b'\n')
            new_tree.write(f,'utf-8')
    
    @classmethod
    def write_del_ceil_mlx(cls, output_dir: str, y_info: dict, r_floor_cut: float) -> str:
        ''' Create the ceiling deleting mlx file (with very simple cut). '''
        
        # Definitions of file paths
        select_file = join(output_dir, 'select.mlx')
        filter_file = join(filters_dir, 'delete_selected.mlx')
        output_file = join(output_dir, 'select_and_delete.mlx')
        
        # Write ceiling selection mlx
        y_cut = y_info['min'] + r_floor_cut * y_info['dy']
        s = f'y0 < {y_cut} || y1 < {y_cut} || y2 < {y_cut}'
        cls._write_cfs_mlx(s, select_file)
        
        # Combine selection and deletion mlx
        cls.merge_selection_action_mlx(select_file, filter_file, output_file)
        try:
            os.remove(select_file)
        except:
            pass
        return output_file
        
    @staticmethod
    def merge_selection_action_mlx(select_mlx_path, filter_mlx_path, output_mlx_path):
        ''' Combine 2 mlx into 1. Normally the 1st one is to select mesh and the 2nd one is to apply filter actions. '''

        tr_filter = ET.parse(filter_mlx_path)
        tr_select = ET.parse(select_mlx_path)
        root_filter = tr_filter.getroot()
        root_select = tr_select.getroot()

        # Insert selection in front of action
        for child in root_select:
            root_filter.insert(0,child)

        # Write a new merged xml out
        tree_combined = ET.ElementTree(root_filter)
        with open(output_mlx_path, 'wb') as f:
            f.write('<!DOCTYPE FilterScript>'.encode('utf8'))
            tree_combined.write(f,'utf-8')

    @classmethod
    def merge_aligned_mlx_from_segments(cls, normal_test_thr: float, seg_dir: str, logger,
                                        align_axis: np.ndarray = np.array([0,1,0])) -> list:
        '''
        From all groups / segments:
            - pick the groups whose normal aligns with given axis,
            - merge them as another mlx file, and
            - return id of all selected faces as a list.
        '''
        
        group_size_list = []
        normal_dot_list = []
        s_list = []
        merged_mlx_path = join(seg_dir, 'aligned_groups.mlx')

        # Read metadata from json files
        logger.info('Reading group metadata from group json files...')
        logger.info(f'Axis to be aligned with: {align_axis}')
        for json_file in os.listdir(seg_dir):
            if fnmatch(json_file, "group*size*.json"):
                with open(join(seg_dir, json_file), 'r') as f:
                    json_dict = json.load(f)
                normal = np.array(json_dict['normal_seed'])
                normal_dot_list.append(np.dot(normal, align_axis))
                group_size_list.append((json_dict['group'], json_dict['size']))
                logger.info(f'Group {json_dict["group"]} (size {json_dict["size"]}): ' +
                            f'normal = {normal}, alignment = {normal_dot_list[-1]}')
        normal_dot_array = np.array(normal_dot_list)

        # Choose not to use absolute value in order to pick only the floor not the ceiling, but it's just experience and can break anytime......
        # TODO (hkazami): Find a better way to pick the face we want and discard those which are parallel but we don't want (e.g. call floor cutting function)
        is_aligned = -normal_dot_array > normal_test_thr
        # Use the truth table of alignment to get the indexing and pick floor group and size
        aligned_index_list = np.where(is_aligned)[0].tolist()
        floor_group_size_list = [group_size_list[idx] for idx in aligned_index_list]
        logger.info(f'List of selected floor groups (group id, size): {floor_group_size_list}')

        # Read selected group mlx files and write all in another mlx
        for group, size in floor_group_size_list:
            mlx_file = f'group_{group}_size_{size}_face.mlx'
            tree = ET.parse(join(seg_dir, mlx_file))
            root = tree.getroot()
            for child in root:
                for grand_child in child:
                    s = grand_child.attrib['value']
                    s_list.append(s)
        selection_str = ' || '.join(s_list)  # This the the string of all faces
        cls._write_cfs_mlx(selection_str, merged_mlx_path)
        logger.info(f'Merged group mlx is written to {merged_mlx_path}.')
        
        # Parse the string of all faces to get face list
        fid_str_list = selection_str.split("||")
        fid_list = []
        for fid_str in fid_str_list:
            start = fid_str.find('== ')
            end = fid_str.find(')')
            fid = fid_str[start+3:end]
            fid_list.append(int(fid))
        
        return fid_list, merged_mlx_path

            
class GenMesh(object):
    ''' Container of all kinds of mesh generating functions. '''
    def __init__(self):
        pass
    
    @staticmethod
    def _gen_convexhull(input_mesh_path: str, output_dir: str, logger):
        '''
        Generate convex hull ply which closes input mesh.
        Currently (2020.09.01) called only by other utility functions.
        '''

        output_mesh_path = join(output_dir, 'convex_hull.ply')
        
        logger.info('Start generating convex hull information...')
        proc = run_log_cmd(logger, meshlab_cli + [
            '-i', input_mesh_path,
            '-o', output_mesh_path,
            '-s', join(filters_dir, 'convex_hull.mlx')])

        mesh_data = MeshData(output_mesh_path)
        x = mesh_data.points['x'].values
        y = mesh_data.points['y'].values
        z = mesh_data.points['z'].values
        dx = np.max(x) - np.min(x)
        dy = np.max(y) - np.min(y)
        dz = np.max(z) - np.min(z)
        v = np.array([dx, dy, dz])

        logger.info(f'Convex hull info generation ended with return code {proc.returncode}.')        
        return proc.returncode, output_mesh_path, np.linalg.norm(v), {'min': np.min(y), 'dy': dy}

    @classmethod
    def del_isolated_mesh(cls, input_mesh_path: str, output_dir: str, output_mesh_name: str, logger, min_mesh_size: int = 500):
        ''' 
        Equip input mesh with curvature information and save as new file:
            1. Calculate convex hull that closes the mesh, in order to get boundaries
            2. Use the boundary info to generate curvature-calculating mlx from template
            3. Calculate curvature, brutally reduce mesh size, then save everything as new ply
        '''
        # Definitions of file paths
        clean_mesh_path = join(output_dir, output_mesh_name)
        clean_filter_template_path = join(filters_dir, 'delete_isolated_mesh_template.mlx')
        clean_filter_path = join(output_dir, 'delete_isolated_mesh.mlx')

        # Read template mlx and insert own numbers
        root_filter = ET.parse(clean_filter_template_path).getroot()
        for child in root_filter:
            if (child.attrib["name"] == "Remove Isolated pieces (wrt Face Num.)"):
                for grand_child in child:
                    if (grand_child.attrib["name"] == "MinComponentSize"):
                        grand_child.attrib["value"] = str(min_mesh_size)

        # Write new mlx file
        output_et = ET.ElementTree(root_filter)
        with open(clean_filter_path, 'wb') as f:
            f.write('<!DOCTYPE FilterScript>'.encode('utf8'))
            f.write(b'\n')
            output_et.write(f,'utf-8')    

        # Run meshlab to generate reduced mesh with curvature
        logger.info(f'Start deleting isolated mesh smaller than {min_mesh_size}...')
        proc = run_log_cmd(logger, meshlab_cli + [
            '-i', input_mesh_path,
            '-o', clean_mesh_path,
            '-m', 'vc', 'vn',
            '-s', clean_filter_path])
        logger.info(f'Mesh cleaning ended with return code {proc.returncode}.')
        return proc.returncode, clean_mesh_path

    @classmethod
    def gen_curv_mesh(cls, input_mesh_path: str, output_dir: str, output_mesh_name: str, logger):
        ''' 
        Equip input mesh with curvature information and save as new file:
            1. Calculate convex hull that closes the mesh, in order to get boundaries
            2. Use the boundary info to generate curvature-calculating mlx from template
            3. Calculate curvature, brutally reduce mesh size, then save everything as new ply
        '''
        # Definitions of file paths
        curv_mesh_path = join(output_dir, output_mesh_name)
        curv_filter_template_path = join(filters_dir, 'gen_curv_template.mlx')  # gen_cur_v2.mlx
        curv_filter_path = join(output_dir, 'gen_curv_mesh.mlx')

        # Calcuate threshold (boundaries)
        returncode, _, threshold, _ = cls._gen_convexhull(input_mesh_path, output_dir, logger)
        if returncode != 0: 
            logger.error('Stop running further commands.')
            return returncode

        # Read template mlx and insert own numbers
        root_filter = ET.parse(curv_filter_template_path).getroot()
        for child in root_filter:
            if (child.attrib["name"] == "Simplification: Clustering Decimation"):
                for grand_child in child:
                    if (grand_child.attrib["name"] == "Threshold"):
                        grand_child.attrib["value"] = str(threshold/100.)
                        grand_child.attrib["max"] = str(threshold)

        # Write new mlx file
        output_et = ET.ElementTree(root_filter)
        with open(curv_filter_path, 'wb') as f:
            f.write('<!DOCTYPE FilterScript>'.encode('utf8'))
            f.write(b'\n')
            output_et.write(f,'utf-8')    

        # Run meshlab to generate reduced mesh with curvature
        logger.info('Start generating mesh with curvature...')
        proc = run_log_cmd(logger, meshlab_cli + [
            '-i', input_mesh_path,
            '-o', curv_mesh_path,
            '-m', 'vc', 'vn',
            '-s', curv_filter_path])
        logger.info(f'Mesh with curvature generation ended with return code {proc.returncode}.')
        return proc.returncode, curv_mesh_path

    @staticmethod
    def gen_simplified_mesh(mesh_path: str, seg_size_thr: list, seg_dir: str, output_dir: str, logger):
        # Definitions of file paths
        filter_file_big = join(filters_dir, 'smooth_simplify_big.mlx')
        filter_file_small = join(filters_dir, 'smooth_simplify_small.mlx')
        filter_file_medium = join(filters_dir, 'smooth_simplify_medium.mlx')
        total_filter_file = join(filters_dir, 'final_reduce.mlx')
        iter_script_file = join(output_dir, 'part_simp_iter.mlx')        
        output_file_part = join(output_dir, 'partially_simplified.ply')
        output_file_tot = join(output_dir, 'simplified_possion_mesh.ply')
        
        # Partial reduction
        input_file = mesh_path
        for filename in os.listdir(seg_dir):
            if (fnmatch(filename, "*box*.mlx")):
                size = int(filename.split('_')[3])
                select_file = os.path.join(seg_dir, filename)
                # Select filter file (how to simplify) according to total mesh number
                if (size > seg_size_thr[2]):
                    filter_file = filter_file_big
                elif (size > seg_size_thr[1]):
                    filter_file = filter_file_medium
                else:
                    filter_file = filter_file_small
                # Generate combined filter
                ProcMlx.merge_selection_action_mlx(select_file, filter_file, iter_script_file)
                # Run simplification
                logger.info(f'Start simplifying mesh with filter {filename}...')
                proc = run_log_cmd(logger, meshlab_cli + [
                    '-i', input_file,
                    '-o', output_file_part, # '-m', 'wt',
                    '-s', iter_script_file])
                logger.info(f'End simplifying mesh with filter {filename} with returncode {proc.returncode}.')                
                input_file = output_file_part
        try:
            os.remove(iter_script_file)
        except:
            logger.warn(f'Failed to remove {iter_script_file}.')
        
        # Total reduction
        logger.info(f'Start simplifying the whole mesh...')
        proc = run_log_cmd(logger, meshlab_cli + [
            '-i', input_file,
            '-o', output_file_tot, # '-m', 'wt',
            '-s', total_filter_file])
        logger.info(f'End simplifying the whole mesh with returncode {proc.returncode}.')
        
        return proc.returncode, output_file_tot

    @staticmethod
    def calc_alignment_rot(input_mesh_path: str, select_mlx_path: str, output_dir: str, logger):
        ''' Calculate the rotation needed to level the fine mesh using coarse mesh.'''

        # Definitions of file paths
        filter_mlx_path = join(filters_dir, 'fit_xz_nofreeze.mlx')
        try_correct_lv_mlx_path = join(output_dir, 'try_correct_lv.mlx')
        try_correct_lv_mlp_path = join(output_dir, 'try_correct_lv.mlp')
        tmp_mesh_path = join(output_dir, 'tmp.ply')
        
        # Read rotation that MeshLab saves in the mlp file
        def read_rotation(rot_mlp_file) -> dict:
            root = ET.parse(rot_mlp_file).getroot()
            for child in root:
                for gchild in child:
                    for ggchild in gchild:
                        rot_trans_str = ggchild.text
            rot_trans_str = rot_trans_str.replace('\n','').strip().split(' ')  # It's 4x4 transformation matrix
            rot_matrix = np.array(rot_trans_str).reshape(4,4).astype(np.float)[:3, :3]
            rot_obj = R.from_matrix(rot_matrix)  # TODO (hkazami): Upgrade SciPy and use as_matrix instead
            rot_vec = rot_obj.as_rotvec()
            rot_axis = rot_vec / np.linalg.norm(rot_vec)
            rot_angle_deg = np.rad2deg(np.linalg.norm(rot_vec))
            logger.info(f'From mlp file: Alignment rotation axis = {rot_axis}, angle (deg) = {rot_angle_deg}')

            # To avoid upside-down (if would result in using ceiling as floor in next task...)
            if (rot_angle_deg < 90) and (rot_angle_deg > -90):
                pass  # No need to do correction
            elif (rot_angle_deg > 90) or (rot_angle_deg < -90):
                logger.warn('Upside-down phenomenon detected, adding 180 rotation about z-axis...')
                flip_back_obj = R.from_rotvec(np.pi * np.array([0, 0, 1]))
                rot_obj = flip_back_obj * rot_obj  # Add 180 deg rotation about z-axis to correct upside-down
                rot_vec = rot_obj.as_rotvec()  # TODO (hkazami): Upgrade SciPy and use as_matrix instead
                rot_axis = rot_vec / np.linalg.norm(rot_vec)
                rot_angle_deg = np.rad2deg(np.linalg.norm(rot_vec))
            else:
                logger.error(f'Alignment rotation angle = {rot_angle_deg} is invalid number, stop...')
                raise RuntimeError
            logger.info(f'Final alignment rotation axis = {rot_axis}, angle (deg) = {rot_angle_deg}')
            return {'scale': 1, 'angle_deg': rot_angle_deg, 'axis': rot_axis.tolist(), 'offset': [0, 0, 0]}

        # Make mlx file
        ProcMlx.merge_selection_action_mlx(select_mlx_path, filter_mlx_path, try_correct_lv_mlx_path)

        # Try to rotate to get rotation
        logger.info(f'Start trying to rotate the reduced mesh {input_mesh_path} to get rotation...')
        proc = run_log_cmd(logger, meshlab_cli + [
            '-i', input_mesh_path,
            '-o', tmp_mesh_path, # '-m', 'wt',
            '-w', try_correct_lv_mlp_path,
            '-s', try_correct_lv_mlx_path])
        try:
            os.remove(tmp_mesh_path)
            os.remove(input_mesh_path.replace('.ply','_out.ply'))  # This ply is auto generated by Meshlab (probably because of the mlp)
        except:
            logger.warn(f'Failed to remove {tmp_mesh_path} and {input_mesh_path.replace(".ply", "_out.ply")}.')
        logger.info(f'End trying to rotate the reduced mesh to get rotation with returncode {proc.returncode}.')
        
        trans_dict = read_rotation(try_correct_lv_mlp_path)
        return proc.returncode, trans_dict
    
    @staticmethod
    def gen_transformed_mesh(input_mesh_path: str, output_dir: str, output_mesh_name: str, trans_dict: dict, logger) -> int:
        ''' Rotate input mesh with input rotation axis & angle (deg), and save as new mesh. '''
        
        # Definitions of file paths
        trans_mesh_path = join(output_dir, output_mesh_name)
        trans_mlx_path = join(output_dir, output_mesh_name.replace('.ply', '.mlx'))
        
        ProcMlx.write_trans_mlx(trans_mlx_path, trans_dict['axis'], trans_dict['angle_deg'], trans_dict['scale'], trans_dict['offset'])

        logger.info(f'Start doing rotation to the mesh {input_mesh_path}...')
        proc = run_log_cmd(logger, meshlab_cli + [
            '-i', input_mesh_path,
            '-o', trans_mesh_path, # '-m', 'wt',
            '-s', trans_mlx_path])
        try:
            os.remove(trans_mlx_path)
        except:
            logger.warn(f'Failed to remove {trans_mlx_path}.')
        logger.info(f'End doing rotation to the mesh as saved as {trans_mesh_path} with returncode {proc.returncode}.')
        return proc.returncode, trans_mesh_path

    @classmethod
    def cut_floor_mesh(cls, input_mesh_path: str, output_dir: str, r_floor_cut: float, logger):
        ''' Just assume that everything with y below some threshold is floor, and just cut it off. '''
 
        # Definitions of file paths
        floor_mesh_path = join(output_dir, 'floor_mesh.ply')
        delete_ceiling_file = join(output_dir, 'delete_ceiling.mlx')

        # Generate the convex hull of mesh and use the info to generate mlx to delete ceiling
        _, _, _, y_info = cls._gen_convexhull(input_mesh_path, output_dir, logger)
        delete_ceiling_mlx_path = ProcMlx.write_del_ceil_mlx(output_dir, y_info, r_floor_cut)

        logger.info(f'Start cutting floor mesh off from {input_mesh_path}...')
        proc = run_log_cmd(logger, meshlab_cli + [
            '-i', input_mesh_path,
            '-o', floor_mesh_path,
            '-s', delete_ceiling_mlx_path])
        logger.info(f'End cutting floor mesh off with returncode {proc.returncode}.')
        return proc.returncode, floor_mesh_path
    
    @staticmethod
    def MeshLabPly2BabylonObj(input_mesh_path: str, logger):
        ''' It does 2 things: Flip x & y, and save as obj file. '''
        
        # Definitions of file paths
        output_mesh_path = input_mesh_path.replace('.ply', '.obj')
        mlx_path = join(filters_dir, 'flip_xy.mlx')

        logger.info(f'Start exporting obj file for Babylon from {output_mesh_path}...')
        proc = run_log_cmd(logger, meshlab_cli + [
            '-i', input_mesh_path,
            '-o', output_mesh_path, # '-m', 'wt',
            '-s', mlx_path])
        try:
            os.remove(output_mesh_path + '.mtl')
        except:
            logger.warn(f'Failed to remove {output_mesh_path + ".mtl"}.')
        logger.info(f'End exporting obj file for Babylon with returncode {proc.returncode}.')
        return proc.returncode, output_mesh_path
