#!/opt/conda/bin/python

import json
import numpy as np
import os, sys
import pickle
import shutil
import glob

from copy import deepcopy
from os.path import join

from project import ReconProj, LocProj
from utils.exportutil import copy_2_tower, export_route_obj_visualizations, copy_from_tower
from utils.genutil import get_current_time_str, TransData, transform_sfm_data, decompose_trf_mat
from utils.logger import get_logger, close_all_handlers
from utils.genutil import safely_create_dir
from utils.meshlabutil import GenMesh
from utils.safe_yaml import safe_yaml_load, safe_yaml_dump
from utils.safe_json import safe_json_dump
from tool.tree_generator import TreeGenerator, load_src_markers, gen_NID2SID
from tool.tower_stage_manager.tower_file_genutils import load_src_marker_from_tower_repo
from tool.tree_to_nested_dict import Tree2NestedDict
import tool.routes_to_gps as R2G
from tool.tower_stage_manager.tower_file_genutils import gen_repo_basename, \
        gen_folder_path_from_base, get_tower_repo_path_from_repo_name,\
        load_repo_name_list, load_src_marker_from_path_list
    

class StageLocalization:
    """
    transfrom routes within a stage into coordinate system of root routes
    root route is chosen such that most routes connecte to
    the input config dictionary should contain
    stage_dir: path where stage localization project runs
               the program will construct sub-folders under stage_dir
               as tree_0, tree_1,...corresponding to different trees
    tower_stage_dir: path of all the tower repos  within a stage
    loc_prj_template: config template of localization project, the important
        one is pipeline, which defines tasks will be execute within this proj
        project:
            project:
                type: LocProj
                project_dir: 'Filled by script'
                force_new_project_dir: false  # true for the 1st time, otherwise false
                force_copy_data_dir_main: false  # true for the 1st time, otherwise false
                n_thread: 20
                process_prio: 2
            n_lense_for_rot: 0
            pipeline:
                - SfM_MATCH_SUB2MAIN
                - CALCULATE_TRANSFORMATION_SUB2MAIN
                - APPLY_TRANSFORMATION_EXPORT_SUBS
                - CONCAT_CONVERT_TO_GEOJSON
        transform:
            prj_dir_main: 'Filled by script' # /volume/cpl-dev/sfm/pipeline/superprj/SML_TC_05_07_23/loc
            data_dir_name_main: 'Filled by script' # /data_SML_TC_05_07_23_000
            prj_dir_sub: 'Filled by script' # //volume/cpl-dev/sfm/pipeline/superprj/SML_TC_06_04/loc
            data_dir_name_sub: 'Filled by script' # /data_SML_TC_06_04_000
            threshold: 1
            intrinsics_sub: separate
            type_main_prj: LocProj
            type_sub_prj: LocProj
            opt_with_footprints: false
            opt_with_stepsize_anchor: true
        apply:
            overwrite_subs: false
 
   tree:
        src_root: path of all SfM project in stage
        src_mrk_tag_label: str: 'tag_{commit tag}_label_{file-label}' tag and label of source marker file
        gps_anchor_tag_label: 'tag_{commit tag}_label_{file-label}' tag and label of gps anchor file
        complete_repo_path: str: file path of that stores list of completed repo_name tower preprocessor
        main_tag: str: commit tag of main repo
    Args: input yaml file path
    """
    def __init__(self, stage_loc_cfg_path):
        # Load config
        stage_loc_cfg = safe_yaml_load(stage_loc_cfg_path)
        self.stage_dir_root = stage_loc_cfg['stage_dir']
        self.tower_stage_dir = stage_loc_cfg['tower_stage_dir']
        self.loc_prj_template = stage_loc_cfg['loc_prj_template']
        self.src_root = stage_loc_cfg['tree']['src_root']
        self.opt_with_stepsize_anchor =\
                self.loc_prj_template['transform']['opt_with_stepsize_anchor']
        src_mrk_tag_label = stage_loc_cfg['tree']['src_mrk_tag_label']
        self.gps_anchor_tag_label = stage_loc_cfg['tree']['gps_anchor_tag_label']
        repo_name_file = stage_loc_cfg['tree']['complete_repo_path']
        self.main_tag = stage_loc_cfg['tree']['main_tag']
        self.recon_fps = stage_loc_cfg['tree']['recon_fps']
        gps_anchor_path_file = stage_loc_cfg['tree']['gps_anchor_path']
        
        self.repo_name_list = load_repo_name_list(repo_name_file)
        
        if not os.path.isdir(self.stage_dir_root):
            os.makedirs(self.stage_dir_root)
        

        self.logger = get_logger('stage_loc.log', self.stage_dir_root)

        
        self.repo_path_list = get_tower_repo_path_from_repo_name(self.tower_stage_dir, self.repo_name_list, self.main_tag)
        self.prepare_route_src_dir(self.repo_path_list, self.repo_name_list)
        
        self.gps_anchor_name_list = load_repo_name_list(gps_anchor_path_file)
       
        src_marker_path_list = \
                get_tower_repo_path_from_repo_name(self.tower_stage_dir, self.repo_name_list, src_mrk_tag_label)
        
        tree_generator = self.init_tree_generator_from_tower_repo(src_marker_path_list, fps=recon_fps)
        self.tree_generator = tree_generator
    
   
    def init_tree_generator_from_tower_repo(self, src_marker_path_list, fps=2, slice_frame_num=30):
        '''
        from srouce markers generate the graph of connectivity betweern routes
        construct TreeGenerator 
        Args: 
        src_mark_path: str
            path where all source_marker.json files in stage located
        fps: int, number of frame used in 1 sec
        slice_fram_num: int, frame number in one slice
        Retrun:
            TreeGenerator
        '''
        
        src_marker_data_list = load_src_marker_from_path_list(src_marker_path_list)
        slice_time_list = [slice_frame_num/fps]*len(self.repo_path_list)
        NID2SID_list  =  gen_NID2SID(src_marker_data_list, slice_time_list)

        tree_generator = TreeGenerator(self.repo_name_list, NID2SID_list)
        g = tree_generator.get_graph()
        return tree_generator
    

    def prepare_route_src_dir(self, repo_path_list, repo_name_list):
        '''
        under self.src_root create folders with repo_name_list, and copy from tower
        from repo_path_list
        Args: 
        repo_path_list: list of str: list of repo path
        repo_name_list: list of str: list of repo name 
        '''
        for repo_name, repo_path in zip(repo_name_list, repo_path_list):
            local_route_dir = os.path.join(self.src_root, repo_name)
            if not os.path.isdir(local_route_dir):
                dst = safely_create_dir(self.src_root, repo_name)
            #copy data from tower
            copy_from_tower(local_route_dir, repo_path)
            self._rename_data_dirs(local_route_dir, repo_name)
    
    def _rename_data_dirs(self, src_dir, repo_name):
        """
        while copying data* folders from tower repo, the naming is related to commitID instead of repo name
        here we rename the data_* folder according to our convention since we are using repo name
        to manage our stage repos
        Args:
        src_dir: str: path to put the copied tower repos
        repo_name: str: repo name
        """
        data_dirs = glob.glob(join(src_dir, 'data_*'))
        commitID = os.path.basename(data_dirs[0]).split('_')[-2]
        for data_dir in data_dirs:
            if os.path.isdir(data_dir):
                if commitID in data_dir:
                    os.rename(data_dir, data_dir.replace(commitID, repo_name))
        
    def get_stage_tree(self):
        '''
        from self.tree_generator, get a BFS tree and convert it to a nested dict.
        this function can be called multiple times and different tree will be reported with different calls.
        if the number of called-times exceeds the number of BFS trees that self.tree_generator can report, 
        it will return None
        '''
        bfs_tree, root_node = self.tree_generator.get_BFS_tree()
        if bfs_tree == None or root_node == None:
            return None
        else:
            key_root = root_node
            graph = self.tree_generator.get_graph()
            tree_2_nested_dict = Tree2NestedDict(graph, bfs_tree, key_root)
            nested_dict = tree_2_nested_dict.get_nested_dict()
            return nested_dict

   
    def safe_copy_data_dir(self, src, dst):
        '''
        copy folder from src to dst
        Args:
        src: path of src
        dst: path of dst
        '''
        if not os.path.isdir(dst):
            os.makedirs(dst)
        
        for item in os.listdir(src):
            if (item == 'img_prj') or (item == 'matches'):
                print(f'Making symbolic link for {item} from {src} to {dst}...')
                os.symlink(join(src, item), join(dst, item))  # To save space
            elif item == '.ipynb_checkpoints':
                print(f'Ignored {item}.')
            else:
                try:
                    print(f'Copying {item} from {src} to {dst}...')
                    shutil.copy(join(src, item), join(dst, item))  # Except for img and matches, everything is file not folder
                except:
                    print(f'Unexpected folder {item} is in sub data directory. Ignored.')
        return
    

    def get_transformation_from_gps_anchor_list(self, stage_dir, gps_anchor_name_list):        
        """
        use gps anchor files in gps_anchor_path_list, to determine the transformation
        between local SfM coordinates and global GPS system
        Args:
        stage_dir: str: path of workspace(where SfM projects can be found)
        gps_anchor_path_list: list of str: path of gps anchor files
        """
        gps_anchor_path_list = \
                get_tower_repo_path_from_repo_name(self.tower_stage_dir, \
                gps_anchor_name_list, self.gps_anchor_tag_label)
 
        gps_anchor_file_list = self.get_geojson_file_list(gps_anchor_path_list)
        gps_path_list, gps_path_obj_list  = self.gen_gps_path_list(gps_anchor_file_list)
        sfm_path_list = self.gen_sfm_path_list(gps_anchor_name_list, stage_dir, \
                                               gps_path_obj_list)


          
        start_microsec = R2G.get_start_microsec_from_gps_path(gps_path_list[0])
        c, Rmat, t, utm_data_list =  \
            R2G.get_trf_from_sfm_gps_path_list(sfm_path_list, gps_path_list, 100)
    
        utm_data = utm_data_list[0]

        return c, Rmat, t, utm_data, start_microsec
    
    def gen_output_list(self, stage_dir, node_list):
        """
        We export GPS from SfM routes to two paths:
        1. local working dir (stage_dir), with name "repo_name".geojson to be easily reviewd
        2. tower repos as name "geo_data_sfm_stage_loc.geojson" 
        Args:
        stage_dir: str
        node_list: list of str: list of node(repo name) that will be exported to GPS
        """
        output_path_list = []
        output_tower_path = get_tower_repo_path_from_repo_name(self.tower_stage_dir, node_list, self.main_tag)
        for node, tower_path in zip(node_list, output_tower_path):
            output_path = []
            output_path.append(join(stage_dir, '{}.geojson'.format(node)))
            output_path.append(join(tower_path,'geo_data_sfm_stage_loc.geojson'))
            output_path_list.append(output_path)
        return output_path_list


    def apply_trf_and_export_on_routes(self, node_list, stage_dir, utm_data, c, Rmat, t, start_microsec):
        """
        apply transformation on all routes in stage
        c: Scale
        Rmat: 3X3 rotation matrix
        t: translation  
        within sfm coordinate system
 
        Args:
        node_list: list of str: repo name
        stage_dir: str: workspace path
        utm_data: List of tuples (utm_east, utm_north, utm_zone, utm_letter)
                  ex: [(276778.07023962133, 2602132.268205515, 51, 'Q')]
        c: float. scale 
        Rmat: List[3,3]. 3X3 rotation matrix
        t: List[3]. translation
        """
        node_list = sorted(node_list)
        sfm_path_list = self.gen_sfm_path_list(node_list, stage_dir)
        output_path_list = self.gen_output_list(stage_dir, node_list)
        for sfm_path, output_path in zip(sfm_path_list, output_path_list):
            R2G.apply_trf_and_export_to_gps(sfm_path, utm_data, c, Rmat, t, start_microsec, output_path)
    
    def get_geojson_file_list(self, gps_anchor_path_list):
        """
        get geojsonfile list in gps_anchor_path_list
        Args:
        gps_anchor_path_list: list of str: path that contains anchor geojson file
        Return:
        geojson_list: list of str: path of geojson file
        """
        geojson_list = []
        for gps_anchor_path in gps_anchor_path_list:
            gps_geojson_file = glob.glob(gps_anchor_path+'/*geojson')[0]
            geojson_list.append(gps_geojson_file)
        return geojson_list

    def gen_gps_path_list(self, geojson_list):
        """
        Args: list of str: list of path of gps.geojson file
        Return:
        gps_path_list: list of R2G.GPS_Path
        """
        gps_path_list = []
        gps_path_obj_list = []
        if self.opt_with_stepsize_anchor:
            read_footprint = False
        else:
            read_footprint = True
        for geojson in geojson_list:
            gps_geojson = R2G.load_geojson(geojson)
            gps_path_obj = R2G.get_gps_path(gps_geojson, read_footprint=read_footprint)
            gps_path = gps_path_obj.gps_path
            gps_path_list.append(gps_path)
            gps_path_obj_list.append(gps_path_obj)
        return gps_path_list, gps_path_obj_list
    
    def gen_sfm_path_list(self, node_list, stage_dir, gps_path_obj_list=[]):
        """
        this function generates list of R2G.SfM_path for two cases:
        1. finding the transformation between sfm and GPS coord. system
        2. apply transformation on all sfm routes
    
        for case 1. gps_path_obj_list will be set since it will be used to calculate stepsize anchor footprint
                    and sfm_path with step-size anchor as footprint will be generated
        for case 2. gps_path_obj_list can be left as default, sfm_path without footprint will be returned

        from stage_dir detect the data* folders with certain repo name, 
        and use them to create SfM_path object
        
        Args: 
        node_list: list of str: list of repo name
        stage_dir: str: path of workspace
        Return:
        list of R2G.SfM_path
        """
        sfm_path_list = []
        sfm_path_obj_list = []
        if self.opt_with_stepsize_anchor:
            read_footprint = False
        else:
            read_footprint = True
        for node in node_list:
            _data = 'data*'+node+'*'
            data_dir_list = glob.glob(join(stage_dir, _data))
            sfm_path_obj = R2G.get_sfm_path_from_data_dir_list(data_dir_list, \
                                                read_footprint = read_footprint)
            sfm_path = sfm_path_obj.sfm_path
            sfm_path_list.append(sfm_path)
            sfm_path_obj_list.append(sfm_path_obj)
         
        if self.opt_with_stepsize_anchor and len(gps_path_obj_list) > 0:
            sfm_path_w_stepsize_anchor_list = []
            for sfm_path_obj, gps_path_obj in zip(sfm_path_obj_list, gps_path_obj_list):
                sfm_path_w_stepsize_anchor = \
                        R2G.gen_sfm_path_stepsize_anchor(sfm_path_obj, gps_path_obj)
                sfm_path_w_stepsize_anchor_list.append(sfm_path_w_stepsize_anchor)
            return sfm_path_w_stepsize_anchor_list
        else:
            return sfm_path_list
 
    def correct_level(self, folder_path, delta_trf_mat, logger):
        '''
        apply transfromation delta_trf_mat onto data within folder_path
        Args:
        folder_path: path of data to be transformed
        delta_trf_mat: (4,4) array of 3D affine trasformation matrices
        [ 
          R11 R12 R13 t1
          R21 R22 R23 t2
          R31 R23 R33 t3
          0   0   0   1
        ]
        '''
        # Paths & names
        trans_json_path = join(folder_path, 'transformation.json')
        sfm_data_path = join(folder_path, 'sfm_data_transformed.json')
        mesh_name = 'transformed_mesh.ply'
        if self.opt_with_stepsize_anchor:
            sfm_data_stepsize_anchor_path = join(folder_path, 'sfm_data_stepsize_anchor_transformed.json')
        # Apply transformation
        trans_data = TransData().set_from_json(trans_json_path)
        trans_data.append_trf(delta_trf_mat.tolist(), note='stage_loc: correct_level')
        trans_data.write2json(trans_json_path)            
        # Sfm data
        transform_sfm_data(
            input_sfm_data_path=sfm_data_path, output_sfm_data_path=sfm_data_path, trf_mat=delta_trf_mat, logger=logger)
        if self.opt_with_stepsize_anchor:
            transform_sfm_data(
                input_sfm_data_path=sfm_data_stepsize_anchor_path, \
                output_sfm_data_path=sfm_data_path, trf_mat=delta_trf_mat, logger=logger)
 
        # Mesh
        if os.path.isfile(join(folder_path, mesh_name)):
            delta_trf_dict = decompose_trf_mat(delta_trf_mat.tolist())
            _, _ = GenMesh.gen_transformed_mesh(
                input_mesh_path=join(folder_path, mesh_name), \
                output_dir=folder_path, \
                output_mesh_name=mesh_name, \
                trans_dict=delta_trf_dict, logger=logger)
        
        return
    
    def get_node_list_from_node_tree(self, node_tree):
        '''
        Given a nested tree,
        for example:
        { 
        "repo_src_path/repo_name_1": {
            'parent_to_own_slice': ['003','000'],
            'child_node': [    
                {
                    "repo_src_path/repo_name_child1": {
                    'parent_to_own_slice': ['002','001'],
                    'child_node': []
                    }
                },
                {
                    "repo_src_path/repo_name_child2": {
                    'parent_to_own_slice': ['000','000'],
                    'child_node': []
                    },
                }    
            ]

          }
        }
        
        return list of all the nodes within this tree
        in this case, node_list = [
                                    "repo_src_path/repo_name_1",
                                    "repo_src_path/repo_name_child1",
                                    "repo_src_path/repo_name_child2"
                                  ]

        Args: dict that represent a nested tree
        Reture: list of nodes in tree
        '''
        queue = [node_tree]
        node_list = []
        while queue:
            node = queue.pop(0)
            node_source = list(node.keys())[0]
            node_list.append(node_source)
            child_node_list = node[node_source]['child_node']
            if child_node_list:
                for child_node in child_node_list:
                    queue.append(child_node)
        return node_list 
    
    def run_stage_loc(self, debug=False):
        '''
        fetch tree and feed it to bfs_execute_tree()
        after execution of each tree there is list of failed_nodes
        if failed_nodes is empty, all the Loc Proj runs successfully with in tree
        then the while loop shall stop.
        if failed_nodes is not empty, the failed_nodes will be exported to file
        and another tree will be ask for the next tree execution
        
        If within auto pipeline, one can limit the number of tree it asked to 1.
        So the tree execution will be run only one time
        '''
        failed = True
        treeID = 0
        while(failed):
            stage_tree = self.get_stage_tree()
            if stage_tree == None:
                self.logger.info("can not get stage tree.")
                break
 
            stage_dir = os.path.join(self.stage_dir_root, "tree_{}".format(treeID))
            if not os.path.isdir(stage_dir):
                os.makedirs(stage_dir)
           
            treeyaml = os.path.join(stage_dir, "tree_{}.yaml".format(treeID))
            self.dump_tree(stage_tree, treeyaml)
            logger_tree = get_logger('stage_loc_tree_{}.log'.format(treeID), stage_dir)                    

            all_nodes = self.get_node_list_from_node_tree(stage_tree)
            main_node = list(stage_tree.keys())[0]
            failed_nodes = self.localize_nodes_on_bfs_tree(stage_tree, stage_dir, logger_tree)
            #failed_nodes = pickle.load(open(os.path.join(stage_dir, 'failed_nodes.p'),'rb'))
            node_list = list(set(all_nodes) - set(failed_nodes))
            # remove gps anchor if it is not in node_list
            gps_anchor_name_list = list(set(self.gps_anchor_name_list).intersection(set(node_list)))
            
            c, Rmat, t, utm_data, start_microsec = self.get_transformation_from_gps_anchor_list(stage_dir, gps_anchor_name_list)

            self.apply_trf_and_export_on_routes(node_list, stage_dir, utm_data, c, Rmat, t, start_microsec)
            self.logger.info(f'failed_nodes {failed_nodes}')

            if (len(failed_nodes) > 0):
                failed = True
                outfile = "failed_nodes_tree_{}.p".format(treeID)
                outfile = os.path.join(stage_dir, outfile)
                pickle.dump(failed_nodes, open(outfile,'wb'))
                #self.write_failed_nodes(failed_nodes, outfile)
            else:
                failed = False
                self.logger.info("stage localization ends successfully.")
            # export node_list
            node_list_file = os.path.join(stage_dir, 'complete_node_list.p')
            pickle.dump(node_list, open(node_list_file, 'wb'))
            # export main node
            main_node_file = os.path.join(stage_dir, 'main_node.p')
            pickle.dump(main_node, open(main_node_file, 'wb'))
            treeID += 1
            if debug==False:
                break

    def dump_tree(self, stage_tree, treeyaml):
        '''
        export stage_tree to a yaml file
        Args: 
        stage_tree: dict of nested tree
        treeyaml: path of yaml file to be saved
        '''  
        safe_yaml_dump(stage_tree, treeyaml)

    def write_failed_nodes(self, failed_nodes, outfile):
        '''
        export failed_nodes to file outfile
        Args: 
        failed_nodes: list of str
        outfile: path of file with failed_nodes written 
        '''
        with open(outfile, 'w') as f:
            for node in failed_nodes:
                f.write(node)
        f.close()
            
        
    def localize_nodes_on_bfs_tree(self, stage_tree, stage_dir, logger):
        '''
        Execute SfM localization Projects with oder of BFS in stage_tree
        If there is some routes within stage failed the SfM localization Porj,
        they will be skipped and so do their children in the tree.
        The skipped routes will be noted in failed_node_list
        Args:
        srage_tree: nested dict
        stage_dir: path of running the stage localization Project 
        logger: logger object
        
        Retrun:
        failed_node_list: list 
        '''
        failed_node_list = []
        queue = [stage_tree]
        loc_successed = False
        while queue:
            
            node = queue.pop(0)
            node_source = list(node.keys())[0]
            node_source_path = join(self.src_root, node_source)
            child_node_list = node[node_source]['child_node']

            # If it's the root then copy data_dirs directly
            origin_slice_path = join(stage_dir, 'data_' + os.path.basename(node_source_path) + '_000')
            first = list(stage_tree.keys())[0]
            if (not os.path.isdir(origin_slice_path)
                and node_source == list(stage_tree.keys())[0]):
                for item in os.listdir(node_source_path):
                    # Copy data directories
                    if os.path.isdir(join(node_source_path, item)) and item[:5]=='data_':
                        src = join(node_source_path, item)
                        dst = join(stage_dir, item)
                        try:
                            self.safe_copy_data_dir(src, dst)
                        except Exception as e:
                            print(f'Failed to copy: {e}')
                            
                # Do level correction
                with open(join(origin_slice_path, 'routes.json'), 'r') as f:
                    init_frame_rot = json.load(f)['frames'][0]['rotation']
                delta_trf_mat = np.eye(4)
                delta_trf_mat[:3,:3] = init_frame_rot
                for folder in os.listdir(stage_dir):
                    if folder[:5]=='data_':
                        self.correct_level(join(stage_dir, folder), delta_trf_mat, logger)
                        print(f'Level correction done for slice {folder}.')
                    
            
            if child_node_list:
                for child_node in child_node_list:
                    loc_successed = False
                    child_node_source = list(child_node.keys())[0]
                    child_node_source_path = join(self.src_root, child_node_source)

                    parent_slice_nr = list(child_node.values())[0]['parent_to_own_slice'][0]
                    child_slice_nr = list(child_node.values())[0]['parent_to_own_slice'][1]

                    prj_dir_main = stage_dir
                    data_dir_name_main = 'data_' + os.path.basename(node_source_path) + '_' + parent_slice_nr
                    prj_dir_sub = child_node_source_path
                    data_dir_name_sub = 'data_' + os.path.basename(child_node_source_path) + '_' + child_slice_nr

                    loc_prj_cfg = deepcopy(self.loc_prj_template)
                    
                    loc_prj_cfg['project']['project_dir'] = stage_dir
                    loc_prj_cfg['transform']['prj_dir_main'] = prj_dir_main
                    loc_prj_cfg['transform']['data_dir_name_main'] = data_dir_name_main
                    loc_prj_cfg['transform']['prj_dir_sub'] = prj_dir_sub
                    loc_prj_cfg['transform']['data_dir_name_sub'] = data_dir_name_sub

                    loc_prj_cfg_name = (os.path.basename(child_node_source_path) + '_' + child_slice_nr + '__TO__' + 
                                        os.path.basename(node_source_path) + '_' + parent_slice_nr + '.yaml')
                    loc_prj_cfg_path = join(stage_dir, loc_prj_cfg_name)

                    # Run localization if the loc prj config does not exist yet
                    if os.path.isfile(loc_prj_cfg_path):
                        logger.info(f'The localization config {loc_prj_cfg_path} exists, which means it is already done. Ignore.')
                        loc_successed = True
                    else:
                        logger.info(f'The localization config {loc_prj_cfg_path} does not exist yet, start to run it...')
                        returncode_loc = LocProj(loc_prj_cfg).run_pipeline()
                        logger.info(f'returncode_loc {returncode_loc}')
                        if returncode_loc == 0:
                            loc_prj_cfg['done_timestamp'] = get_current_time_str()
                            file_path = join(stage_dir, loc_prj_cfg_name)
                            safe_yaml_dump(loc_prj_cfg, file_path)
                            if copy_2_tower(stage_dir,
                                join(self.tower_stage_dir, os.path.basename(child_node_source_path)),
                                os.path.basename(child_node_source_path)) != 0:
                                raise RuntimeError('Copy to Tower failed.')
                            loc_successed = True
                            logger.info(f'The localization {loc_prj_cfg_path} successfully ends.') 
                        else:
                            logger.info(f'loc node {list(child_node.keys())} failed')
                            failed_nodes = self.get_node_list_from_node_tree(child_node)
                            failed_node_list += failed_nodes
                            logger.info(f'failed_node_list {failed_node_list}')
                    logger.info(f'loc_successed {loc_successed}')
                    if loc_successed:
                        queue.append(child_node)
                    
        return failed_node_list
                

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config", required=True, help='config file')
    parser.add_argument("-b", "--debug", action="store_true")

    input_vals = parser.parse_args()
    input_config = input_vals.config
    input_debug = input_vals.debug
    StageLocalization(input_config).run_stage_loc(debug=input_debug)
