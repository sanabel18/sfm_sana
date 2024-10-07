import os, shutil
import pickle
from GPSProcessor.align_routes.align_routes import AlignRoutes, gen_route_list, \
        gen_src_mrk_data_list
from tool.tower_stage_manager.tower_file_genutils import gen_repo_basename, \
        gen_folder_path_from_base, get_tower_repo_path_from_repo_name, \
        load_repo_name_list
from utils.safe_yaml import safe_yaml_load, safe_yaml_dump

class AlignGPS:
    """
    In stage_dir, we have all {repo_name}.geojson after stage_loc, export_to_gps
    Now we would like to align the nodes(intersection points) and export as
    {repo_name}_aligned.geojson
    Args:
    stage_dir: str:path of workspace
    tower_stage_dir: str: root path of tower repos
    main_tag: str: commit tag of main repo
    src_mrk_tag_label: str: commit tag and file label for source marker
                           ex: tag_{commit_tag}_label_{file-label}
    repo_name_file: str: path of file that stores the info of completed node list from stage_loc
    main_node_file: str: path of file that stores the repo_name of main node from stage_loc
    Returns:
    the aligned {repo_name}_aligned.geojson will be exported at stage_dir, and copy
    to its corresponding tower repo
    """
    def __init__(self, align_gps_cfg):
        self.stage_dir = align_gps_cfg['stage_dir']
        self.tower_stage_dir = align_gps_cfg['tower_stage_dir']
        main_tag = align_gps_cfg['main_tag']
        src_mrk_tag_label = align_gps_cfg['src_mrk_tag_label']
        repo_name_file = align_gps_cfg['complete_node_list']
        main_node_file = align_gps_cfg['main_node_file'] 
        
        self.repo_name_list = load_repo_name_list(repo_name_file)
        self.main_node = pickle.load(open(main_node_file,'rb'))

        self.tower_repo_path_list = \
            get_tower_repo_path_from_repo_name(self.tower_stage_dir, self.repo_name_list, main_tag)
 
        src_marker_path_list = \
            get_tower_repo_path_from_repo_name(self.tower_stage_dir, self.repo_name_list, src_mrk_tag_label)
        gps_file_list = self.get_gps_file_list(self.stage_dir, self.repo_name_list)
        src_mrk_file_list = self.get_src_mrk_file_list(src_marker_path_list)
        
        self.geo_src_list = self.gen_geo_src_list(self.repo_name_list, src_mrk_file_list, gps_file_list)
        self.src_mrk_data_list = gen_src_mrk_data_list(self.geo_src_list)
        route_list = gen_route_list(self.geo_src_list)
        self.align_routes = AlignRoutes(route_list)
        self.align_routes.set_route_to_nodes_from_src_marker(self.src_mrk_data_list)
    
    def get_src_mrk_file_list(self, src_mrk_path_list):
        """
        Args: list of str: list of path that contains source_marker.json
        Return: list of str: list of path of abspath of source_marker.json
        """
        src_mrk_file_list = []
        for src_mrk_path in src_mrk_path_list:
            src_mrk_file = os.path.join(src_mrk_path, 'source_marker.json')
            src_mrk_file_list.append(src_mrk_file)
        return src_mrk_file_list

    
    def get_gps_file_list(self, stage_dir, repo_name_list):
        """
        Args: 
        stage_dir: path of that contains 
        repo_name_list: list of str: list of repo_name
        Return: list of str: list of full path of gps.geojson files
        """
        gps_file_list = []
        for repo_name in repo_name_list:
            gps_file = os.path.join(stage_dir, '{}.geojson'.format(repo_name))
            gps_file_list.append(gps_file)
        return gps_file_list

    def gen_geo_src_list(self, repo_name_list, src_mrk_file_list, gps_file_list):
        """
        the format of geo_src_list looks like
        list of [route_type, src_mrk_file, geojson file]
        route_type is 'main' for main node, 
        route_type is repo_name for other child node
        Args:
        repo_name_list: list of str: list of repo_name
        src_mrk_file_list: list of str: list of file path of source_marker.json
        gps_file_list: list of str: list of file path of gps.geojspn
        Return:
        geo_src_list: list of list 
        """
        geo_src_list = []
        for repo_name, src_mrk_file, gps_file in zip(repo_name_list, src_mrk_file_list, gps_file_list):
            if repo_name == self.main_node:
                file_base = os.path.basename(gps_file)
                file_path = os.path.dirname(gps_file)
                filename_base = os.path.splitext(file_base)[0]
                filename_ext = os.path.splitext(file_base)[1]
                file_name_edit = f'{filename_base}_edit{filename_ext}'
                gps_file_edit = os.path.join(file_path, file_name_edit)
                geo_src = ["main", src_mrk_file, gps_file_edit]
            else:
                geo_src = [repo_name, src_mrk_file, gps_file]
            geo_src_list.append(geo_src)
        return geo_src_list
    
    def export_to_tower(self, repo_name_list, tower_repo_path_list, stage_dir):
        """
        copy aligned .geojson to its corresponing tower repos
        Args: 
        repo_name_list: list of str: list of repo_name
        tower_repo_path_list: list of str: list of tower_repo_path
        stage_dir: str: path of stage workspace
        Return:
        geo_data_sfm_stage_loc_aligned.geojson will be copy to tower repo
        """
        for repo_name, tower_path, in zip(repo_name_list, tower_repo_path_list):
            if repo_name == self.main_node:
                file_name = '{}_edit.geojson'.format(repo_name)
            else:
                file_name = '{}_aligned.geojson'.format(repo_name)
            src_path = os.path.join(stage_dir, file_name)
            dst_path = os.path.join(tower_path, 'geo_data_sfm_stage_loc_aligned.geojson')
            shutil.copy(src_path, dst_path)

    def run(self):
        self.align_routes.align_routes(self.stage_dir)
        self.export_to_tower(self.repo_name_list, self.tower_repo_path_list, self.stage_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config", required=True, help='config file')

    input_vals = parser.parse_args()
    input_config_file = input_vals.config
    input_config = safe_yaml_load(input_config_file) 
    AlignGPS(input_config).run()

