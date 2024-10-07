import os, sys
import tool.routes_to_gps as R2G
import logging
import argparse

from utils.safe_json import safe_json_dump
from utils.logger import get_logger, close_all_handlers
DEFAULT_LOGGER = logging.getLogger(__name__)


def test(loglevel, test_case, logfilename=''):
    '''
    run the test and use plot_geojson.py to see the 
    geojson file from sfm_path
    and geojson file from gps_path + sfm_path
    
    if they matches well then the transfomration is good
    the tree_walk case is baseline, it matches the results
    both from sfm_path, so the final result should be perfectly aligned.

    little_train is matching sfm_path to gps_path , so we will expect
    some non-perfect matcheing, however the 2D projection along gravity direction 
    should look reasonably good, as in little_train case.
    Args:
    loglevel: 0: INFO 
              >0: DEBUG
    logfilename:
        if set, will output to file with logfilename
    '''
    if loglevel == 0:
        level= logging.INFO
    if loglevel > 0:
        level = logging.DEBUG
    if len(logfilename)==0:
        logging.basicConfig(level=level)
    else:
        logging.basicConfig(filename = logfilename,level=level)
    logger = DEFAULT_LOGGER
    if test_case == 'mt_dir':
        test_mt_dir(logger)
    elif test_case == 'tree_walk':
        test_tree_walk(logger)
    elif test_case == 'expand_gps':
        test_expand_gps(logger)
    else:
        logger.info('test case not suuported')

def test_mt_dir(logger):
    input_routes_base_dir = './mt_dir'
    input_geojson_file = './mt_dir/geodata_RTK.geojson'
    output_geojson_file = 'test_mt_dir.geojson'
 

    output_geojson_file_list = [output_geojson_file]

    sfm_path_obj = R2G.get_sfm_path(input_routes_base_dir, read_footprint=False)

    gps_geojson = R2G.load_geojson(input_geojson_file)
    gps_path_obj = R2G.get_gps_path(gps_geojson, read_footprint=False)
    
    sfm_path_interp = R2G.gen_sfm_path_stepsize_anchor(sfm_path_obj, gps_path_obj)

    logger.info(f'test case no 1') 
    R2G.export_routes_w_GPS_anchor(sfm_path_interp, gps_geojson,  
                               output_geojson_file_list, 
                               150, 
                               0,
                               False,
                               logger = logger,
                               smoothen_match = False, 
                               smooth_type = 'anchor',
                               match_footprint = True, 
                               read_footprint = False,
                               combined = False)
    
def test_tree_walk(logger):        
    logger.info(f'test case no 2') 
    input_routes_base_dir = './tree_walk'
    input_geojson_file = './tree_walk/geo_data_sfm.geojson'
    output_geojson_file = 'test_tree_walk.geojson'
    output_geojson_file_combo = output_geojson_file.split('.')[0] + "_combo.geojson"
    
    output_geojson_file_list = [output_geojson_file, output_geojson_file_combo]
    sfm_path_obj = R2G.get_sfm_path(input_routes_base_dir, read_footprint=True)
    sfm_path = sfm_path_obj.sfm_path
    gps_geojson = R2G.load_geojson(input_geojson_file)
    R2G.export_routes_w_GPS_anchor(sfm_path, gps_geojson,  
                               output_geojson_file_list, 
                               150, 
                               0,
                               False,
                               logger = logger, 
                               smoothen_match = False, 
                               smooth_type = 'anchor',
                               match_footprint = True, 
                               read_footprint = True,
                               combined = False)
def test_expand_gps(logger):        
    logger.info(f'test case no 3') 
    input_routes_base_dir = './mt_dir'
    input_geojson_file = './mt_dir/geodata_RTK.geojson'
 

    gps_geojson = R2G.load_geojson(input_geojson_file)
    gps_path_obj = R2G.get_gps_path(gps_geojson, read_footprint=True)
    gps_path_obj.expand_gps(num=100)
    gps_geojson = {'type': 'Feature',
            'geometry': {'type': 'LineString', 'coordinates': []},
            'properties': {'video': []}}
    
    geodata = []
    for gps in gps_path_obj.gps_path:
        geo_entry = [gps['coordinates'][0], \
                     gps['coordinates'][1], \
                     gps['coordinates'][2], \
                     gps['timestamp'],\
                     gps['height'],
                     0,0,0]
        geodata.append(geo_entry)
    gps_geojson['geometry']['coordinates'] = geodata
    output_geojson_file = 'mt_dir_expanded.geojson' 
    output_path_list = [output_geojson_file]
    for output_path in output_path_list:
        with open(output_path, 'w') as f:
            safe_json_dump(gps_geojson, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--test_case", required=True, help='tree_walk/mt_dir/expand_gps')
    input_vals = parser.parse_args()
    test_case = input_vals.test_case 
    test(0, test_case)
