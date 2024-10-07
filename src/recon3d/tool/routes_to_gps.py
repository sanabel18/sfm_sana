import os, sys
import fnmatch
import json
import numpy as np
import pyproj
import utm
from utils.transformation import ransac_similarity_transform, apply_transform, similarity_transform
from utils.safe_json import safe_json_dump
from utils.logger import get_logger, close_all_handlers
from point_filter.poly_fit_filter import PolyFitFilter
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import copy
import pandas as pd


R_SFM_TO_UTM = np.array([[1,0,0],[0,0,1],[0,-1,0]])
R_UTM_TO_SFM = np.array([[1,0,0],[0,0,-1],[0,1,0]])
ELLPS = 'WGS84'
DATUM = 'WGS84'
TIME_DELTA_TOLERANCE = 100 # in ms
DEFAULT_LOGGER = get_logger("routes_to_gps.log", '.')
GEO_ENTRIES = {
        'long': 0, # longitude
        'lat': 1,  # latitude
        'elev': 2, # elevation: height of camera from sea level
        'utc': 3,  # utc timestamp in milliseconds
        'height': 4, # in meter, height of camera from the local ground
        'alpha': 5, # babylon angle
        'beta': 6,  # babylon angle
        'gamma': 7  # babylon angle
                }
class SfM_path:
    '''
    route: list of dict that contains each frame in a route 
    
    {
        'timestamp': utc_timestamp, 
        'position': camera position, List[3]
        'footprint': projection of camera position on the ground, List[3]
        'rotation': camera orientation, List[3,3]
    }
    '''
 
    def __init__(self, timestamp, position, rotation ):
        sfm_path = []
        for ts, pos, rot in zip(timestamp, position, rotation):
            sfm_path.append(
                {
                    'timestamp': ts,
                    'position': pos,
                    'rotation': rot
                }
                )
        self.sfm_path = sfm_path
    
    def set_footprint(self, footprint):
        for frame, fp in zip(self.sfm_path, footprint):
            frame['footprint'] = fp
         
    def set_dummy_footprint(self):
        for frame in self.sfm_path:
            frame['footprint'] = [0,0,0]
 
    def set_footprint_w_stepsize(self, stepsize_fac):
        """
        set step-size anchor as footprint
        Args:
        stepsize_fac: float, the factor between step size and step-size anchor
        """
        stepsize_list = self.gen_stepsize_list(stepsize_fac)
        stepsize_anchor_list = self.gen_stepsize_anchor_list(stepsize_list)
        self.set_footprint(stepsize_anchor_list)

    def gen_stepsize_anchor_list(self, stepsize):
        """
        Args:
        stepsize: list of float with length of self.sfm_path, stepsize corresponding to each camera pose
        Retruns:
        stepsize_anchor_list: list of 1x3 list, each 1x3 list represent a 3D point
                              [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],...]  

        """
        stepsize_anchor_list = []
        for frame, ss in zip(self.sfm_path, stepsize):
            direction = np.array(frame['rotation'])[1, :]
            camera_position = np.array(frame['position'])
            stepsize_anchor = camera_position + ss*direction
            stepsize_anchor_list.append(stepsize_anchor.tolist())
        return stepsize_anchor_list

    def gen_stepsize_list(self, fac):
        """
        Generate step size from 3D camera positions in self.sfm_path
        then apply fac onto it
        Args:
        fac: float
        Returns:
        stepsize_list: list of float, scaled step size list
        """
        sfm_xyz = [frame['position'] for frame in self.sfm_path]
        stepsize = gen_stepsize(sfm_xyz)
        stepsize_list = [ss*fac for ss in stepsize]
        return stepsize_list

class GPS_path:
    '''
    gps_path: list of dict 
    
    {
        'coordinates': List[3]
                       [long, lat, elevation], 
        'timestamp': int 
                     utc_timestamp, 
        'height': float in meter, 
                  distance between camera and the ground 
        'footprint': List[3]
                     [long, lat, elevation-height]
                
    }
    '''
    def __init__(self, raw_gps_path, read_footprint=False):
        '''
        Args:
        the raw data information fetched from original gps geojson file
        list of dict 
        {
            'coordinates': List[3]
                           [long, lat, elevation], 
            'timestamp': int 
                         utc_timestamp, 
            'height': float in meter, 
                      distance between camera and the ground 
        }

        '''
        self.stepsize_fac = None
        self.gps_path = copy.deepcopy(raw_gps_path)
        if read_footprint:
            self._make_footprint()
        else:
            stepsize_anchor = self._gen_stepsize_anchor()
            self._make_footprint_w_stepsize(stepsize_anchor)
    
    def _make_footprint(self):
        """
        make up footprint by substracting height from its z-coordinate 
        """
        for gps_path_tmp in self.gps_path:
            gps_coords = copy.deepcopy(gps_path_tmp['coordinates'])
            gps_path_tmp['footprint'] = gps_coords
            gps_path_tmp['footprint'][2] -=  gps_path_tmp['height']
  

    def _make_footprint_w_stepsize(self, stepsize_anchor):
        """
        make up footprint by substracting stepsize anchor from its z-coordinate
        Args:
        stepsize_anchor: list of float, stores the distance between camera poses and footprint
        """
        for gps_path_tmp, ss_anchor in zip(self.gps_path, stepsize_anchor):
            gps_coords = copy.deepcopy(gps_path_tmp['coordinates'])
            gps_path_tmp['footprint'] = gps_coords
            gps_path_tmp['footprint'][2] -=  ss_anchor
  
    def _gen_stepsize_anchor(self):
        """
        Generate step size from GPS points
        1. convert GPS to utm then sfm coordinate
        2. find step size between 3D points in sfm coordinate
        3. calculate stepsize_fac, this factor will make stepsize close to 1.8
        4. apply stepsize_fac to step size to make step size anchor
        Returns:
        stepsize_anchor: list of step size anchor
        
        """
        gps_xyz_utmdata = [gps_to_xyz_utm(*gps_path_tmp['coordinates'][0:3]) for gps_path_tmp in self.gps_path]
        gps_xyz_utm = [data[0] for data in gps_xyz_utmdata]
        utm_data = [data[1] for data in gps_xyz_utmdata]
        gps_xyz_utm = np.array(gps_xyz_utm)
        # transform from utm -> sfm
        gps_xyz = apply_transform(gps_xyz_utm, 1, R_UTM_TO_SFM, [0,0,0], is_colvec=False)
        stepsize = gen_stepsize(gps_xyz)
        mean_stepsize = sum(stepsize)/len(stepsize)
        self.stepsize_fac = 1.8/mean_stepsize
        stepsize_anchor = [ss*self.stepsize_fac for ss in stepsize]
        return stepsize_anchor
    
    def get_stepsize_fac(self):
        return self.stepsize_fac
    
    def expand_gps(self, num=10):
        """
        expand gps path to a new one with point number = num
        intepolate with timestamp generated with start and end timestamp from 
        original gps path
        update to self.gps_path
        Args:
        num: int, number of expanded gps path
        """
        gps_timestamp = [gps_pt['timestamp'] for gps_pt in self.gps_path]
        gps_coords = [gps_pt['coordinates'] for gps_pt in self.gps_path]
        gps_height = [gps_pt['height'] for gps_pt in self.gps_path]
        gps_footprint = [gps_pt['footprint'] for gps_pt in self.gps_path]
        ts_interp = np.linspace(gps_timestamp[0],gps_timestamp[-1],num)
        gps_coords_interp = \
                interpolate_position(gps_coords, gps_timestamp, ts_interp)
        gps_height_interp = np.interp(ts_interp, gps_timestamp, gps_height)
        gps_footprint_interp = \
                interpolate_position(gps_footprint, gps_timestamp, ts_interp)
        # set to self.gps_path
        self.gps_path = []
        for ts, coords, height, fp in \
                zip(ts_interp, gps_coords_interp, gps_height_interp, gps_footprint_interp): 
            gps_dict = {
                    'coordinates': coords,
                    'timestamp': ts,
                    'height': height,
                    'footprint': fp
                    }
            self.gps_path.append(gps_dict)

class Matched_SfM_GPS_AnchorPair:
    ''' 
    paired data from SfM_path and GPS_path
    camera_position and camera_footprint are from SfM_path and are in SfM coordinate system
    gps_coordinates and gps_footprint are from GPS_path and are in GPS cooridnate system
    anchor_group_labels marks anchor group of each SfM_path-GPS_path pair
        "start" marks the group of the starting region of SfM path
        "end" marks the gourp of the ending region of SfM path
    gps_anchor_index is the index of the paired-GPS_path from its original GPS_path 

    Args:
    camera_position: List[N,3]
                     list of [x,y,z]
                     camera postion as in SfM_path
    camera_footprint: List[N,3]
                      list of [x,y,z]
                      camera footprint as in SfM_path
    gps_coordinates: List[N,3]
                     list of [lon, lat, elevation]
                     gps cooridante as in GPS_path
    gps_footprint: List[N,3]
                   list of [lon, lat, elevation]
                   gps footprint as defined in GPS_path
    anchor_group_labels: List[N]
                         list of str: ['start','start',....,'end','end','end']
                         
    gps_anchor_index: List[N]
                      list of int
                      the index of gps_coordinates/gps_footprint from gps_path
    gps_path = GPS_path obj
    '''
    def __init__(self,
                 camera_position,
                 camera_footprint,
                 gps_coordinates,
                 gps_footprint,
                 anchor_group_label,
                 gps_anchor_index,
                 gps_path
                ):
        self.camera_position = camera_position
        self.camera_footprint = camera_footprint
        self.gps_coordinates = gps_coordinates
        self.gps_footprint = gps_footprint
        self.anchor_group_label = anchor_group_label
        self.gps_anchor_index = gps_anchor_index
        self.gps_path = gps_path
    def set_camera_Y(self, camera_position_y_smoothed, 
                           camera_footprint_y_smoothed):
        '''
        set Y-component(2nd-component) of position and footprint to smoothed values
        Args:
        camera_position_y_smoothed: List[N]
                                    list of float 
        camera_footprint_y_smoothed: List[N]
                                     list of float 
        '''
        for pos, pos_y_smoothed in zip(self.camera_position, camera_position_y_smoothed):
            pos[1] = pos_y_smoothed
        for fp, fp_y_smoothed in zip(self.camera_footprint, camera_footprint_y_smoothed):
            fp[1] = fp_y_smoothed

    def set_elevation(self, gps_coordinates_elevation_smoothed, 
                                    gps_footprint_elevation_smoothed):
        '''
        set 3rd component of gps_coordinates and gps_footprint_elevation to smoothed values
        gps_coordinates_elevation_smoothed: List[N]
                                            list of float 
        gps_footprint_elevation_smoothed: List[N]
                                          list of float 
        '''
        for gps, gps_elev_smoothed in zip(self.gps_coordinates, gps_coordinates_elevation_smoothed):
            gps[2] = gps_elev_smoothed
        for gps_fp, gps_fp_elev_smoothed in zip(self.gps_footprint, gps_footprint_elevation_smoothed):
            gps_fp[2] = gps_fp_elev_smoothed

def gen_sfm_path_stepsize_anchor(sfm_path_obj, gps_path_obj):
    """

    """
    stepsize_fac = gps_path_obj.get_stepsize_fac()
    sfm_path_obj_interp = interp_sfm_path_w_gps_path(\
                                sfm_path_obj.sfm_path,\
                                gps_path_obj.gps_path)
    sfm_path_obj_interp.set_footprint_w_stepsize(stepsize_fac)
    sfm_path_interp = sfm_path_obj_interp.sfm_path
    return sfm_path_interp

def gen_stepsize(xyz_list):
    """
    from a series of 3D points, calculate the distance between
    current one and the next one (we called step size here)
    o o o o o   points 
     - - - -    step sizes
    for the last point, we assign the step size calculated from it previous point

    Args:
    xyz_list: [N,3] list that contains xyz of N 3D points
    Retruns: list of [N] list of step size 
    """
    xyz_arr = np.array(xyz_list)
    #reduce xyz
    reduced_xyz_arr = reduce_xyz(xyz_arr, 'y')
    xyz_dist = reduced_xyz_arr[1:,:] - reduced_xyz_arr[0:-1,:]
    last_row = np.array([xyz_dist[-1,:]])
    extented_xyz_dist = np.append(xyz_dist, last_row,0)
    stepsize = np.linalg.norm(extented_xyz_dist, axis=1)
    return stepsize.tolist()


def interp_sfm_path_w_gps_path(sfm_path, gps_path):
    '''
    interpolate sfm_path with timestamp from gps_path, and save to 
    new SfM_path object
    1. interpolate position
    2. interpolate rotation
    Args:
    sfm_path: SfM_path.sfm_path
    gps_path: GPS_path.gps_path
    Return:
    SfM_path
    '''
    sfm_timestamp = [frame['timestamp'] for frame in sfm_path]
    sfm_position = [frame['position'] for frame in sfm_path]
    sfm_rotation = [frame['rotation'] for frame in sfm_path]
    gps_timestamp = [gps_pt['timestamp'] for gps_pt in gps_path]
    gps_start_time = gps_timestamp[0]
    gps_ts_rel = sorted([(gps_ts - gps_start_time)*1e-3 for gps_ts in gps_timestamp])
    gps_ts_rel_cropped = crop_ts_interp(sfm_timestamp, gps_ts_rel)
    sfm_position_interp = interpolate_position(sfm_position, \
                                                sfm_timestamp, \
                                                gps_ts_rel_cropped)
    sfm_rotation_interp = interpolate_rotation(sfm_rotation, \
                                                sfm_timestamp, \
                                                gps_ts_rel_cropped)
    #create intepolated sfm_path
    sfm_path_interp = SfM_path(gps_ts_rel, sfm_position_interp, sfm_rotation_interp)
    return sfm_path_interp

def crop_ts_interp(ts, ts_interp):
    """
    crop ts_interp to be consistant with the start and end time of ts
    Args:
    ts: list of float
    ts_interp: list of float
    Retruns:
    ts_interp_cropped: list of float
    """
    max_t = max(ts)
    min_t = min(ts)
    ts_interp_cropped = [t for t in ts_interp if t <= max_t and t >= min_t]
    return ts_interp_cropped

def interpolate_rotation(rotation, timestamp, timestamp_interp):
    """
    interpolate rotation parametrized with timestamp to timestamp_interp
    use slerp to do it
    Args:
    rotation: list of N [3x3] list
    timestamp: list of N float
    timestamp_interp: list of M float, timestamp to be interpolate to
    Returns list of M [3x3] list
    """
    R_obj_rotation = Rotation.from_matrix(rotation)
    slerp_obj = Slerp(timestamp, R_obj_rotation)
    rotation_interp_obj = slerp_obj(timestamp_interp)
    rotation_interp = rotation_interp_obj.as_matrix()
    return rotation_interp

def interpolate_position(position, timestamp, timestamp_interp):
    """
    interpolate position parametrized with timestamp to timestamp_interp
    Args:
    position: list of N [3,1] list
    timestamp: list of N float
    timestamp_interp: list of M float
    Returns:
    list of M [3,1] list
    """
    position_T = np.array(position).transpose()
    x = position_T[0]
    y = position_T[1]
    z = position_T[2]
    x_interp = np.interp(timestamp_interp, timestamp, x) 
    y_interp = np.interp(timestamp_interp, timestamp, y) 
    z_interp = np.interp(timestamp_interp, timestamp, z) 
    position_interp = np.stack((x_interp, y_interp, z_interp), axis=0)
    position_interp_list = position_interp.transpose().tolist()
    return position_interp_list
 
def gps_to_xyz(lon, lat, alt):
    transformer = pyproj.Transformer.from_crs(
        {"proj":'latlong', "ellps":ELLPS, "datum":DATUM},
        {"proj":'geocent', "ellps":ELLPS, "datum":DATUM},
    )
    
    x, y, z = transformer.transform(lon, lat, alt, radians=False)
    return (x, y, z)


def xyz_to_gps(x, y, z):
    transformer = pyproj.Transformer.from_crs(
        {"proj":'geocent', "ellps":ELLPS, "datum":DATUM},
        {"proj":'latlong', "ellps":ELLPS, "datum":DATUM},
    )
    lon, lat, alt = transformer.transform(x, y, z, radians=False)
    return (lon, lat, alt)

def gps_to_xyz_utm(lon, lat, alt):
    utm_data = utm.from_latlon(lat, lon)
    utm_xy = np.array(utm_data[:2])
    x = utm_xy[0]
    y = utm_xy[1]
    z = alt
    return (x, y, z), utm_data

def xyz_to_gps_utm(x, y, z, utm_data):
    gps = utm.to_latlon(x, y, utm_data[2], utm_data[3])
    return (gps[1], gps[0], z)

def get_sfm_path(base_dir, read_footprint=False):
    '''
    collect frame information from folder of slices (data_*_000, data_*_001...etc.) 
    to construct SfM_path objects.
    Args: directory contains data_* folder of slices
    Retrun: SfM_path
    '''
    data_dir_list = fnmatch.filter(os.listdir(base_dir), 'data_*')
    full_data_dir_list = []
    for data_dir in data_dir_list:
        full_data_dir_list.append(os.path.join(base_dir, data_dir))
    sfm_path = get_sfm_path_from_data_dir_list(full_data_dir_list, read_footprint=read_footprint)
    return sfm_path

def get_sfm_path_from_data_dir_list(data_dir_list, read_footprint=False):
    '''
    collect frame information from folder of slices (data_*_000, data_*_001...etc.) 
    to construct SfM_path objects.
    Args: directory contains data_* folder of slices
    Retrun: SfM_path
    '''
    data_dir_list = sorted(data_dir_list)

    frames = []
    timestamp = []
    position = []
    footprint = []
    rotation = []
    for data_dir in data_dir_list:
        with open(os.path.join(data_dir, 'routes.json'), 'r') as f:
            one_slice = json.load(f)
        frames.extend(one_slice['frames'])
    frames = sorted(frames, key=lambda k: k['timestamp'])
    for frame in frames:
        timestamp.append(frame['timestamp'])
        position.append(frame['position'])
        if read_footprint:
            footprint.append(frame['footprint'])
        rotation.append(frame['rotation'])
    sfm_path = SfM_path(timestamp, position, rotation)
    if read_footprint:
        sfm_path.set_footprint(footprint)
    return sfm_path


def get_gps_geometry(geojson):
    '''
    Args: dict loaded from geojson
    Return: list of list that contains geo_entries
            [[long, lat, elev, utc...],
            [long, lat, elev, utc...],
            [long, lat, elev, utc...],...]
    '''
    return geojson['geometry']['coordinates']

def load_geojson(geojson_file):
    '''
    the loaded geojson dict might be one of these 
    two format:
    ============
    Format1
    {
        "type":"Feature",
        "geometry" {
            "type":"LineString"
            "coordinates": []
            }
        "properties":{
            "AbsoluteUtcMilliSec": []
            "RelativeMilliSec":    []
        }

    }
    coordinates stores list of [longitude, latitude, elevation]
    AbsoluteUtcMilliSec stores list of UTC timestamp in millisec
    ===========
    Fromat2
    { 
        "type":"Feature",
        "geometry" {
            "type":"LineString"
            "coordinates": []
            }
        "properties":{}
    }
    coordinates stores list of GEO_DATA
    GEO_DATA = 
    [
        longitude,
        latitude,
        elevation,  height of camera from sea level
        timestamp,  utc timestamp in milliseconds
        height,     height of camera from the local ground in meter
        alpha,      babylon angle in radian
        beta,       babylon angle in radian
        gamma,      babylon angle in radian
    ]
 
    '''
    with open(geojson_file, 'r') as f:
        geojson = json.load(f) 
    return geojson

def update_geodata(geojson, geodata):
    '''
    Args: 
    geojson: 
        dict that stores gps info
    { 
        "type":"Feature",
        "geometry" {
            "type":"LineString"
            "coordinates": []
            }
        "properties":{}
    }
    geodata:
        list of GEO_DATA
    GEO_DATA = 
    [
        longitude,
        latitude,
        elevation,  height of camera from sea level
        timestamp,  utc timestamp in milliseconds
        height,     height of camera from the local ground in meter
        alpha,      babylon angle in radian
        beta,       babylon angle in radian
        gamma,      babylon angle in radian
    ]
 
    '''
    geojson['geometry']['coordinates'] = geodata
    return geojson


def get_gps_path(geojson, read_footprint):
    '''
    get lat, lon, elevation, height and utc timestamp from GPS or RTK_GPS.
    there are two kind of format of GPS.geojson:
    
    ============
    Format1
    {
        "type":"Feature",
        "geometry" {
            "type":"LineString"
            "coordinates": []
            }
        "properties":{
            "AbsoluteUtcMilliSec": []
            "RelativeMilliSec":    []
        }

    }
    coordinates stores list of [longitude, latitude, elevation]
    AbsoluteUtcMilliSec stores list of UTC timestamp in millisec
    if we got gps with Format1, a nother attribute 'height' will
    be added and set value with 1.8.
    It stands for height of camera from the local ground in meter, 
    and this number will be a resonable choice if the 360 video was 
    taken by a male adult with stablized holder.

    ===========
    Fromat2
    { 
        "type":"Feature",
        "geometry" {
            "type":"LineString"
            "coordinates": []
            }
        "properties":{}
    }
    coordinates stores list of GEO_DATA
    GEO_DATA = 
    [
        longitude,
        latitude,
        elevation,  height of camera from sea level
        timestamp,  utc timestamp in milliseconds
        height,     height of camera from the local ground in meter
        alpha,      babylon angle in radian
        beta,       babylon angle in radian
        gamma,      babylon angle in radian
    ]
 
    
    Args: dict loaded from geojson
    Retrun: GPS_path
    '''
    is_GPS_Format1 = len(geojson['geometry']['coordinates'][0]) < 4
    if is_GPS_Format1:
        raw_gps_path = []
        for coordinates, timestamp in zip(geojson['geometry']['coordinates'],
                                          geojson['properties']['AbsoluteUtcMilliSec']):
            frame = {
                'coordinates': coordinates, 
                'timestamp': timestamp,
                'height': 1.8
            } 
            raw_gps_path.append(frame)

    else:
        raw_gps_path = []
        for coordinates in geojson['geometry']['coordinates']: 
            frame = {
                'coordinates': coordinates[0:3], 
                'timestamp': coordinates[GEO_ENTRIES['utc']],
                'height': coordinates[GEO_ENTRIES['height']],
            } 
            raw_gps_path.append(frame)
    
    gpspath_obj = GPS_path(raw_gps_path, read_footprint=read_footprint)
    return gpspath_obj

def get_sfm_path_xyz(sfm_path):
    '''
    Args: SfM_path.sfm_path
    Return:
        List[N,3]
        list of routes position xyz
        [[x,y,z],[x,y,z]...]
    '''
    sfm_path_xyz = [frame['position'] for frame in sfm_path]
    return sfm_path_xyz

def filter_sfm_path(sfm_path):
    '''
    Remove outlier in routes if exist any.
    Args: SfM_path.sfm_path 
    Return: SfM_path.sfm_path
    '''
    sfm_path_xyz = get_sfm_path_xyz(sfm_path)
    poly_degree = 5
    max_iter = 10
    polyfit = PolyFitFilter(poly_degree, max_iter)
    valid_index = polyfit.filter(sfm_path_xyz)
    cleaned_sfm_path = [sfm_path_tmp for idx, sfm_path_tmp in enumerate(sfm_path) if idx in valid_index] 
    return cleaned_sfm_path

def get_start_microsec_from_gps_path(gps_path):
    start_microsec = gps_path[0]['timestamp']
    return start_microsec

def find_sfm_gps_anchor_pair(sfm_path, gps_path, anchor_pts_number):
    '''
    try to find match between these two sets of data:
        -position: [x, y, z] in SfM coordinate system
        -gps_coordinate: [lat, lon, elevation]

    matched pair was found by matching their correspoding utc timestamp, 
    if the time difference is small than 100 ms, it will be detected as a matched pair
    Args: 
    SfM_path.sfm_path, 
    GPS_path.gps_path, 
    anchor_pts_number: int
        number of anchor points from one end of sfm_path. 
        The total number of anchor points will be 2*anchor_pts_number at most
        (the repeated ones will be removed in reduce_to_one2one_pair())
    Return: 
    Matched_SfM_GPS_AnchorPair 
    start_microsec: int
        utc timestamp of first point in gps_path
    matched_pair_time_difference_valid: List[N]
        list of float 
        N is the the number of matched sfm gps-anchor pair detected
    '''
    start_microsec = gps_path[0]['timestamp']
    sfm_path_timestamp = np.array([frame['timestamp']* 1e3 + start_microsec for frame in sfm_path])
    anchor_start = [*range(anchor_pts_number)]
    anchor_end = [*range(len(sfm_path_timestamp)-anchor_pts_number, len(sfm_path_timestamp), 1)]
    anchor = anchor_start + anchor_end
    anchor_group_label = ['start']*len(anchor_start) + ['end']*len(anchor_end)
    
    # If total number of anchor exceeds the lentgh of sfm_path, the grouping of 'start', 'end'
    # get merged to 'start' only
    if (len(anchor) > len(sfm_path_timestamp)):
        sfm_path_anchor_timestamp = sfm_path_timestamp
        anchor_label = ['start']*len(sfm_path_timestamp)
        anchor = range(0,len(sfm_path_timestamp),1)
    else:
        sfm_path_anchor_timestamp = sfm_path_timestamp[anchor]


    sfm_path_anchor = np.array(sfm_path)[anchor]
    gps_path_timestamp = np.array([gps['timestamp'] for gps in gps_path])

    time_difference = np.abs(sfm_path_anchor_timestamp.reshape((-1, 1)) - gps_path_timestamp.reshape((1, -1)))
    gps_anchor_index_list = np.argmin(time_difference ,axis=1)
    valid_list = time_difference[np.arange(time_difference.shape[0]), gps_anchor_index_list] < TIME_DELTA_TOLERANCE
    matched_pair_time_difference_list = time_difference[np.arange(time_difference.shape[0]), gps_anchor_index_list]
    
    position_valid = []
    footprint_valid = []
    gps_coordinates_valid = []
    gps_footprint_valid = []
    gps_anchor_index_valid = []
    matched_pair_time_difference_valid = []
    anchor_group_label_valid = []
    for i, (gps_anchor_index, valid) in enumerate(zip(gps_anchor_index_list, valid_list)):
        if not valid:
            continue
        position_valid.append(sfm_path_anchor[i]['position'])
        footprint_valid.append(sfm_path_anchor[i]['footprint'])
        gps_coordinates_valid.append(gps_path[gps_anchor_index]['coordinates'])
        gps_footprint_valid.append(gps_path[gps_anchor_index]['footprint'])
        gps_anchor_index_valid.append(gps_anchor_index)
        matched_pair_time_difference_valid.append(matched_pair_time_difference_list[i])
        anchor_group_label_valid.append(anchor_group_label[i])
    
    matched_sfm_gps_anchor_pair = Matched_SfM_GPS_AnchorPair(position_valid,            
                                                              footprint_valid, 
                                                              gps_coordinates_valid, 
                                                              gps_footprint_valid,
                                                              anchor_group_label_valid,
                                                              gps_anchor_index_valid,
                                                              gps_path
                                                              )   
    
    return matched_sfm_gps_anchor_pair, matched_pair_time_difference_valid 

def get_gps_anchor_index(matched_sfm_gps_anchor_pair):
    '''
    get start and end index of anchor region
    '''
    gps_anchor_index  = matched_sfm_gps_anchor_pair.gps_anchor_index 
    return min(gps_anchor_index), max(gps_anchor_index)

def reduce_to_one2one_pair(matched_sfm_gps_anchor_pair, 
                           matched_pair_time_difference):
    '''
    we might have many-to-one matched pair,
    because the pts from SfM routes is much denser than that from GPS data.
    from many-to-one pairs we only keep the one with smallest time distance. 
    Args:
    Matched_SfM_GPS_AnchorPair
    matched_pair_time_difference: List[N]
        list of float 
        N is the the number of matched sfm gps-anchor pair detected
    Return: 
    Matched_SfM_GPS_AnchorPair
    '''
    match_dict_full = vars(matched_sfm_gps_anchor_pair)
    match_dict = copy.deepcopy(match_dict_full)
    match_dict.pop('gps_path')
    match_dict['match_pair_time_difference'] = matched_pair_time_difference

    
    df = pd.DataFrame(data = match_dict)
    g = df.groupby('gps_anchor_index')
    reduced = df.loc[g.match_pair_time_difference.idxmin()]
    reduced_dict = reduced.to_dict('list') 
    
    reduced_matched_sfm_gps_anchor_pair = Matched_SfM_GPS_AnchorPair(reduced_dict['camera_position'],            
                                      reduced_dict['camera_footprint'],
                                      reduced_dict['gps_coordinates'],
                                      reduced_dict['gps_footprint'],
                                      reduced_dict['anchor_group_label'],
                                      reduced_dict['gps_anchor_index'],
                                      match_dict_full['gps_path']
                                    )



    return reduced_matched_sfm_gps_anchor_pair

def smoothen_horizontal_area(matched_sfm_gps_anchor_pair, logger=DEFAULT_LOGGER, smooth='anchor'): 
    '''
    We have two anchor regions from matched_sfm_gps_anchor_pair. 
    the elevation of those resgions can be smoothened if one believe those regions represent flat areas.
    Args: 
    Matched_SfM_GPS_AnchorPair,
    logger: logger object
    smooth: str
        'anchor': within each anchor regions, the elevation gets assigned to their mean values.
        'full': for all anchor regions, the elevation gets assigned to the their overall mean value.
    Return: Matched_SfM_GPS_AnchorPair
            the Y-component in camera_position/camera_footprint  and 
            elevation in gps_coordinates/gps_footprint have been replaced with smoothed value 
        
    '''
    matched_sfm_gps_anchor_pair = copy.deepcopy(matched_sfm_gps_anchor_pair)
    match_dict = {'camera_position': [], 
                  'camera_footprint':[], 
                  'gps_coordinates':[],
                  'gps_footprint':[],
                  'anchor_group_label':[],
                  'elevation':[],
                  'footprint_elevation':[],
                  'position_Y':[],
                  'footprint_Y':[]
                  } 

   
    match_dict['camera_position'] = matched_sfm_gps_anchor_pair.camera_position
    match_dict['camera_footprint'] = matched_sfm_gps_anchor_pair.camera_footprint
    match_dict['gps_coordinates'] = matched_sfm_gps_anchor_pair.gps_coordinates
    match_dict['gps_footprint'] = matched_sfm_gps_anchor_pair.gps_footprint
    match_dict['anchor_group_label'] = matched_sfm_gps_anchor_pair.anchor_group_label
    match_dict['elevation'] = [gps_coord[2] for gps_coord in  matched_sfm_gps_anchor_pair.gps_coordinates]
    match_dict['footprint_elevation'] = [gps_fpt[2] for gps_fpt in matched_sfm_gps_anchor_pair.gps_footprint]
    match_dict['position_Y'] = [pos[1] for pos in matched_sfm_gps_anchor_pair.camera_position]
    match_dict['footprint_Y'] = [fpt[1] for fpt in matched_sfm_gps_anchor_pair.camera_footprint]
 
    
    #group by anchor_label
    df = pd.DataFrame(data = match_dict)
    label = df['anchor_group_label'].to_frame()
    
    if smooth == 'anchor':
        elev_mean = label.merge(
                df.groupby('anchor_group_label')['elevation'].mean().reset_index(
                name="elevation_mean"),            
                on='anchor_group_label')[['elevation_mean']]
        elev_footprint_mean = label.merge(
                df.groupby('anchor_group_label')['footprint_elevation'].mean().reset_index(
                name="footprint_elevation_mean"),
                on='anchor_group_label')[['footprint_elevation_mean']]
        position_Y_mean = label.merge(
                df.groupby('anchor_group_label')['position_Y'].mean().reset_index(
                name="position_Y_mean"),
                on='anchor_group_label')[['position_Y_mean']]
        footprint_Y_mean = label.merge(
                df.groupby('anchor_group_label')['footprint_Y'].mean().reset_index(
                name="footprint_Y_mean"),
                on='anchor_group_label')[['footprint_Y_mean']]
       
        df_tmp = [df, elev_mean, elev_footprint_mean, position_Y_mean, footprint_Y_mean]
        df_extend = pd.concat(df_tmp,axis=1)
    elif smooth == 'full':
        elev_mean = df['elevation'].mean()
        elev_footprint_mean = df['footprint_elevation'].mean()
        position_Y_mean = df['position_Y'].mean()
        footprint_Y_mean = df['footprint_Y'].mean()
        
        df_tmp = [df,
                  pd.Series([elev_mean for i in range(len(df['elevation']))], name='elevation_mean'),
                  pd.Series([elev_footprint_mean for i in range(len(df['footprint_elevation']))],
                            name='footprint_elevation_mean'),
                  pd.Series([position_Y_mean for i in range(len(df['position_Y']))],
                            name='position_Y_mean'),
                  pd.Series([footprint_Y_mean for i in range(len(df['footprint_Y']))],
                             name='footprint_Y_mean')
                ]
        df_extend = pd.concat(df_tmp, axis=1)
    else:
        e = 'smooth type {} is not a valid type, should be anchor or full'.format(smooth)
        logger.error(e)
        raise ValueError(e)
    
    smoothed_dict = df_extend.to_dict('list')
    
    
    matched_sfm_gps_anchor_pair.set_camera_Y(
        smoothed_dict['position_Y_mean'],
        smoothed_dict['footprint_Y_mean']) 
    
    
    
    matched_sfm_gps_anchor_pair.set_elevation(
        smoothed_dict['elevation_mean'],
        smoothed_dict['footprint_elevation_mean'])
    
    smoothed_matched_sfm_gps_anchor_pair = matched_sfm_gps_anchor_pair
    return smoothed_matched_sfm_gps_anchor_pair


def export_sfm_gps_match(matched_sfm_gps_anchor_pair, smooth=True, footprint=True):
    '''
    prepare the final match list.
    one can choose if smoothened result should be exported. 
    one can choose if footprint should be exported.
    Args: Matched_SfM_GPS_AnchorPair 
    Retrun:
    list of sfm-gps match pair
            dict{
                position:[x,y,z]
                coordinates:[long,lat,elev]
            }
 
    '''

    
    sfm_gps_match = []
    for i in range(len(matched_sfm_gps_anchor_pair.camera_position)): 
        sfm_gps_match.append(
            {
                'position': matched_sfm_gps_anchor_pair.camera_position[i],
                'coordinates': matched_sfm_gps_anchor_pair.gps_coordinates[i]
            }
        )
        if footprint:
            sfm_gps_match.append(
                {
                    'position': matched_sfm_gps_anchor_pair.camera_footprint[i],
                    'coordinates': matched_sfm_gps_anchor_pair.gps_footprint[i]
                }
            )

    return sfm_gps_match

def convert_match_list_into_sfm_coord(sfm_gps_match):
    '''
    fetch sfm_xyz from sfm_gps_match
    fetch gps_coordinates from sfm_gps_match
    transform gps_coordinates: first to utm, then to sfm.
    
    Args: 
        list of sfm_gps_match dict
            dict{
                position: List[3]
                          [x, y, z]
                coordinates: List[3]
                             [long, lat, elev]
            }


    Return: 
        sfm_xyz: array(N,3) 
        gps_xyz: array(N,3)
        utm_data: List of tuples (utm_east, utm_north, utm_zone, utm_letter)
                  ex: [(276778.07023962133, 2602132.268205515, 51, 'Q')]
    
    '''
    sfm_xyz = [match['position'] for match in sfm_gps_match]
    gps_xyz_utmdata = [gps_to_xyz_utm(*match['coordinates'][0:3]) for match in sfm_gps_match]
    gps_xyz_utm = [data[0] for data in gps_xyz_utmdata]
    utm_data = [data[1] for data in gps_xyz_utmdata]
    sfm_xyz = np.array(sfm_xyz)
    gps_xyz_utm = np.array(gps_xyz_utm)
    # transform from utm -> sfm
    gps_xyz = apply_transform(gps_xyz_utm, 1, R_UTM_TO_SFM, [0,0,0], is_colvec=False)
    return sfm_xyz, gps_xyz, utm_data

def get_matched_transformation(xyz_from, xyz_to):
    '''
    we use similarity transformation to find corresponding tranfromation between xyz_from and xyz_to
    xyz_to = c*np.matmul(R,xyz_from) + t
    c: scale
    r: 3x3 rotation matrix 
    t: translation

    Args: array(N,3), array(N,3)
    Return: 
    c: float. scale
    r: list[3,3]. 3x3 rotation matrix 
    t: list[3]. translation
    '''
    c, R, t = similarity_transform(xyz_from, xyz_to, is_colvec=False)
    return c, R, t

def trf_orientation_2_babylon(rotation: list) -> list:
    '''
    1) Only the orientation need to be in Babylon (left-handed) system. 
    2) The output will be used to rotate dome, so transpose is needed.
    3) Its sequence has to be yaw->pitch->row (i.e. y-x-z) in Babylon.
    4) The output is [alpha, beta, gamma].
    Args:
    list of List[3,3]
    Return:
    List[3]
    '''
        
    trf_mat = np.array([[1,0,0], [0,-1,0], [0,0,1]])
    rot_mat = np.array(rotation)
    new_rot_mat = np.matmul(trf_mat, np.matmul(rot_mat, trf_mat.transpose()))
    new_rot_mat_inv = new_rot_mat.transpose()  # Inverse works for dome-rotating babylon frontend implementation
    new_rot_euler_inv = Rotation.from_matrix(new_rot_mat_inv).as_euler('yxz', degrees=False)
    return [new_rot_euler_inv[1], new_rot_euler_inv[0], new_rot_euler_inv[2]]


def combine_geometry(sfm_start, sfm_end, gps_geometry, sfm_geometry):
    '''
    combine lists of  7 component-list with indexing defined in GEO_ENTRIES
    GEO_DATA = 
    [
        longitude,
        latitude,
        elevation,  height of camera from sea level
        timestamp,  utc timestamp in milliseconds
        height,     height of camera from the local ground in meter
        alpha,      babylon angle in radian
        beta,       babylon angle in radian
        gamma,      babylon angle in radian
    ]
 
    Args:
    sfm_start: index on gps_path where sfm_path starts
    sfm_end: index on gps_path where sfm_path ends
    gps_geometry: list of GEO_DATA from gps_path
    sfm_geometry: list of GEO_DATA from sfm_path
    Return:
    list of GEO_DATA
    '''
    gps_head = gps_geometry[:sfm_start]
    gps_tail = gps_geometry[sfm_end+1:]

    geodata = gps_head + sfm_geometry + gps_tail
    return geodata

def apply_trf_to_sfm_path(sfm_path, c, Rmat, t):
    '''
    apply physical tranfromation 
    c: Scale
    R: 3X3 rotation matrix
    t: translation  
    within sfm coordinate system
    
    Args: 
    SfM_path.sfm_path
    c: float. scale 
    R: List[3,3]. 3X3 rotation matrix
    t: List[3]. translation
    Return: SfM_path.sfm_path
    '''

    transformed_position = []
    transformed_footprint = []
    transformed_rotation = []
    timestamp = []
    if_footprint = False
    for frame in sfm_path:
        # apply physical transfromation in sfm
        position_sfm = frame['position']
        transformed_position_sfm = apply_transform(position_sfm, c, Rmat, t, is_colvec=False)
        transformed_position_sfm = np.squeeze(transformed_position_sfm)
        
        if 'footprint' in frame:
            if frame['footprint'] != None:
                if_footprint = True
        if if_footprint:
            footprint_sfm = frame['footprint']
            transformed_footprint_sfm = apply_transform(footprint_sfm, c, Rmat, t, is_colvec=False)
            transformed_footprint_sfm = np.squeeze(transformed_footprint_sfm)
        
        rotation = np.array(frame['rotation'])
        transformed_rotation_sfm = np.matmul(rotation, Rmat.transpose())

        transformed_position.append(transformed_position_sfm.tolist())
        if if_footprint:
            transformed_footprint.append(transformed_footprint_sfm.tolist())
        transformed_rotation.append(transformed_rotation_sfm.tolist())
        timestamp.append(frame['timestamp'])

    transformed_sfm_path = SfM_path(timestamp, transformed_position, 
                                    transformed_rotation)
    if if_footprint:
        transformed_sfm_path.set_footprint(transformed_footprint)
    return transformed_sfm_path.sfm_path

def transform_sfm_path_sfm2utm(sfm_path):
    '''
    apply coordinate transformation sfm -> utm
    Args: SfM_path.sfm_path
    Return: SfM_path.sfm_path
    '''
    position = []
    footprint = []
    rotation = []
    timestamp = []
    if_footprint = False
    for frame in sfm_path:
        position_sfm = frame['position']
        transformed_position_utm = apply_transform(position_sfm, 1, R_SFM_TO_UTM, [0,0,0], is_colvec=False)
        transformed_position_utm = np.squeeze(transformed_position_utm)
        if 'footprint' in frame:
            if_footprint = True
        if if_footprint:
            footprint_sfm = frame['footprint']
            transformed_footprint_utm = apply_transform(footprint_sfm, 1, R_SFM_TO_UTM, [0,0,0], is_colvec=False)
            transformed_footprint_utm = np.squeeze(transformed_footprint_utm)
        
        position.append(transformed_position_utm.tolist())
        if if_footprint:
            footprint.append(transformed_footprint_utm.tolist())
        rotation.append(frame['rotation'])
        timestamp.append(frame['timestamp'])

    transformed_sfm_path = SfM_path(timestamp, position, rotation)
    if if_footprint:
        transformed_sfm_path.set_footprint(footprint)
    return transformed_sfm_path.sfm_path

def get_rotations_from_sfm_path(sfm_path) -> list:
    '''
    Args: SfM_path
    Retrun: list of rotation List[3,3]
    '''
    rotation_list =[]
    for frame in sfm_path:
        rotation = frame['rotation']
        rotation_list.append(rotation)
    return rotation_list

def transform_rotation_to_babylon_angles(rotation_list: list) -> list:
    '''
    Args: list of rotation List[3,3]
    return: list of angles_list List[3]
    '''
    babylon_angles_list = []
    for rotation in rotation_list:
        babylon_angles = trf_orientation_2_babylon(rotation)
        babylon_angles_list.append(babylon_angles)
    return babylon_angles_list


def make_geo_entries(sfm_path:list, babylon_angles: list, start_microsec: int, utm_data:list ) -> list:
    '''
    make a list of  7 component-list with indexing defined in GEO_ENTRIES
    GEO_DATA = 
    [
        longitude,
        latitude,
        elevation,  height of camera from sea level
        timestamp,  utc timestamp in milliseconds
        height,     height of camera from the local ground in meter
        alpha,      babylon angle in radian
        beta,       babylon angle in radian
        gamma,      babylon angle in radian
    ]
    
    Args:
    SfM_path,
    list of List[3]
    start_microsec: int
                    first timestamp in gps_path
    utm_data: List of tuples (utm_east, utm_north, utm_zone, utm_letter)
                  ex: [(276778.07023962133, 2602132.268205515, 51, 'Q')]
    

    Returns:
    list of GEO_DATA
        
    '''
    gps_coord_list = []
    for frame, babylon_euler_angles in zip(sfm_path, babylon_angles):
        timestamp = frame['timestamp']
        abs_time_ms = start_microsec + timestamp * 1e3
        if 'footprint' in frame:
            height = np.linalg.norm(np.array(frame['position']) - np.array(frame['footprint']))
        else:
            height = 1.8
        position = frame['position']
        gps_coord = list(xyz_to_gps_utm(*position, utm_data)) + [abs_time_ms] + [height] + babylon_euler_angles
        gps_coord_list.append(gps_coord)
    return gps_coord_list

def reduce_xyz(xyz, remove_axis):
    '''
    reduce dimensionality of array(N,3) to array(N,2) along remove_axis
    Args: array(N,3), str
    Return: array(N,2)
    '''
    if remove_axis == 'x':
        xyz_reduced = xyz[:,[1,2]]
    elif remove_axis == 'y':
        xyz_reduced = xyz[:,[0,2]]
    elif remove_axis == 'z':
        xyz_reduced = xyz[:,[0,1]]
    
    return xyz_reduced

def get_sub_xyz(xyz, gps_xyz,remove_axis):
    sfm_xyz_sub = remove_axis_xyz(sfm_xyz, remove_axis)
    gps_xyz_sub = remove_axis_xyz(gps_xyz, remove_axis)
    return sfm_xyz_sub, gps_xyz_sub

def extend_rot_2to3(R_2d, rot_axis):
    '''
    Args: list[2,2]: 2D rotation matrix, str: rotation axis
    Return: np.array(3,3) 3D rotation matrix
    '''
    if rot_axis == 'x':
        R_3d = np.array([[1,         0,          0],
                         [0,  R_2d[0,0],  R_2d[0,1]],
                         [0,  R_2d[1,0],  R_2d[1,1]]])
    elif rot_axis == 'y':
        R_3d = np.array([[R_2d[0,0],  0,  R_2d[0,1]],
                         [0,          1,          0],
                         [R_2d[1,0],  0, R_2d[1,1]]])
    elif rot_axis == 'z':
        R_3d = np.array([[R_2d[0,0],  R_2d[0,1],  0],
                         [R_2d[1,0],  R_2d[1,1],  1],
                         [0,                  0,  1]])
    return R_3d

def get_subR(xyz_from, xyz_to, axis):
    '''
    Get 2D Rotation by solving a 2D problem, then extend it to
    3D Rotation matrix
    Args: np.array(N,3), np.array(N,3), str
    Return: np.array(3,3)

    '''
    
    reduced_from = reduce_xyz(xyz_from, axis)
    reduced_to = reduce_xyz(xyz_to, axis)
    c, Rmat, t = get_matched_transformation(reduced_from, reduced_to)
    subR = extend_rot_2to3(Rmat, axis)
    return subR

def convert_path_to_xyz_data(sfm_path, gps_path,  
                               anchor_pts_number, 
                               logger = DEFAULT_LOGGER,
                               smoothen_match = False, 
                               smooth_type = 'anchor',
                               match_footprint = True, combined = False, debug=False):
   
    '''
    main function for exporting routes with GPS anchor
    Args: sfm_path: SfM_path.sfm_path
          gps_geojson: dict that contains info from geojson file
          anchor_pts_number: int 
                             Number of anchor points from one end. 
                             The total number of anchor points will be 2*anchor_pts_number at most
                             (the repeated ones will be removed in unify_match_list())
          logger: logger object
          smoothen_match: bool 
                          if anchor horizonal plane should be smoothened
          smooth_type: str
                       'anchor' or 'full'
                       smoothen horizotal plane within each anchor or 
                       smoothen horizotal plane with all anchors
          match_footprint: bool
                           if footprint included in transfromation finding problem
          combined: bool
                    if the output geojson should be combined with original GPS path
    Retrun:
        geojson files will be written at output_path_list

    '''
    sfm_path_cleaned = filter_sfm_path(sfm_path) 
    #get matched sfm-gps pair 
    matched_sfm_gps_anchor_pair, \
            match_time_difference = find_sfm_gps_anchor_pair(sfm_path_cleaned,
                                                             gps_path, 
                                                             anchor_pts_number)
    
    reduced_matched_sfm_gps_anchor_pair = reduce_to_one2one_pair(matched_sfm_gps_anchor_pair, 
                                                                 match_time_difference) 

    
    if smoothen_match:
        smoothed_matched_sfm_gps_anchor_pair = smoothen_horizontal_area(reduced_matched_sfm_gps_anchor_pair, 
                                                                        logger=logger, 
                                                                        smooth=smooth_type)
        final_matched_sfm_gps_anchor_pair = smoothed_matched_sfm_gps_anchor_pair    
    else:
        final_matched_sfm_gps_anchor_pair = reduced_matched_sfm_gps_anchor_pair    
        
    final_sfm_gps_match = export_sfm_gps_match(final_matched_sfm_gps_anchor_pair, 
                                               smoothen_match, 
                                               match_footprint)
 
    # fetch data for getting transformation 
    xyz_sfm, xyz_gps, utm_data = convert_match_list_into_sfm_coord(final_sfm_gps_match)
    return xyz_sfm, xyz_gps, utm_data

def get_trf_from_sfm_gps_path_list(sfm_path_list, gps_path_list,\
                                    anchor_pts_number,\
                                    logger = DEFAULT_LOGGER):

    """
    find transformatioin between SfM and GPS from list of sfm_path_list and gps_path_list
    c: scale
    r: 3x3 rotation matrix 
    t: translation

    Args:
    sfm_path_list: list of SfM_path.sfm_path
    gps_path_list: list of GPS_path.gps_path
    anchor_pts_number: int 
    Number of anchor points from one end. 
    The total number of anchor points will be 2*anchor_pts_number at most
    (the repeated ones will be removed in unify_match_list())
    logger: logger object
    
    Return: 
    c: float. scale
    Rsub: list[3,3]. 3x3 rotation matrix 
    t: list[3]. translation
    utm_data: List of tuples (utm_east, utm_north, utm_zone, utm_letter)
                  ex: [(276778.07023962133, 2602132.268205515, 51, 'Q')]
 
    """
    xyz_sfm_all = np.empty(shape = (0,0))
    xyz_gps_all = np.empty(shape = (0,0))
    utm_data_list = []
    for sfm_path, gps_path in zip(sfm_path_list, gps_path_list):
        xyz_sfm, xyz_gps, utm_data  = convert_path_to_xyz_data(sfm_path, gps_path,  
                                      anchor_pts_number, 
                                      logger = DEFAULT_LOGGER,
                                      smoothen_match = False, 
                                      smooth_type = 'anchor',
                                      match_footprint = False, combined = False, debug=False)
        if xyz_sfm_all.shape == (0,0):
            xyz_sfm_all = xyz_sfm
        else:
            xyz_sfm_all = np.concatenate((xyz_sfm_all, xyz_sfm))
        if xyz_gps_all.shape == (0,0):
            xyz_gps_all = xyz_gps
        else:
            xyz_gps_all = np.concatenate((xyz_gps_all, xyz_gps))
        utm_data_list.append(utm_data)
    # Get transfomation
    c, Rmat, t = get_matched_transformation(xyz_sfm_all, xyz_gps_all)
    Rsub = get_subR(xyz_sfm_all, xyz_gps_all, 'y')
    logger.info("Rsub")
    logger.info(Rsub)
    return c, Rsub, t, utm_data_list

def apply_trf_and_export_to_gps(sfm_path, utm_data, c, Rsub, t, start_microsec, output_path_list):
    """
    Args:
    sfm_path: SfM_path.sfm_path
    utm_data: List of tuples (utm_east, utm_north, utm_zone, utm_letter)
                  ex: [(276778.07023962133, 2602132.268205515, 51, 'Q')]
    c: float. scale
    Rsub: list[3,3]. 3x3 rotation matrix 
    t: list[3]. translation
    start_microsec: int UTC time
    output_path_list: list of str: list of output path
    """
    sfm_path_cleaned = filter_sfm_path(sfm_path) 
    # Apply transfromation on sfm_path
    transformed_path_sfm = apply_trf_to_sfm_path(sfm_path_cleaned, c, Rsub, t)
    transformed_path_utm = transform_sfm_path_sfm2utm(transformed_path_sfm)
    rotation_sfm_list = get_rotations_from_sfm_path(transformed_path_sfm)
    babylon_angles_list = transform_rotation_to_babylon_angles(rotation_sfm_list)
   
    # Export to gps 
    sfm_geodata = make_geo_entries(transformed_path_utm, babylon_angles_list, start_microsec, utm_data[0]) 

    gps_geojson = {'type': 'Feature',
               'geometry': {'type': 'LineString', 'coordinates': []},
               'properties': {'video': []}}
    
    geojson_export = update_geodata(gps_geojson, sfm_geodata) 
    
    for output_path in output_path_list:
        with open(output_path, 'w') as f:
            safe_json_dump(geojson_export, f)


def export_routes_w_GPS_anchor(sfm_path, gps_geojson,  
                               output_path_list, 
                               anchor_pts_number, 
                               src_mrk_start_sec,
                               use_cutted_video,
                               logger = DEFAULT_LOGGER,
                               smoothen_match = False, 
                               smooth_type = 'anchor',
                               match_footprint = True, \
                               combined = False, \
                               read_footprint = False,
                               debug=False):
   
    '''
    main function for exporting routes with GPS anchor
    Args: sfm_path: SfM_path.sfm_path
          gps_geojson: dict that contains info from geojson file
          ourput_path_list: list of str
                            list of output file paths
          anchor_pts_number: int 
                             Number of anchor points from one end. 
                             The total number of anchor points will be 2*anchor_pts_number at most
                             (the repeated ones will be removed in unify_match_list())
          src_mrk_start_sec: int start time in source marker, in seconds
          use_cutted_video: bool, if use cutted video according to start/end time from source marker
          logger: logger object
          smoothen_match: bool 
                          if anchor horizonal plane should be smoothened
          smooth_type: str
                       'anchor' or 'full'
                       smoothen horizotal plane within each anchor or 
                       smoothen horizotal plane with all anchors
          match_footprint: bool
                           if footprint included in transfromation finding problem
          combined: bool
                    if the output geojson should be combined with original GPS path
    Retrun:
        geojson files will be written at output_path_list

    '''
    sfm_path_cleaned = filter_sfm_path(sfm_path) 
    gps_path_obj = get_gps_path(gps_geojson, read_footprint=read_footprint)
    gps_path = gps_path_obj.gps_path
    if len(gps_path) < 2:
        raise ValueError('gps path does not have enough points, need at least two')
    elif len(gps_path) <= 3:
        gps_path_obj.expand_gps()
        gps_path = gps_path_obj.gps_path
    #get matched sfm-gps pair 
    matched_sfm_gps_anchor_pair, \
            match_time_difference = find_sfm_gps_anchor_pair(sfm_path_cleaned,
                                                             gps_path, 
                                                             anchor_pts_number)
    
    reduced_matched_sfm_gps_anchor_pair = reduce_to_one2one_pair(matched_sfm_gps_anchor_pair, 
                                                                 match_time_difference) 

    
    if smoothen_match:
        smoothed_matched_sfm_gps_anchor_pair = smoothen_horizontal_area(reduced_matched_sfm_gps_anchor_pair, 
                                                                        logger=logger, 
                                                                        smooth=smooth_type)
        final_matched_sfm_gps_anchor_pair = smoothed_matched_sfm_gps_anchor_pair    
    else:
        final_matched_sfm_gps_anchor_pair = reduced_matched_sfm_gps_anchor_pair    
        
    final_sfm_gps_match = export_sfm_gps_match(final_matched_sfm_gps_anchor_pair, 
                                               smoothen_match, 
                                               match_footprint)
 
    # fetch data for getting transformation 
    xyz_sfm, xyz_gps, utm_data = convert_match_list_into_sfm_coord(final_sfm_gps_match)
    
    # Get transfomation
    c, Rmat, t = get_matched_transformation(xyz_sfm, xyz_gps)
    Rsub = get_subR(xyz_sfm, xyz_gps, 'y')
    logger.info("Rsub")
    logger.info(Rsub)
    
    # Apply transfromation on sfm_path
    transformed_path_sfm = apply_trf_to_sfm_path(sfm_path_cleaned, c, Rsub, t)
    transformed_path_utm = transform_sfm_path_sfm2utm(transformed_path_sfm)
    rotation_sfm_list = get_rotations_from_sfm_path(transformed_path_sfm)
    babylon_angles_list = transform_rotation_to_babylon_angles(rotation_sfm_list)
   
    # Export to gps 
    if use_cutted_video:
        start_microsec = src_mrk_start_sec*1000000 + get_start_microsec_from_gps_path(gps_path) 
    else:
        start_microsec = get_start_microsec_from_gps_path(gps_path) 
    
    sfm_geodata = make_geo_entries(transformed_path_utm, babylon_angles_list, start_microsec, utm_data[0]) 
    gps_geodata = get_gps_geometry(gps_geojson)
    start, end = get_gps_anchor_index(reduced_matched_sfm_gps_anchor_pair)
    
    if combined:
        geodata_combined = combine_geometry(start, end, gps_geodata , sfm_geodata)
        geojson_export = update_geodata(gps_geojson, geodata_combined) 
    else:
        geojson_export = update_geodata(gps_geojson, sfm_geodata) 
    
    for output_path in output_path_list:
        with open(output_path, 'w') as f:
            safe_json_dump(geojson_export, f)


