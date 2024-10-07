import json
import numpy as np
import glob
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from pathlib import Path
import copy

class RouteFiller:
    """Fill missing frames in a routes.json
    The example routes.json file can be found in src/recon3d/tool/assets/routes_pretty.json
    Args:
        route_file (str): path of routes.json file
        logger: Logger object

    self.data: data dict of input routes.json
    self.t_arr (array[N]) time stamp routes.json
    self.pos_arr (array[N,3]) camera position in routes.json
    self.rot_arr (array[N,3,3]) camera orientation in routes.json
    self.footprint_arr (array[N,3]) camera footprint in routes.json
    self.with_footprint (bool): flag of if footprint exists in rotues.json
    self.frame_idx_arr (array[N]): frame index in routes.json
    self.image_dir (str): image path of in routes.json
    self.slice_num (int): ID number of slice in routes.json
    self.time_step (float): timestep of this route
    self.image_path_lsit (List[str]): image file path in image_dir
    self.total_frame_num (int): number of images in image_dir
    self.full_ts (array[N]): complete time stamps generated with time step and total frame number
    self.full_index (array[N]): complete frame index generated with total frame number
    """
    def __init__(self, route_file, logger):
        f = open(route_file,"rb")
        self.data = json.load(f)
        f.close()
        self.logger = logger 
        self.t_arr, self.pos_arr, self.rot_arr, self.with_footprint, self.footprint_arr, self.frame_idx_arr, self.image_dir, self.slice_num = self.get_frame_data()
        self.time_step = self.get_time_step(self.frame_idx_arr, self.t_arr)
        self.image_path_list = self.get_image_list(self.image_dir)
        self.total_frame_num = len(self.image_path_list)
        self.full_ts = self.gen_full_ts(self.time_step, self.total_frame_num)      
        self.full_index = self.gen_full_index(self.total_frame_num)

    def fill_route(self):
        """Fill route with full time stamp.
        For missing frames at head and tail, just append repeated camera poses. 
        For missing poses in the middle of routes, use \
        interpoated position and Slerp for orientation.
        """
        t, pos, rot = self.extend_head_tail(self.t_arr, self.pos_arr, self.rot_arr, self.frame_idx_arr, \
                self.total_frame_num, self.time_step)
        pos_interp = self.interpolate_pos(pos, t, self.full_ts)
        rots_interp = self.interpolate_rot(rot, t, self.full_ts)
        if self.with_footprint:
            t, footprint, rot = self.extend_head_tail(self.t_arr, self.footprint_arr, self.rot_arr, \
                self.frame_idx_arr, self.total_frame_num, self.time_step)
            footprint_interp = self.interpolate_pos(footprint, t, self.full_ts)
        else:
            footprint_interp = [None]*len(t)

        full_frames = self.make_full_frames(self.image_path_list, self.full_ts, self.full_index, \
                pos_interp, footprint_interp, rots_interp, self.slice_num)
        
        self.logger.info("original route length: {}".format(self.t_arr.shape))
        self.logger.info("intepoated route length: {}".format(pos_interp.shape))
 
        new_route = copy.deepcopy(self.data)
        new_route['frames'] = full_frames 
        return new_route

    def get_frame_data(self):
        """Parse frame information from route data dict
        Returns:
            timestamp_arr (array[N]) time stamp routes.json
            pos_arr (array[N,3]) camera position in routes.json
            rot_arr (array[N,3,3]) camera orientation in routes.json
            footprint_arr (array[N,3]) camera footprint in routes.json
            with_footprint (bool): flag of if footprint exists in rotues.json
            frame_idx_arr (array[N]): frame index in routes.json
            image_dir (str): image path of in routes.json
            slice_num (int): ID number of slice in routes.json 
        """
        frames = self.data["frames"]
        timestamp_list = []
        pos_list = []
        rot_list = []
        footprint_list = []
        frame_idx_list = []
        for f in frames:
            image_dir = f["dir"]
            timestamp = f["timestamp"]
            pos = f["position"]
            rot = f["rotation"]
            footprint = f["footprint"]
            frame_idx = f["frame_idx"]
            slice_num = f["slice"]
            timestamp_list.append(timestamp)
            pos_list.append(pos)
            footprint_list.append(footprint)
            rot_list.append(rot)
            frame_idx_list.append(frame_idx)

        timestamp_arr = np.array(timestamp_list)
        pos_arr = np.array(pos_list)
        rot_arr = np.array(rot_list)
        
        if all(fp is None for fp in footprint_list):
            with_footprint = False
            footprint_arr = None
        else:
            with_footprint = True
            footprint_arr = np.array(footprint_list)
        frame_idx_arr = np.array(frame_idx_list)
        return timestamp_arr, pos_arr, rot_arr, with_footprint, footprint_arr, frame_idx_arr, image_dir, slice_num 

    def make_full_frames(self,image_path_list, full_ts, full_index, pos_interp, fp_interp, rots_interp, slice_num):
        """Make frame dict with full time stamps
        Args:
            image_path_list (List[Path]): path obj of images
            full_ts (List[float]): full timestamp in millisecond
            full_idx (List[int]): frame index
            pos_interp (List[N,3]): interpolated camera pos with full timestamp
            fp_interp (List[N,3]): interpolated footprint pos with full timestamp
            rots_interp (List[N,3,3]): interpolated camerat orientation with full timestamp
            slice_num (int): slice ID of this frame dict
        Returns:
            frame dict in route data structure
        """
        frames = []
        for image_path, ts, idx, pos, fp, rot in zip(image_path_list, full_ts, full_index, pos_interp, fp_interp, rots_interp):
            frame_dict = {
                    "dir": str(image_path.parent),
                    "slice": slice_num,
                    "frame_idx": idx,
                    "timestamp": ts,
                    "filenames": [str(image_path.name)],
                    "lense_positions": [pos],
                    "lense_rotations": [rot],
                    "lense_footprints": [fp],
                    "position": pos,
                    "rotation": rot,
                    "footprint": fp,
                   }
            frames.append(frame_dict)
        return frames

    def interpolate_pos(self, pos, ts , ts_interp):
        """Intepolate camera position 
        array size N <= M
        Args:
            pos (array[N, 3]): camera positions
            ts (array[N]): timestamp 
            ts_interp (array[M]): timestampe to be interpolated to
        Returns: 
            pos_interp (array[M, 3]): interpolated timestamp
        """
        x = pos[:,0]
        y = pos[:,1]
        z = pos[:,2]
        # to avoid nemeric problem, setting boundary of ts_interp to ts
        ts_interp[-1] = ts[-1]
        ts_interp[0] = ts[0]
        x_interp = np.interp(ts_interp, ts, x)
        y_interp = np.interp(ts_interp, ts, y)
        z_interp = np.interp(ts_interp, ts, z)
        pos_interp = np.stack((x_interp, y_interp, z_interp), axis=1)
        return pos_interp

    def interpolate_rot(self, rots, ts, ts_interp):
        """Intepolate camera orientation 
        array size N <= M
        Args:
            rots (array[N, 3, 3]): camera orientation 
            ts (array[N]): timestamp
            ts_interp (array[M]): timestampe to be interpolated to
        Returns:
            pos_interp (array[M, 3, 3]): interpolated camera orientation
        """ 
        r = R.from_matrix(rots)
        slerp = Slerp(ts, r)
        # to avoid nemeric problem, setting boundary of ts_interp to ts
        ts_interp[-1] = ts[-1]
        ts_interp[0] = ts[0]
        r_interp = slerp(ts_interp)
        rots_interp = r_interp.as_matrix()
        return rots_interp

    def get_image_list(self, image_dir):
        """Fetch image list within image dir
        Args:
            image_dir (str): path of image directory
        Return:
            image_path_list (List(Path)): Path object of image file path  
        """
        image_list = sorted(glob.glob(image_dir+'/*'))
        image_path_list = []
        for image in image_list:
            image_path = Path(image)
            image_path_list.append(image_path)
        return image_path_list
     
    def extend_head_tail(self, t_in, pos_in, rot_in, frame_idx, total_frame_num, time_step):
        """Append repeated camera pose at missing head or tail
        Args:
            t_in (array[N]): timestamp in millisecond
            pos_in (array[N, 3]): camera position
            rot_in (array[N, 3, 3]): camera orientation
            frame_idx (array[N]): frame index
            total_frame_num (int): number of all images
            time_step (float): time step in millisec
        Returns:
            t (array[M]): timestamp in millisecond with head and tail appended
            pos (array[M, 3]): camera position with head and tail appended
            rot (array[M, 3, 3]): camera orientation with head and tail appended
        """
        pos = copy.deepcopy(pos_in)
        rot = copy.deepcopy(rot_in)
        t = copy.deepcopy(t_in)
        idx_start = frame_idx[0]
        idx_end = frame_idx[-1]
        if idx_start > 0:
            self.logger.info("need to append head")
            start_list = range(idx_start-1,-1,-1)
            self.logger.info("start list {}".format(start_list))
            for start in start_list:
                if start == 0:
                    t = np.insert(t, 0, 0.0 )
                    pos = np.insert(pos,0,pos[0,:],axis=0)
                    rot = np.insert(rot,0,rot[0,:,:],axis=0)
                else:
                    t = np.insert(t, 0, start*time_step)
                    pos = np.insert(pos,0,pos[0,:],axis=0)
                    rot = np.insert(rot,0,rot[0,:,:],axis=0)
        if idx_end < total_frame_num -1:
            self.logger.info("need to append end")
            self.logger.info("end list {}".format(idx_end))
            end_list = [i for i in range(idx_end + 1, total_frame_num)]
            for end in end_list:
                t = np.append(t, end*time_step)
                pos = np.append(pos, pos[-1,:].reshape((1,3)), axis=0)
                rot = np.append(rot, rot[-1,:,:].reshape((1,3,3)), axis=0)
        return t, pos, rot

       

    def get_time_step(self, frame_idx, t):
        """From list of timestamp get time step
        Args: 
            frame_idx (array[N]): frame index
            t (array[N]): timestamp in millisec
        Return:
            time_step (float): time step in millisec
        """
        idx_diff = frame_idx[1:] - frame_idx[0:-1]
        ts_diff = t[1:] - t[0:-1]
        time_step = np.median(ts_diff[np.where(idx_diff == 1)])
        return time_step
     
    def gen_full_ts(self, time_step, frames_num):
        """Generate full timestamp with time step
        Args:
            time_step (float): time step in millisec
            frames_num (int): total frame number
        Retrun:
            full_ts (array[N]): timestamp in millisec
        """
        full_ts = np.arange(0, time_step*frames_num, time_step)
        return full_ts

    def gen_full_index(self, frames_num):
        """Generate full frame index
        Args:
            frame_num (int): total frame number
        Return:
            full_index (array[N]): full frame index
        """
        full_index = [i for i in range(frames_num)]
        return np.array(full_index)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def main():
    route_file = 'routes.json'
    route_filler = RouteFiller(route_file)
    new_route = route_filler.fill_route()
    print(new_route)
    json_object = json.dumps(new_route, indent=4, cls=NpEncoder)
 
     # Writing to sample.json
    with open("filled_route.json", "w") as outfile:
         outfile.write(json_object)

if __name__ == "__main__":
    main()
