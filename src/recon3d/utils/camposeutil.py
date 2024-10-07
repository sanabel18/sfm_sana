import os
import json

def gen_dummy_campose(out_json, end_timestamp):
    '''
    create dumppy camera pose if movement is stationary
    entries of dummy traj list: [timestamp(in millisec), x, y, z, qx, qy, qz, qw]
    Args:
        out_json: str, output file path 
        end_timestamp: float,  end time stamp in millisec 
    '''
    traj_list = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                 [end_timestamp, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    with open(out_json, 'w') as f:
        json.dump(traj_list, f, indent=4)

