#!/opt/conda/bin/python

import os
import json
import shutil
import sys

from os.path import join


def remove_if_exists(file: str):
    try:
        os.remove(file)
    except OSError:
        print(f'File {file} may not exist.')
    return

def rmtree_if_exists(folder: str):
    try:
        shutil.rmtree(folder)
    except OSError:
        print(f'Folder {folder} may not exist.')
    

def revert_to_original(slice_folder: str):
    # Delete things
    for folder in os.listdir(slice_folder):
        if folder[:5]=='data_':
            data_folder = join(slice_folder, folder)
            remove_if_exists(join(data_folder, 'routes_succinct.json'))
            remove_if_exists(join(data_folder, 'routes.json'))
            remove_if_exists(join(data_folder, 'route_poses.ply'))
            remove_if_exists(join(data_folder, 'route_footprints.ply'))
            
            remove_if_exists(join(data_folder, 'sfm_data_transformed.json'))
            remove_if_exists(join(data_folder, 'sfm_data_poses.ply'))
            remove_if_exists(join(data_folder, 'sfm_data_footprints.ply'))

            remove_if_exists(join(data_folder, 'transformed_mesh.ply'))
            remove_if_exists(join(data_folder, 'transformed_mesh.obj'))
            
    # Copy things
    for folder in os.listdir(slice_folder):
        if folder=='subsidiaries':
            subsidiaries = join(slice_folder, folder)
            shutil.copy2(join(subsidiaries, 'loc', 'sfm_data_expanded_reordered.json'),
                         join(data_folder, 'sfm_data_expandFPS.json'))
            shutil.copy2(join(subsidiaries, 'loc', 'sfm_data_expanded_reordered.json'),
                         join(data_folder, 'sfm_data_transformed.json'))
            shutil.copy2(join(subsidiaries, 'mvs', 'scene_dense_mesh.ply'),
                         join(data_folder, 'original_mesh.ply'))
            shutil.copy2(join(subsidiaries, 'mvs', 'scene_dense_mesh.ply'),
                         join(data_folder, 'transformed_mesh.ply'))
            
    # Clean up transformation.json
    with open(join(data_folder, 'transformation.json'), 'r') as f:
        trf = json.load(f)
    I = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
    trf_new = {'concatenated': I,
               'list':[{'matrix': I, 'timestamp': trf['list'][0]['timestamp'], 'note': trf['list'][0]['note']}]}
    with open(join(data_folder, 'transformation.json'), 'w') as f:
        json.dump(trf_new, f)

    # Clean up done_tasks.json
    with open(join(slice_folder, 'done_tasks.json'), 'r') as f:
        done_tasks = json.load(f)
    done_tasks_new = {}
    for key in done_tasks:
        done_tasks_new[key] = done_tasks[key]
        if key == 'MVS_RECONSTRUCT_MESH': break
    with open(join(slice_folder, 'done_tasks.json'), 'w') as f:
        json.dump(done_tasks_new, f)
    
    return
        

if __name__ == "__main__":
    superprj_path = sys.argv[1]
    loc_path = join(superprj_path, 'loc')
    recon_path = join(superprj_path, 'recon')
    
    print(f'Start reverting super project {superprj_path}...')
    rmtree_if_exists(loc_path)
    print('Loc folder deleted.')
    for slice_nr in os.listdir(recon_path):
        slice_folder = join(recon_path, slice_nr)
        revert_to_original(slice_folder)
        print(f'Slice folder nr.{slice_nr} reverted.')
        