import numpy as np
import pandas as pd

from fnmatch import fnmatch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from utils.meshdata import CurvMeshData
from utils.meshlabutil import ProcMlx

    
def seg_kmeans(data: CurvMeshData, n_clusters: int, normal_test_thr: float,
               bounding_box_buffer: float, min_seg_size: int, output_dir: str, logger):
    ''' Run kmeans to get seeds, then run BFS for all seeds. '''
    
    logger.info('Start selecting seeds by KMeans...')
    seg_seed_fids = select_seeds_kmeans(data, n_clusters)
    logger.info(f'End selecting seeds by KMeans with seed fids: {seg_seed_fids}.')

    for seg_seed_fid in seg_seed_fids:
        # If the seed already got a label in previous iterations, then don't run BFS with it 
        if data.seg_labels[seg_seed_fid] == -1:
            logger.info(f'Seed fid {seg_seed_fid} is not labeled yet, label it with {data.n_seg_label} and start BFS from it...')
            seg_BFS(data, seg_seed_fid, data.n_seg_label, normal_test_thr, bounding_box_buffer, min_seg_size, output_dir, logger)
            logger.info(f'Grouping / labeling from fid {seg_seed_fid} with label {data.n_seg_label} done.')
            data.n_seg_label += 1
        else: 
            logger.warning(f'Seed fid {seg_seed_fid} is already labeled, ignore it...')

            
def select_seeds_kmeans(data: CurvMeshData, n_clusters: int) -> list:
    '''
    Select seeds for BFS search using kmeans:
        1. Do kmeans on all flat points and get label of all flat points
        2. Find the points for each cluster which is closest to centroid
    '''
    
    # Initialize kmeans object, fit and then predict the label of all points that are flat
    kmeans = KMeans(n_clusters=n_clusters)
    flat_pts_info = data.normal_info[data.is_pt_flat]
    kmeans.fit(flat_pts_info)
    flat_pts_info['label'] = kmeans.predict(flat_pts_info)
    
    # Prepare to find nearest neighbor for each cluster
    centroids = kmeans.cluster_centers_
    nnbrs = NearestNeighbors(n_neighbors=1, algorithm='auto')
    seed_pt_ids = []
    # Find for each cluster
    for i in range(n_clusters):
        center = centroids[i].tolist()
        center.append(i)  # Append label to unify format
        labeled_flat_pts_info = flat_pts_info[flat_pts_info['label'] == i]
        nnbrs.fit(labeled_flat_pts_info.values)  # TODO (hkazami): Check the effect of phi & theta
        id1st = nnbrs.kneighbors([center], 1 , return_distance=False)
        id_flat = labeled_flat_pts_info.iloc[id1st[0][0]].name
        seed_pt_ids.append(id_flat)
    seed_face_ids = data.vid2fid_flat(seed_pt_ids)  # Vertex id to face id
    return seed_face_ids
            

def seg_BFS(data: CurvMeshData, seg_seed_id: int, seg_label: int, normal_test_thr: float,
            bounding_box_buffer: float, min_seg_size: int, output_dir: str, logger):
    '''
    Do breadth first search from given segment seed with normal criteria, inorder to find
    all faces which should be in the same segment as the seed.
    '''
    
    def normal_test(normals: dict, face_normal_seed, thr: float):
        dot1 = np.dot(normals['v1'], face_normal_seed)
        dot2 = np.dot(normals['v2'], face_normal_seed)
        dot3 = np.dot(normals['v3'], face_normal_seed)
        return (dot1 > thr and dot2 > thr and dot3 > thr)
    
    lv_bfs = 0  # For debugging only
    search_query = []
    search_query.append(seg_seed_id)
    data.is_face_visited[seg_seed_id] = True

    # Get face normal of the seed face
    face_normal_seed = data.get_normals(seg_seed_id)['f']
    while(search_query):
        lv_bfs += 1
        logger.info(f'BFS running at level / depth {lv_bfs}...')
        
        search_fid = search_query.pop(0)
        # Set label
        data.seg_labels.loc[search_fid] = seg_label
        # Find neighbors
        nbr_list = data.get_nbr_faces(search_fid)
        # Check if neighbor meets criteria, if yes then append in query
        for neighbor in nbr_list:
            normal_align = normal_test(data.get_normals(neighbor), face_normal_seed, normal_test_thr)
            if (normal_align and data.is_face_visited[neighbor] == False):
                search_query.append(neighbor)
                data.is_face_visited[neighbor] = True
    
    # Write mlx file
    ProcMlx.write_seg_mlx(data, seg_label, face_normal_seed, bounding_box_buffer, min_seg_size, output_dir, logger)
    logger.info(f'Generation of mlx files for segment no. {seg_label} done.')


