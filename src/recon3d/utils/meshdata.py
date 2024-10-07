import numpy as np
import pandas as pd
from utils.plyutil import read_ply


class MeshData(object):
    def __init__(self, mesh_path: str):
        '''
        Read mesh without curvature (color) and convert to:
            - pandas dataframe
            - various arrays
        All mesh data processing or extraction functions are written here.
        '''
    
        # Read ply
        ply_dataframe = read_ply(mesh_path)
        # They are all pandas dataframes
        self.points = ply_dataframe['points'][['x','y','z']]
        self.faces = ply_dataframe['mesh'][['v1','v2','v3']]

    def get_nbr_faces(self, search_fid: int):
        ''' Find all neighboring faces of a certain face id. '''

        search_face = self.faces.iloc[search_fid, :]
        # Find all faces that has vertex 1, 2 or 3
        fid_with_v1 = self.faces.index[(self.faces['v1'] == search_face['v1']) |
                                  (self.faces['v2'] == search_face['v1']) |
                                  (self.faces['v3'] == search_face['v1'])].tolist()
        fid_with_v2 = self.faces.index[(self.faces['v1'] == search_face['v2']) |
                                  (self.faces['v2'] == search_face['v2']) |
                                  (self.faces['v3'] == search_face['v2'])].tolist()
        fid_with_v3 = self.faces.index[(self.faces['v1'] == search_face['v3']) |
                                  (self.faces['v2'] == search_face['v3']) |
                                  (self.faces['v3'] == search_face['v3'])].tolist()

        # Neighboring faces shares 2 vertices
        fid_with_v1_v2 = list(set(fid_with_v1) & set(fid_with_v2))
        fid_with_v2_v3 = list(set(fid_with_v2) & set(fid_with_v3))
        fid_with_v1_v3 = list(set(fid_with_v1) & set(fid_with_v3))

        # Remove search face itself
        if search_fid in fid_with_v1_v2:
            fid_with_v1_v2.remove(search_fid)
        if search_fid in fid_with_v2_v3:
            fid_with_v2_v3.remove(search_fid)
        if search_fid in fid_with_v1_v3:
            fid_with_v1_v3.remove(search_fid)
        return (fid_with_v1_v2 + fid_with_v2_v3 + fid_with_v1_v3)

    def get_bounding_box(self, fid_list: list, buffer: float) -> list:
        ''' Get a naive bounding box of all vertices of all input faces. '''
        
        def get_min_max_dict(xyz_list: list) -> list:
            min_max_list = [min(xyz_list), max(xyz_list)]
            delta = min_max_list[1] - min_max_list[0]
            # Type casting to float to be able to write in json
            return {'min': float(min_max_list[0] - delta * buffer / 2), 'max': float(min_max_list[1] + delta * buffer / 2)}

        # There are a lot of repetitive vids but it doesn't affect the result
        vid_list = self.faces.iloc[fid_list].v1.tolist() + \
            self.faces.iloc[fid_list].v2.tolist() + \
            self.faces.iloc[fid_list].v3.tolist()
        x_list = self.points.iloc[vid_list].x.tolist() 
        y_list = self.points.iloc[vid_list].y.tolist() 
        z_list = self.points.iloc[vid_list].z.tolist()
        return {'x': get_min_max_dict(x_list), 'y': get_min_max_dict(y_list), 'z': get_min_max_dict(z_list)}
              

class CurvMeshData(MeshData):
    def __init__(self, mesh_path: str):
        '''
        Read mesh with curvature and convert to:
            - pandas dataframe
            - various arrays
        All mesh data processing or extraction functions are written here.
        '''
        
        super().__init__(mesh_path)
        
        # Color (curvature) related initializations
        ply_dataframe = read_ply(mesh_path)
        self.normals = ply_dataframe['points'][['nx','ny','nz']]
        self.colors = ply_dataframe['points'][['red','green','blue']]
        self.is_pt_flat = None
        self._calc_pt_flat_info()

        # Data container for kmeans
        # TODO (hkazami): Refactor (It's quite redundant)
        self.normal_info = pd.DataFrame({
            'theta': np.full(len(self.points.index), -1, dtype=float).tolist(),
            'phi': np.full(len(self.points.index), -1, dtype=float).tolist(),
            'x': self.points['x'], 'y': self.points['y'], 'z': self.points['z']})
        self._calc_theta_phi_info()

        # Face indices vs. segmentation labels
        self.seg_labels = pd.Series(  # Series is like 1D DataFrame
            np.full(len(self.faces.index), -1, dtype=int))
        self.is_face_visited = [False] * len(self.faces.index)
        self.n_seg_label = 0                 
    
    def safe_div(self, num, den):
        if np.abs(den) > 0:
            return num / den
        else:
            return num
            #raise ZeroDivisionError('Denominator is zero, division failed!')

    def _calc_pt_flat_info(self):
        ''' Assumption: Red largest means flat. '''
        
        red = []
        for index, row in self.colors.iterrows():
            if ((row['red'] > row['green']) and (row['red'] > row['blue'])):
                red.append(True)
            else:
                red.append(False)
        self.is_pt_flat = pd.Series(red)  # Series is like 1D DataFrame
    
    def _calc_theta_phi_info(self):
        ''' Calculate orientation (phi, theta) from normal. '''
        
        for idx, row in self.normals.iterrows():
            normal = np.array([row['nx'], row['ny'], row['nz']])
            normal = self.safe_div(normal, np.linalg.norm(normal))
            self.normal_info['phi'][idx] = np.arctan2(normal[1],normal[0])

    def get_normals(self, fid: int) -> dict:
        vid1 = self.faces.iloc[fid]['v1']
        vid2 = self.faces.iloc[fid]['v2']
        vid3 = self.faces.iloc[fid]['v3']
        
        # Face normal as mean of vertex normals
        normal1 = self.normals.iloc[vid1].values
        normal2 = self.normals.iloc[vid2].values
        normal3 = self.normals.iloc[vid3].values
        normal1 = self.safe_div(normal1, np.linalg.norm(normal1))
        normal2 = self.safe_div(normal2, np.linalg.norm(normal2))
        normal3 = self.safe_div(normal3, np.linalg.norm(normal3))
        face_normal = normal1 + normal2 + normal3
        face_normal = self.safe_div(face_normal, np.linalg.norm(face_normal))
        
        # Face normal as cross product (unused)
        '''
        p1 = self.points.iloc[vid1].values
        p2 = self.points.iloc[vid1].values
        p3 = self.points.iloc[vid1].values
        v1 = p2 - p1
        v2 = p3 - p1
        face_normal = np.cross(v1, v2)
        face_normal = self.safe_div(face_normal, np.linalg.norm(face_normal))
        '''
        return {'v1': normal1, 'v2': normal2, 'v3': normal3, 'f': face_normal}
            
    def get_mean_aligned_face_normal(self, fid_list: list, sigma_fac: float,
                                     align_axis: np.ndarray = np.array([0, 1, 0])) -> np.ndarray:
        '''
        Arguments:
            - Axis to be aligned with (default is positive y axis)
            - Face id list
            - A factor of standard deviation
        Returns:
            - The mean of the normals whose dot product w/ align axis is within sigma_fac of standard distribution
            - The dot prodoct of align axis and the mean of the normals
        '''
        normals = []
        
        # Gather all face normals
        for fid in fid_list:
            face_normal = self.get_normals(fid)['f']
            normals.append(face_normal)
        normals = np.array(normals)
        # Get the component of all normals on the given axis to align
        # If all input faces have the same direction of normals, the the sign of axis components here is not important.
        # TODO (hkazami): Check the statement above.
        axis_components = np.dot(normals, align_axis.reshape(-1,1))  # Reshape to column vector to execute dot product
        
        # Select only normals whose y component is within a certain factor of stddev from mean
        # (Using the concept of normal distribution) 
        stddev_axis = np.std(axis_components)
        abs_diff = np.abs(axis_components - np.mean(axis_components))
        is_inlier_1D = (abs_diff < sigma_fac * stddev_axis).ravel()  # Reshaped to 1D
        normals_inlier = normals[is_inlier_1D, :]
        
        # Sum over all valid normals, and normalize it again
        sum_normals = np.sum(normals_inlier, axis=0)  # Now it should be numpy 1D array with 3 elements
        mean_face_normal = self.safe_div(sum_normals, np.linalg.norm(sum_normals))        
        dot_product = np.dot(mean_face_normal, align_axis)
        return mean_face_normal, dot_product
    
    def vid2fid_flat(self, vid_list: list) -> list:
        ''' Given vertex indices, return corresponding flat face indices. '''

        def color_test(face_row):
            color1 = self.colors.iloc[face_row['v1']]
            color2 = self.colors.iloc[face_row['v2']]
            color3 = self.colors.iloc[face_row['v3']]
            # If red is max or not
            red1 = ((color1['red'] > color1['blue']) and (color1['red'] > color1['green']))
            red2 = ((color2['red'] > color2['blue']) and (color2['red'] > color1['green']))
            red3 = ((color3['red'] > color3['blue']) and (color3['red'] > color3['green']))
            return (red1 and red2 and red3)

        fid_list = []
        for vid in vid_list:
            fids = self.faces.index[
                (self.faces['v1'] == vid) |
                (self.faces['v2'] == vid) |
                (self.faces['v3'] == vid)].tolist()
            for fid in fids:
                all_flat = color_test(self.faces.iloc[fid,:])
                if all_flat:
                    fid_list.append(fid)
                    break
        return fid_list       
    
