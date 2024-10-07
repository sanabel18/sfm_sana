import os
import subprocess
import shutil
import open3d as o3d
import numpy as np

from os.path import join
from task import Task
from utils.genutil import run_log_cmd


class DensifyPCL(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
        
        self.n_thread = prj_cfg['project']['n_thread']
        self.resol_lv = prj_cfg['mvs']['densify_pcl']['resol_lv']
        self.fuse_mode = prj_cfg['mvs']['densify_pcl']['fuse_mode']
        self.resol_limit = prj_cfg['mvs']['resol_limit']
        self.number_views = prj_cfg['mvs']['densify_pcl']['number_views']
        
        self.mvs_dir = prj_dirs['mvs']
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
        
    def run_task_content(self) -> int:
        ''' Just run the point cloud densification command. '''

        self.logger.info('MVS point cloud densification started.')
        cmd = ['DensifyPointCloud',
               '-w', self.mvs_dir,
               'scene.mvs',
               '--process-priority', str(2),
               '--max-threads', str(self.n_thread),               
               '--resolution-level', str(self.resol_lv),
               '--min-resolution', str(self.resol_limit[0]),
               '--max-resolution', str(self.resol_limit[1]),
               '--fusion-mode', str(self.fuse_mode),
               '--estimate-colors', '2',
               '--estimate-normals', '1',
               '--sample-mesh', '0',
               '--filter-point-cloud', '0',
               '--number-views', str(self.number_views)]
        for i in range(2):
            proc = run_log_cmd(self.logger, cmd)
            self.logger.info(f'Densify pointcloud iteration #{i} ended successfully.')

        self.logger.info(f'MVS point cloud densification ended with return code {proc.returncode}.')
        return proc.returncode
    
    
class GenPoissonMesh(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
        
        self.depth = prj_cfg['mvs']['poisson_mesh']['depth']
        self.scale = prj_cfg['mvs']['poisson_mesh']['scale']
        self.quantile = prj_cfg['mvs']['poisson_mesh']['quantile']
        self.tol = prj_cfg['mvs']['poisson_mesh']['tol']
        
        self.data_dir = prj_dirs['data']
        self.mvs_dir = prj_dirs['mvs']
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
        
    def run_task_content(self) -> int:
        ''' Run Open3D Poisson mesh creation, then remove vertices according to density. '''

        # Definition of filepaths
        poisson_mesh_path = join(self.mvs_dir, 'poisson_mesh.ply')
        final_mesh_path = join(self.data_dir, 'transformed_mesh.ply')
        
        self.logger.info('Starting Open3D Poisson mesh reconstruction...')
        try:
            pcd = o3d.io.read_point_cloud(join(self.mvs_dir, 'scene_dense.ply'))
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                   pcd, depth=self.depth, scale=self.scale)
            vertices_to_remove = densities < np.quantile(densities, self.quantile) * self.tol
            mesh.remove_vertices_by_mask(vertices_to_remove)
            o3d.io.write_triangle_mesh(poisson_mesh_path, mesh)
            
            shutil.copyfile(poisson_mesh_path, final_mesh_path)
            self.logger.info('The poisson mesh is copied to data directory as transformed_mesh.ply.')
            
            returncode = 0
            self.logger.info(f'Open3D Poission mesh reconstruction ended with return code {returncode}.')
            return returncode
        
        except Exception as e:
            self.logger.error(f'Poisson mesh reconstruction failed with error: {e}')
            returncode = 1
            self.logger.info(f'Open3D Poission mesh reconstruction ended with return code {returncode}.')
            return returncode
    
    
class ReconMesh(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
        
        self.n_thread = prj_cfg['project']['n_thread']
        self.min_point_distance = prj_cfg['mvs']['reconstruct_mesh']['min_point_distance']
        self.free_space_support = prj_cfg['mvs']['reconstruct_mesh']['free_space_support']
        self.decimate = prj_cfg['mvs']['reconstruct_mesh']['decimate']
        
        self.mvs_dir = prj_dirs['mvs']
        self.data_dir = prj_dirs['data']
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
        
    def run_task_content(self) -> int:
        ''' Just run the image listing command. '''
        # Definition of filepaths
        mesh_path = join(self.mvs_dir, 'scene_dense_mesh.ply')
        final_mesh_path = join(self.data_dir, 'transformed_mesh.ply')
        backup_mesh_path = join(self.data_dir, 'original_mesh.ply')

        self.logger.info('MVS mesh reconstruction started.')
        cmd = ['ReconstructMesh',
               '-w', self.mvs_dir,
               'scene_dense.mvs',
               '--process-priority', str(2),
               '--max-threads', str(self.n_thread),
               '--verbosity', '3',             # log verbosity (default: 2)
               '--min-point-distance', str(self.min_point_distance),  # 兩個點在不同 projection 至少要相差這麼多才會被認為是不同點 (default: 2.5)
               '--quality-factor', '1',        # quality weight multiplier (default: 1)
               '--free-space-support', str(self.free_space_support),    # 幫助建 weakly represented surfaces (default 0: disabled)
               '--decimate', str(self.decimate),              # 要不要 down sampling (default 1: disabled)
               '--remove-spurious', '10',      # remove spurious，數字越小刪掉越多 (default: 20; 0: disabled)
               '--remove-spikes', '1',         # remove spikes，這是 on/off 開關 (default 1: enable)
               '--close-holes', '30',          # close small holes (default: 30; 0: disabled)
               '--smooth', '2',                # number of iteration to smooth the surface (default: 2; 0: disabled)
            ]
        proc = run_log_cmd(self.logger, cmd)

        # Copy the output ply to data directory
        shutil.copyfile(mesh_path, final_mesh_path)
        shutil.copyfile(mesh_path, backup_mesh_path)
        self.logger.info('The mesh is copied to data directory as original_mesh.ply & transformed_mesh.ply.')

        self.logger.info(f'MVS mesh reconstruction ended with return code {proc.returncode}.')
        return proc.returncode

    
class RefineMesh(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
        
        self.n_thread = prj_cfg['project']['n_thread']
        self.scales = prj_cfg['mvs']['refine_texture']['scales']
        
        self.mvs_dir = prj_dirs['mvs']
        self.data_dir = prj_dirs['data']
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
        
    def run_task_content(self) -> int:
        ''' Just run the mesh refinement command. '''
        # Definition of filepaths
        refined_mesh_path = join(self.mvs_dir, 'scene_dense_mesh_refine.ply')
        final_mesh_path = join(self.data_dir, 'transformed_mesh.ply')
        backup_mesh_path = join(self.data_dir, 'original_mesh.ply')

        self.logger.info('MVS mesh refinement started.')
        cmd = ['RefineMesh',
               '-w', self.mvs_dir,
               'scene_dense_mesh.mvs',
               '--process-priority', str(2),
               '--max-threads', str(self.n_thread),               
               '--scales', str(self.scales)]
        proc = run_log_cmd(self.logger, cmd)

        # Copy the output ply to data directory
        shutil.copyfile(refined_mesh_path, final_mesh_path)
        shutil.copyfile(refined_mesh_path, backup_mesh_path)
        self.logger.info('The refined mesh is copied to data directory as original_mesh.ply & transformed_mesh.ply.')

        self.logger.info(f'MVS mesh refinement ended with return code {proc.returncode}.')
        return proc.returncode


class TextureMesh(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
        
        self.n_thread = prj_cfg['project']['n_thread']
        self.resol_lv = prj_cfg['mvs']['refine_texture']['resol_lv']
        self.resol_limit = prj_cfg['mvs']['resol_limit']
        
        self.mvs_dir = prj_dirs['mvs']
        self.data_dir = prj_dirs['data']
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
        
    def run_task_content(self) -> int:
        ''' Just run the mesh texturing command. '''
        
        # Definition of filepaths
        textured_mesh_path = join(self.mvs_dir, 'scene_dense_mesh_refine_texture.ply')
        final_mesh_path = join(self.data_dir, 'transformed_mesh.ply')
        backup_mesh_path = join(self.data_dir, 'original_mesh.ply')

        self.logger.info('MVS mesh texturing started.')
        cmd = ['TextureMesh',
               '-w', self.mvs_dir,
               'scene_dense_mesh_refine.mvs',
               '--process-priority', str(2),
               '--max-threads', str(self.n_thread),               
               '--resolution-level', str(self.resol_lv),
               '--min-resolution', str(self.resol_limit[0]),
               '--decimate', str(0.5)]
        proc = run_log_cmd(self.logger, cmd)
        
        # Copy the output ply to data directory
        shutil.copyfile(textured_mesh_path, final_mesh_path)
        shutil.copyfile(textured_mesh_path, backup_mesh_path)
        self.logger.info('The textured mesh is copied to data directory as original_mesh.ply & transformed_mesh.ply.')

        self.logger.info(f'MVS mesh texturing ended with return code {proc.returncode}.')
        return proc.returncode
    

