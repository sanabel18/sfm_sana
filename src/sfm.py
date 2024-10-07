#!/usr/bin/env python3
#
# Created by stefan on 2020/5/12.
#
import os
import sys
import argparse
import subprocess
import shutil
import time
import multiprocessing
import datetime
from os.path import join, abspath
from colorama import Fore, Style


class Config(object):
    cameradb = '/opt/openmvg/src/openMVG/exif/sensor_width_database/sensor_width_camera_database.txt'
    openmvg_dir = '/opt/openmvg_build/install/bin'
    openmvs_dir = '/usr/local/bin/OpenMVS'

    # 1: Pinhole no distortion
    # 2: Pinhole + 1 radial distortion
    # 3: Pinhole + 3 radial distortion
    # 4: Pinhole + 3 radial distortion + 2 tangential distortion
    # 5: Pinhole + 4 fish-eye distortion
    # 7: Spherical (equirectangular)
    # camera_model = '3'

    # openMVS 的 priority:
    #   IDLE = -3,
    #   LOW = -2,
    #   BELOWNORMAL = -1,
    #   NORMAL = 0,
    #   ABOVENORMAL = 1,
    #   HIGH = 2,
    #   REALTIME = 3
    # 實際測試，設 -1 = PRI:30,Nice:10, -2 = PRI:35,Nice:15, -3 = PRI:39,Nice:19, 設其它值都是 PRI:20,Nice:0
    # 也就是說，只能降低 priority 不能升高。設 >=0 至少不降低。
    process_priority = '2'

    def __init__(self, args):
        Config.cameradb = args.cameradb
        Config.openmvs_dir = args.openmvg_dir
        Config.openmvs_dir = args.openmvs_dir
        Config.camera_model = args.camera_model

        self.input_dir = abspath(args.input_dir.rstrip('/'))
        self.output_dir = abspath(args.output_dir) if hasattr(args, 'output_dir') else f'{self.input_dir}_out'
        self.loc_dir = abspath(args.loc_dir) if hasattr(args, 'loc_dir') else ''
        self.nproc = args.nproc

    @property
    def matches_dir(self) -> str:
        return join(self.sfm_dir, 'matches')

    @property
    def sfm_dir(self) -> str:
        return join(self.output_dir, 'sfm')

    @property
    def mvs_dir(self) -> str:
        return join(self.output_dir, 'mvs')

    @property
    def keypoints_svg_dir(self) -> str:
        return join(self.matches_dir, 'keypoints_svg')

    @property
    def matches_svg_dir(self) -> str:
        return join(self.matches_dir, 'matches_svg')

    @property
    def tracks_svg_dir(self) -> str:
        return join(self.matches_dir, 'tracks_svg')

    @property
    def loc_query_dir(self) -> str:
        return join(self.loc_dir, 'query_img') if self.loc_dir else ''

    @property
    def loc_matches_dir(self) -> str:
        return join(self.loc_dir, 'matches') if self.loc_dir else ''


class TaskResult(object):
    def __init__(self, step: int, name: str, cmd: [str], return_code: int, t: float, color: int):
        self.step = step
        self.name = name
        self.cmd = cmd
        self.return_code = return_code
        self.t = t
        self.color = color


class Task(object):
    def __init__(self, name: str, cmd: [str], color: int = Fore.GREEN):
        self.name = name
        self.cmd = cmd
        self.color = color

    def run(self, pipeline: str, step: int, log_dir: str) -> TaskResult:
        log_out = join(log_dir, f'{pipeline}_step{step:02d}_{self.name}.log')
        log_err = join(log_dir, f'{pipeline}_step{step:02d}_{self.name}_err.log')
        try:
            t = time.time()
            proc = subprocess.run(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            t = time.time() - t

            with open(log_out, 'w') as f:
                f.write(proc.stdout.decode('utf-8'))

            err = proc.stderr.decode('utf-8')
            if len(err) > 0:
                with open(log_err, 'w') as f:
                    f.write(err)

            return TaskResult(step, self.name, self.cmd, proc.returncode, t, self.color)
        except KeyboardInterrupt:
            msg = f'{Fore.RED}Abort: Process canceled by user{Style.RESET_ALL}' + "\n"
            with open(log_err, 'w') as f:
                f.write(msg)
            print(msg)
            return TaskResult(step, self.name, self.cmd, -1, t, self.color)


class SfM(object):

    # INIT_DIRS = 'init_dirs'
    INTRINSICS_ANALYSIS = 'intrinsics_analysis'
    INTRINSICS_ANALYSIS_360 = 'intrinsics_analysis_360'
    COMPUTE_FEATURES = 'compute_features'
    COMPUTE_MATCHES = 'compute_matches'
    COMPUTE_MATCHES_360 = 'compute_matches_360'
    INCREMENTAL_RECONSTRUCTION = 'incremental_reconstruction'
    INCREMENTAL_RECONSTRUCTION2 = 'incremental_reconstruction2'
    GLOBAL_RECONSTRUCTION = 'global_reconstruction'
    COLORIZE_POINT_CLOUD = 'colorize_point_cloud'
    LOCALIZATION = 'localization'
    SPHERICAL_TO_CUBIC = 'spherical_to_cubic'
    EXPORT_TO_OPENMVS = 'export_to_openmvs'
    EXPORT_TO_OPENMVS_360 = 'export_to_openmvs_360'
    DENSIFY_POINT_CLOUD = 'densify_point_cloud'
    RECONSTRUCT_THE_MESH = 'reconstruct_the_mesh'
    RECONSTRUCT_THE_MESH_FROM_SPARSE_POINTS = 'reconstruct_the_mesh_from_sparse_points'
    REFINE_THE_MESH = 'refine_the_mesh'
    TEXTURE_THE_MESH = 'texture_the_mesh'
    CONVERT_SFM_FORMAT = 'convert_sfm_format'
    CONVERT_SFM_FORMAT_360 = 'convert_sfm_format_360'
    EXPORT_KEYPOINTS = 'export_keypoints'
    EXPORT_MATCHES = 'export_matches'
    EXPORT_TRACKS = 'export_tracks'

    def __init__(self, c: Config):
        self.config = c
        self.tasks = {}

        # --- MVG ---

        # 鏡頭 focal length:
        #   iPhone 11 Pro：1.54mm (0.5x), 4.25mm (1x), 6mm (2x)
        #   iPhone XS: 4.25mm (1x), 6mm (2x)
        #   iPhone XR: 4.25mm (1x)
        #   iPhone X: 4mm (1x), 6mm (2x)
        #
        # Sensor Width:
        #   iPhone 6, 7, 8, X: 4.89
        #   iPhone XR, XS, XS Max: 5.6
        #
        # Sensor Active Area:
        #   iPhone XS: 5.6mm x 4.2mm
        #
        # focal length in pixel = max(photo_width, photo_height) * focal_length_mm / sensor_width_mm
        #   iPhone X: max(1920, 1080) * 4.25 / 4.89 = 1668.71
        #   iPhone XR, 11 = max(1920, 1080) * 4.25 / 5.6 = 1457.14
        #   Insta360 Pro fisheye = 1000 (usable number through trial and error)
        #   All 360 images: 1

        self.add_task(Task(SfM.INTRINSICS_ANALYSIS, [
            join(c.openmvg_dir, 'openMVG_main_SfMInit_ImageListing'),
            '-i', c.input_dir,
            '-o', c.matches_dir,
            '-d', c.cameradb,
            '-c', c.camera_model,
            '-f', '1457.14'
        ], Fore.LIGHTBLUE_EX))

        self.add_task(Task(SfM.INTRINSICS_ANALYSIS_360, [
            join(c.openmvg_dir, 'openMVG_main_SfMInit_ImageListing'),
            '-i', c.input_dir,
            '-o', c.matches_dir,
            '-d', c.cameradb,
            '-c', c.camera_model,
            '-f', '1'
        ], Fore.LIGHTBLUE_EX))

        self.add_task(Task(SfM.COMPUTE_FEATURES, [
            join(c.openmvg_dir, 'openMVG_main_ComputeFeatures'),
            '-i', join(c.matches_dir, 'sfm_data.json'),
            '-o', c.matches_dir,
            '-f', '0',              # force to recompute data (default 0: false)
            '-m', 'AKAZE_FLOAT',    # describerMethod: SIFT / AKAZE
            '-u', '0',              # upright (0: don't assume upright; calculate feature orientation)
            '-p', 'ULTRA',          # describerPreset: NORMAL (threshold = 0.04) / HIGH (threshold = 0.01) / ULTRA (upscale once, threshold = 0.01)
            '-n', c.nproc           # number of threads
        ], Fore.BLUE))

        self.add_task(Task(SfM.COMPUTE_MATCHES, [
            join(c.openmvg_dir, 'openMVG_main_ComputeMatches'),
            '-i', join(c.matches_dir, 'sfm_data.json'),
            '-o', c.matches_dir,
            '-f', '0',              # force to recompute data (default 0: false)
            '-r', '0.8',            # ratio to discard non-meaningful matches (default: 0.8)
            '-g', 'f',              # geometric model f: fundamental (default) / e: essential / a: essential angular / o: essential orthographic / h: homography
            '-v', '-1',             # video mode matching (default -1: exhaustive, >0: continuous)
            '-n', 'FASTCASCADEHASHINGL2',  # nearest matching method HNSWL2 / ANNL2 / CASCADEHASHINGL2 / FASTCASCADEHASHINGL2
            '-m', '0',              # guided match, use the found model to improve correspondence (default 0: false)
        ], Fore.BLUE))

        self.add_task(Task(SfM.COMPUTE_MATCHES_360, [
            join(c.openmvg_dir, 'openMVG_main_ComputeMatches'),
            '-i', join(c.matches_dir, 'sfm_data.json'),
            '-o', c.matches_dir,
            '-f', '0',              # force to recompute data (default 0: false)
            '-r', '0.8',            # ratio to discard non-meaningful matches (default: 0.8)
            '-g', 'a',              # geometric model f: fundamental (default) / e: essential / a: essential angular / o: essential orthographic / h: homography
            '-v', '10',             # video mode matching (default -1: exhaustive, >0: continuous)
            '-n', 'FASTCASCADEHASHINGL2',  # nearest matching method HNSWL2 / ANNL2 / CASCADEHASHINGL2 / FASTCASCADEHASHINGL2
            '-m', '0',              # guided match, use the found model to improve correspondence (default 0: false)
        ], Fore.BLUE))

        # incremental v1 版如果沒給 a, b 檔名做 initial pair，會自動選 baseline 最大 (距離最遠) 的一對照片作為 initial pair
        self.add_task(Task(SfM.INCREMENTAL_RECONSTRUCTION, [
            join(c.openmvg_dir, 'openMVG_main_IncrementalSfM'),
            '-i', join(c.matches_dir, 'sfm_data.json'),
            '-m', c.matches_dir,
            '-o', c.sfm_dir,
            '-c', c.camera_model,   # camera model
            '-f', 'ADJUST_ALL',     # refine intrinsic. Adjust all or just principal point / distortion / focal length
            '-t', '3',              # triangulation method (default 3: INVERSE_DEPTH_WEIGHTED_MIDPOINT)
        ], Fore.BLUE))

        # incremental v2 版本不需要給 a,b 檔，而是靠 MAX_PAIR 或 STELLAR 方法選 initial pair
        self.add_task(Task(SfM.INCREMENTAL_RECONSTRUCTION2, [
            join(c.openmvg_dir, 'openMVG_main_IncrementalSfM2'),
            '-i', join(c.matches_dir, 'sfm_data.json'),
            '-m', c.matches_dir,
            '-o', c.sfm_dir,
            '-c', c.camera_model,   # camera model
            '-S', 'STELLAR',        # initializer MAX_PAIR / AUTO_PAIR (not implemented) / STELLAR
            '-f', 'ADJUST_ALL',     # refine intrinsic. Adjust all or just principal point / distortion / focal length
            '-t', '3',              # triangulation method (default 3: INVERSE_DEPTH_WEIGHTED_MIDPOINT)
        ], Fore.BLUE))

        self.add_task(Task(SfM.GLOBAL_RECONSTRUCTION, [
            join(c.openmvg_dir, 'openMVG_main_GlobalSfM'),
            '-i', join(c.matches_dir, 'sfm_data.json'),
            '-m', c.matches_dir,
            '-o', c.sfm_dir
        ], Fore.BLUE))

        self.add_task(Task(SfM.COLORIZE_POINT_CLOUD, [
            join(c.openmvg_dir, 'openMVG_main_ComputeSfM_DataColor'),
            '-i', join(c.sfm_dir, 'sfm_data.bin'),
            '-o', join(c.sfm_dir, 'scene.ply')
        ], Fore.BLUE))

        self.add_task(Task(SfM.LOCALIZATION, [
            join(c.openmvg_dir, 'openMVG_main_SfM_Localization'),
            '-i', join(c.sfm_dir, 'sfm_data.bin'),
            '-m', c.matches_dir,
            '-o', c.loc_dir,
            '-u', c.loc_matches_dir,
            '-q', c.loc_query_dir,
            '-s', 'ON',             # ON: Use the intrinsics in sfm_data.bin
            '-r', '5',              # Optimization threshold
            '-n', c.nproc           # number of threads
        ], Fore.BLUE))

        # --- Convert MVG (result to MVS, images to perspective/undistorted) ---

        self.add_task(Task(SfM.SPHERICAL_TO_CUBIC, [
            join(c.openmvg_dir, 'openMVG_main_openMVGSpherical2Cubic'),
            '-i', join(c.sfm_dir, 'sfm_data.bin'),
            '-o', c.sfm_dir,
        ], Fore.YELLOW))

        self.add_task(Task(SfM.EXPORT_TO_OPENMVS, [
            join(c.openmvg_dir, 'openMVG_main_openMVG2openMVS'),
            '-i', join(c.sfm_dir, 'sfm_data.bin'),
            '-o', join(c.mvs_dir, 'scene.mvs'),
            '-d', join(c.output_dir, 'undistorted'),    # output undistorted images
            '-n', c.nproc                               # number of threads
        ], Fore.YELLOW))

        self.add_task(Task(SfM.EXPORT_TO_OPENMVS_360, [
            join(c.openmvg_dir, 'openMVG_main_openMVG2openMVS'),
            '-i', join(c.sfm_dir, 'sfm_data_perspective.bin'),
            '-o', join(c.mvs_dir, 'scene.mvs'),
            '-d', join(c.output_dir, 'perspective'),    # output perspective images
            '-n', c.nproc                               # number of threads
        ], Fore.YELLOW))

        # --- MVS ---

        self.add_task(Task(SfM.DENSIFY_POINT_CLOUD, [
            join(c.openmvs_dir, 'DensifyPointCloud'),
            '-w', c.mvs_dir,
            'scene.mvs',                                # input 檔名，同時產生 scene_dense.mvs
            '--process-priority', c.process_priority,
            '--max-threads', c.nproc,
            # '--dense-config-file', 'Densify.ini', # we supply options directly. don't need config file
            '--resolution-level', '1',          # 要 scale down 幾次 (>> level) 也就是長寬 /2 幾次，default: 1，若 0 則原尺寸，除非超過 max-resolution
            '--max-resolution', '3200',         # 最大邊 max pixel, default: 3200
            '--min-resolution', '640',          # 最大邊 min pixel, default: 640。如果上面 resolution-level 有做 scale，結果不會小於這個值
            '--estimate-colors', '2',           # default: 2 (把 vertex 上色，只是為了看，後面沒有用到這個顏色值)
            '--estimate-normals', '0',          # default: 0 (把 vertex 加上 normal，也只是為了看)
            '--fusion-mode', '0',               # default: 0 = depth maps fuse mode，有另個跑法是先後做 fusion -1, fusion -2 兩個步驟
            '--optimize', '7',                  # binary flags, 7 = enable all optimization
        ], Fore.MAGENTA))

        self.add_task(Task(SfM.RECONSTRUCT_THE_MESH, [
            join(c.openmvs_dir, 'ReconstructMesh'),
            '-w', c.mvs_dir,
            'scene_dense.mvs',
            '--process-priority', c.process_priority,
            '--max-threads', c.nproc,
        ], Fore.MAGENTA))

        self.add_task(Task(SfM.RECONSTRUCT_THE_MESH_FROM_SPARSE_POINTS, [
            join(c.openmvs_dir, 'ReconstructMesh'),
            '-w', c.mvs_dir,
            '-i', 'scene.mvs',
            '-o', 'scene_dense_mesh.mvs',       # directly output name as it is densified
        ], Fore.MAGENTA))

        self.add_task(Task(SfM.REFINE_THE_MESH, [
            join(c.openmvs_dir, 'RefineMesh'),
            '-w', c.mvs_dir,
            'scene_dense_mesh.mvs',
            '--process-priority', c.process_priority,
            '--max-threads', c.nproc,
            '--scales', '2'
        ], Fore.LIGHTMAGENTA_EX))

        self.add_task(Task(SfM.TEXTURE_THE_MESH, [
            join(c.openmvs_dir, 'TextureMesh'),
            '-w', c.mvs_dir,
            'scene_dense_mesh_refine.mvs',
            '--process-priority', c.process_priority,
            '--max-threads', c.nproc,
            '--resolution-level', '0',          # 要 scale down 幾次 (>> level) 也就是長寬 /2 幾次，default: 0 原尺寸，除非小於 min-resolution
            '--min-resolution', '640',          # 最大邊 min pixel, default: 640。如果上面 resolution-level 有做 scale，結果不會小於這個值
            '--decimate', '0.5',
        ], Fore.LIGHTMAGENTA_EX))

        # --- Other openMVG utilities ---

        self.add_task(Task(SfM.CONVERT_SFM_FORMAT, [
            join(c.openmvg_dir, 'openMVG_main_ConvertSfM_DataFormat'),
            'binary',
            '-i', join(c.sfm_dir, 'sfm_data.bin'),
            '-o', join(c.sfm_dir, 'sfm_data_all.json'),
            '-V',                               # Export views
            '-I',                               # Export intrinsics
            '-E'                                # Export extrinsics
        ], Fore.YELLOW))                        # Do not export structure (-S) and control points (-C)

        self.add_task(Task(SfM.CONVERT_SFM_FORMAT_360, [
            join(c.openmvg_dir, 'openMVG_main_ConvertSfM_DataFormat'),
            'binary',
            '-i', join(c.sfm_dir, 'sfm_data_perspective.bin'),
            '-o', join(c.sfm_dir, 'sfm_data_perspective_all.json'),
            '-V',                               # Export views
            '-I',                               # Export intrinsics
            '-E'                                # Export extrinsics
        ], Fore.YELLOW))                        # Do not export structure (-S) and control points (-C)

        self.add_task(Task(SfM.EXPORT_KEYPOINTS, [
            join(c.openmvg_dir, 'openMVG_main_exportKeypoints'),
            '-i', join(c.sfm_dir, 'sfm_data.bin'),
            '-o', c.keypoints_svg_dir,
            '-d', c.matches_dir,
        ], Fore.YELLOW))

        self.add_task(Task(SfM.EXPORT_MATCHES, [
            join(c.openmvg_dir, 'openMVG_main_exportMatches'),
            '-i', join(c.sfm_dir, 'sfm_data.bin'),
            '-o', c.matches_svg_dir,
            '-d', c.matches_dir,
            '-m', join(c.matches_dir, 'matches.f.bin'),
        ], Fore.YELLOW))

        self.add_task(Task(SfM.EXPORT_TRACKS, [
            join(c.openmvg_dir, 'openMVG_main_exportTracks'),
            '-i', join(c.sfm_dir, 'sfm_data.bin'),
            '-o', c.tracks_svg_dir,
            '-d', c.matches_dir,
            '-m', join(c.matches_dir, 'matches.f.bin'),
        ], Fore.YELLOW))

        # --- Pipeline definitions ---

        self.pipelines = {
            'incremental': [
                SfM.INTRINSICS_ANALYSIS,
                SfM.COMPUTE_FEATURES,
                SfM.COMPUTE_MATCHES,
                SfM.INCREMENTAL_RECONSTRUCTION,
                SfM.COLORIZE_POINT_CLOUD,
                SfM.EXPORT_TO_OPENMVS,
                SfM.DENSIFY_POINT_CLOUD,
                SfM.RECONSTRUCT_THE_MESH,
                SfM.REFINE_THE_MESH,
                SfM.TEXTURE_THE_MESH,
                SfM.CONVERT_SFM_FORMAT,
            ],
            'incremental_360': [
                SfM.INTRINSICS_ANALYSIS_360,
                SfM.COMPUTE_FEATURES,
                SfM.COMPUTE_MATCHES_360,
                SfM.INCREMENTAL_RECONSTRUCTION,
                SfM.COLORIZE_POINT_CLOUD,
                SfM.SPHERICAL_TO_CUBIC,
                SfM.EXPORT_TO_OPENMVS_360,
                SfM.DENSIFY_POINT_CLOUD,
                SfM.RECONSTRUCT_THE_MESH,
                SfM.REFINE_THE_MESH,
                SfM.TEXTURE_THE_MESH,
                SfM.CONVERT_SFM_FORMAT,
                SfM.CONVERT_SFM_FORMAT_360,
            ],
            'global': [
                SfM.INTRINSICS_ANALYSIS,
                SfM.COMPUTE_FEATURES,
                SfM.COMPUTE_MATCHES,
                SfM.GLOBAL_RECONSTRUCTION,
                SfM.COLORIZE_POINT_CLOUD,
                SfM.EXPORT_TO_OPENMVS,
                SfM.DENSIFY_POINT_CLOUD,
                SfM.RECONSTRUCT_THE_MESH,
                SfM.REFINE_THE_MESH,
                SfM.TEXTURE_THE_MESH,
                SfM.CONVERT_SFM_FORMAT,
            ],
            'sparse': [
                SfM.INTRINSICS_ANALYSIS,
                SfM.COMPUTE_FEATURES,
                SfM.COMPUTE_MATCHES,
                SfM.INCREMENTAL_RECONSTRUCTION2,
                SfM.COLORIZE_POINT_CLOUD,
                SfM.EXPORT_TO_OPENMVS,
                SfM.RECONSTRUCT_THE_MESH_FROM_SPARSE_POINTS,
                SfM.REFINE_THE_MESH,
                SfM.TEXTURE_THE_MESH,
                SfM.CONVERT_SFM_FORMAT,
            ],
            'localization': [
                SfM.LOCALIZATION,
            ],
            'mvs': [
                SfM.DENSIFY_POINT_CLOUD,
                SfM.RECONSTRUCT_THE_MESH,
                SfM.REFINE_THE_MESH,
                SfM.TEXTURE_THE_MESH,
            ],
            'export_sfm_data': [
                SfM.CONVERT_SFM_FORMAT,
            ],
            'export_sfm_data_360': [
                SfM.CONVERT_SFM_FORMAT,
                SfM.CONVERT_SFM_FORMAT_360,
            ],
            'export_svg': [
                SfM.EXPORT_KEYPOINTS,
                SfM.EXPORT_MATCHES,
                SfM.EXPORT_TRACKS,
            ]
        }

    def add_task(self, task: Task):
        self.tasks[task.name] = task

    def run_task(self, pipeline: str, name: str, step: int, max_step: int) -> TaskResult:
        if name not in self.tasks:
            raise Exception(f'task "{name}" not found')

        task = self.tasks[name]
        header = f'{pipeline} {step}/{max_step} "{name}"'
        self.print_separator(f'{header} begin', task.color)
        result = task.run(pipeline=pipeline, step=step, log_dir=self.config.loc_dir if (self.config.loc_dir and pipeline=='localization') else self.config.output_dir)
        if result.return_code == 0:
            self.print_separator(f'{header} completed in {datetime.timedelta(seconds=result.t)}', task.color)
        else:
            self.print_separator(f'{header} return error code {result.return_code} in {datetime.timedelta(seconds=result.t)}', Fore.RED)
        return result

    def run_pipeline(self, name: str):
        if name not in self.pipelines:
            raise Exception(f'pipeline "{name}" not found')

        error = False
        results: [TaskResult] = []
        t = time.time()
        pipeline = self.pipelines[name]
        for idx, task_name in enumerate(pipeline):
            result = self.run_task(pipeline=name, name=task_name, step=idx, max_step=len(pipeline))
            if result.return_code != 0:
                error = True
                break
                # sys.exit(f'{Fore.RED}Abort: Process returns error code: {result.return_code}{Style.RESET_ALL}')
            results.append(result)
        t = time.time() - t

        if error:
            pipeline_msg = f'{Fore.RED}[ERROR] pipeline "{name}" ended with error in {datetime.timedelta(seconds=t)}{Style.RESET_ALL}'
        else:
            pipeline_msg = f'{Fore.LIGHTGREEN_EX}[OK] pipeline "{name}" completed in {datetime.timedelta(seconds=t)}{Style.RESET_ALL}'
        print(pipeline_msg)

        # print pipeline run log
        with open(join(self.config.output_dir, f'{name}_pipeline.log'), 'w') as f:
            f.write(f'[{os.getcwd()}] ')
            f.write(' '.join(sys.argv))
            f.write("\n\n")
            for result in results:
                f.write(f'{result.color}--- step {result.step} "{result.name}"{Style.RESET_ALL}' + "\n")
                f.write(f'    ' + ' '.join(result.cmd) + "\n")
                f.write(f'    completed in {datetime.timedelta(seconds=result.t)}')
                if result.return_code != 0:
                    f.write(f' *** with error code {result.return_code}')
                f.write("\n\n")
            f.write(pipeline_msg)
            f.write("\n")

    @staticmethod
    def print_separator(msg: str, color=Fore.GREEN, width=80):
        print(f'{color}{"-"*6} {msg} {color}{"-"*(width-8-len(msg))}{Style.RESET_ALL}')


if __name__ == '__main__':
    # get terminal size
    rows, cols = os.popen('stty size', 'r').read().split()
    nproc = str(multiprocessing.cpu_count())

    parser = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=38, width=int(cols))
    )
    parser.add_argument(
        '-c', '--camera_model', dest='camera_model', type=str, metavar='NUM', default='3',
        help='camera model no.'
    )
    parser.add_argument(
        '--cameradb', dest='cameradb', type=str, metavar='FILE', default=Config.cameradb,
        help='camera sensor db file'
    )
    parser.add_argument(
        '--openmvg_dir', dest='openmvg_dir', type=str, metavar='DIR', default=Config.openmvg_dir,
        help='binary directory of OpenMVG'
    )
    parser.add_argument(
        '--openmvs_dir', dest='openmvs_dir', type=str, metavar='DIR', default=Config.openmvs_dir,
        help='binary directory of OpenMVS'
    )
    parser.add_argument(
        '-n', '--nproc', dest='nproc', type=str, metavar='NUM', default=nproc,
        help='max number of threads used for execution'
    )
    parser.add_argument(
        '-p', '--pipeline', dest='pipeline', type=str, metavar='NAME', default='incremental',
        help='name of the pipeline to execute'
    )
    parser.add_argument(
        '-l', '--localization_dir', dest='loc_dir', type=str, metavar='DIR', default='',
        help='directory for the localization'
    )
    parser.add_argument(
        '--clean', dest='clean', action='store_true', default=False,
        help='clean output_dir before running'
    )
    parser.add_argument(
        'input_dir', type=str,
        help='directory of input files'
    )
    parser.add_argument(
        'output_dir', type=str, nargs='?', default=argparse.SUPPRESS,
        help='directory of output files (default: ${input_dir}_out)'
    )
    args = parser.parse_args()
    config = Config(args)

    # cleanup
    if args.clean:
        if len(config.output_dir) < 5:
            print(f'warning: will not remove directory {config.output_dir}')    # 防呆
        else:
            shutil.rmtree(config.output_dir, ignore_errors=True)

    # create directories
    os.makedirs(config.matches_dir, mode=0o755, exist_ok=True)
    os.makedirs(config.sfm_dir, mode=0o755, exist_ok=True)
    os.makedirs(config.mvs_dir, mode=0o755, exist_ok=True)
    os.makedirs(config.keypoints_svg_dir, mode=0o755, exist_ok=True)
    os.makedirs(config.matches_svg_dir, mode=0o755, exist_ok=True)
    os.makedirs(config.tracks_svg_dir, mode=0o755, exist_ok=True)
    if config.loc_dir and config.loc_query_dir:
        os.makedirs(config.loc_matches_dir, mode=0o755, exist_ok=True)

    # run
    t1 = time.time()
    sfm = SfM(config)
    sfm.run_pipeline(args.pipeline)
    t2 = time.time()
