# SfM Lab

先把 container 跑起來，再 bash 進 container 手動啟動 pipeline。

## 本機執行 Container

直接用 docker 跑 container 在 local 電腦上：

```bash
git submodule update --init --recursive
docker build -t sfm .
docker run -it --rm sfm
```

## K8S 上執行 Container

不需自己 build docker image，已經用 CI build 好在 gitlab 上了，不過要先在 k8s 建立名為 `sfm-lab-registry-read` 的 secret 來 pull image：

```bash
kubectl create secret docker-registry sfm-lab-registry-read \
  --docker-server=registry.corp.ailabs.tw \
  --docker-username=gitlab+deploy-token-455 \
  --docker-password=Vm3ohLzNHqZh4cNNYs-5
```

建好 secret 就可以用 `./sfm-lab.sh` script 來操作 job 的建立：

```bash
./sfm-lab.sh k8srun [k8s host name]  # 建立新 pod，譬如 ./sfm-lab.sh k8srun lab3-g0
./sfm-lab.sh k8sdel [k8s host name]  # 刪除 pod
./sfm-lab.sh k8sls                   # 列出相關 pod 名稱
./sfm-lab.sh k8sbash [pod name]      # 進到 pod 裡面 bash
```

`sfm-lab.sh k8srun` 會根據主機名稱，建立名為 `sfm-[k8s host name]` 的 job 在個人 namespace 裡。然後可以用 `sfm-lab.sh k8sls` 列出 pod 的名稱，再用 `sfm-lab.sh k8sbash` 連進 pod 裡面。

## Recon3D pipeline

### How to run the pipeline

進到 container 內後，打開 `/root/src/recon3d/run_project.py`，選擇要跑的 project type 和要用的 config 檔：

```py
# 跑 reconstruction project 就用 ReconProj 這個 class
prj = ReconProj('/volume/cpl-dev/sfm/recon3d/example_reconstruction.toml')
# 跑 localization project 就用 LocProj 這個 class
prj = LocProj('/volume/cpl-dev/sfm/recon3d/example_localization.toml')
prj.run_pipeline()
```

Config 檔是 toml 格式，語法跟 Python 幾乎一樣。Reconstruction 和 localization 的 config 定義不一樣，請從放在 `/root/src/recon3d/` 裡的 `example_reconstruction.toml` 或 `example_localization.toml` 開始修改。

以下是一個基本的 ReconProj config 的 `[project]` 定義：

```py
[project]
type = 'ReconProj'                  # 必須與 run_project.py 中使用的 class 一樣，否則 return error
name = 'example_recon_project'      # Project 名，也是 folder 名，請取容易辨識的好名字。
parent_dir = '/some_parent_folder'  # Project object 會在 initialization 時在 parent_dir 裡開一個以 name 命名的 folder
n_thread = 20                       # 任何步驟可以指定 maximum thread 數量的都會抓這個變數
process_prio = 2                    # 任何步驟可以指定 process priority 的都會抓這個變數
pipeline = [                        # 依照順序排列所有要執行的 task。名稱定義在 project.py 的 task_classes 裡（每個名稱都對應到一個 task class）
    'EXTRACT_IMAGES',
    'PREPROCESS_IMAGES',
    'SfM_INITIALIZE',
    'SfM_COMPUTE_FEATURES',
    'SfM_COMPUTE_MATCHES',
    'SfM_INCREMENTALLY_RECONSTRUCT',
    'SfM_EXPORT_RESULTS',
    'SfM_EXPAND_FPS',
    'MVS_DENSIFY_POINT_CLOUD',
    'MVS_GENERATE_POISSON_MESH',
    # 'MVS_RECONSTRUCT_MESH',
    # 'MVS_REFINE_MESH',
    # 'MVS_TEXTURE_MESH',
    'SIMPLIFY_MESH',
    'CORRECT_LEVEL',
    'GENERATE_FOOTPRINTS',
    'ZERO_INITIAL_POSE',
    'EXPORT_SCENE_TO_APP'
]
```

設定好 `run_project.py` 和 `config toml` 之後，執行 `./run_project.py` 即可啟動 pipeline。在 project 底下的 log folder 有 project 本身和每一個 task 的 log file，這些 log 都會在 pipeline 執行時即時更新，就算 terminal 斷線，只要重新進 pod 打開 log 就可以繼續監測 pipeline 進度。

若 pipeline 在執行途中中斷，可在 project folder 底下的 `done_tasks.json` 中找到已執行完畢的 tasks 及執行完畢的時間。同樣的 pipeline (i.e. 一模一樣的 config toml) 只要直接重新執行，程式就會讀取 done_tasks.json 並跳過已執行完畢的 tasks，從上次失敗的 task 再開始。

Pipeline 的重要產物都會存在 project 底下的 data folder，一般使用者需要的資料都在裡面；其他中間產物會存在 subsidiaries folder 裡，通常是 debug 時使用。

### Workflow of Recon3D

1. Initialize project object.
    1. Load config file.
    2. (Localization project only) Distinguish parital project types.
    3. Define directories (both dynamic and static ones).
    4. Setup loggers (the project and all tasks have their own logger as member of the object).
    5. Read done tasks.
    6. Initialize all tasks, which are the members of the project.
        - Task classes take project config and folder paths as input arguments, and pick whatever they need to set as member.
2. Run the tasks sequentially.
    - There is no I/O between tasks. They do not communicate with each other at all. A task just reads the files it needs from the project folder, and saves the resulting files back to the project folder. So if you know exactly what the inputs and outputs are, you can "cheat" the tasks.

### Image project and its folder structure

```
.
├── img
│   └── [%04d].jpg
├── img_upsmpl
│   └── [%04d].jpg
├── log_[project creation datetime]
│   ├── record.toml
│   ├── Project.log
│   └── step_[%02d]_[task class name].log
└── routes.json
```

Image project is the very beginning of the whole Recon3D. It should contain all processed images needed for further reconstruction and metadata, no matter if a slice of a large scene or a whole scene or an object is to be reconstructed.

There are 3 things in a image project folder: Image folder, upsampled image folder, and `routes.json`. The images for SfM are in img folder, and the images to extend FPS (needed only for scenes) are in img_upsmpl. If no FPS extension should be done, then img_upsmpl folder doesn't need to exist. Lastly there should be a `routes.json`, which stores if this project is source of a scene or an object, and other information such as image names, timestamps, poses, etc.. Details see the example routes in repo.


### Reconstruction project and its folder structure

```
.
├── data_[project name]
│   ├── img_prj (symbolic link to image project)
│   │   └── ... (see folder structure of image project)
│   ├── matches
│   │   └── ...
│   ├── sfm_data.bin
│   ├── sfm_data_original.json
│   ├── sfm_data_transformed.json
│   ├── transformation.json
│   ├── transformed_mesh.ply
│   ├── transformed_mesh.obj
│   ├── (scene_dense_mesh_refine_texture.png)
│   └── routes.json
├── log_[project creation datetime]
│   ├── record.toml
│   ├── Project.log
│   └── step_[%02d]_[task class name].log
├── subsidiaries
│   └── (All other resulting files of every step)
└── done_tasks.json
```

Important files:

- `sfm_data.bin`: The result file of OpenMVG reconstruction (IncreRecon), it will never be modified.
- `sfm_data_original.json`: The json format of sfm_data.bin.
- `sfm_data_transformed.json`: The sfm data that undergoes FPS expanding localization (ExpandFPS), all coordinate transformations, and footprint generation (footprints are stored in it). During pipeline execution, it will be updated by the tasks that do coordinate transformation.
- `transformation.json`: All coordinate transformations which the sfm data and 3D model undergo are stored in it. It also contains an concatenated transformation which is equivalent to the combination of all transformations applied in sequence.
- `transformed_mesh.ply`: The 3D model in ply that undergoes all coordinate transformations. During pipeline execution, it will be updated by the tasks that do coordinate transformation.
- `transformed_mesh.obj`: The obj format of transformation.ply. But it will not be transformed once it is generated. So everytime you need the obj format, you should run the export task (ExportScene2App) again to ensure that it's the right one.
- `routes.json`: The camera poses and footprints stored in the format that the frontend needs. But it will not be transformed once it is generated. So everytime you need this file, you should run the export task (Export1Model2App) again to ensure that it's the right one.
- `done_tasks.json`: It stores all tasks that have finished successfully, and the timestamp when they have finished. The project object will read it and skip the tasks that are list in this file. If you hack this file, you can force some steps to be executed, even they are already done successfully.
- `img_upsmpl`: The upsampled video frames. It will only be used to expand the FPS right after reconstruction step, so that the camera poses (and thus footprints) are dense enough for frontend.

### Localization project and its folder structure

```
.
├── data_[partial project name]
│   └── ... (see folder structure of reconstruction project)
├── loc__[sub partial project name]__2__[main partial project name]
│   ├── matches
│   │   └── ...
│   ├── query
│   │   └── sub_[%04d].jpg
│   ├── sfm_data_original_main.json
│   ├── sfm_data_original_sub.json
│   ├── sfm_data_sub2main.json
│   ├── loc_trans_sub2main.json
│   └── ...
├── log_[project creation datetime]
│   ├── record.toml
│   ├── Project.log
│   └── step_[%02d]_[task class name].log
└── done_tasks.json
```

Important files:

- `data_[partial project name]`: This is exactly the data folder of the reconstruction project. There are 2 or more data folders in localization project, each of which is called "partial project" in the context of localization and corresponds to a partial 3D model.
- `loc__[sub partial project name]__2__[main partial project name]`: This contains all files related to the localization and transformation from sub partial prject to main partial project. There are 1 or more loc folders in localiztion project. With these loc folders, one can understand what transformations every partial project has undergone and in what sequence.
- `query`: The images that are in img folder of the sub partial project will be copied here and prepend with "sub_".
- `sfm_data_original_main.json`: The json format of the sfm_data.bin of the main partial project. It will never be modified.
- `sfm_data_original_sub.json`: The json format of the sfm_data.bin of the main partial project. It will never be modified.
- `sfm_data_sub2main.json`: The resulting sfm data of localization task. The entries that are the same as sfm_data_original_main.json will be removed. It will never be modified.
- `loc_trans_sub2main.json`: The transformation found using the result of localization task is saved here. It will never be modified.

### How localization project works

In the context of localization, a reconstruction project is a project that contains only 1 partial project (i.e. 1 data folder), and a localization project is a project that contains 2 or more partial projects (i.e. multiple data folder). In essence, localization simply takes 2 partial projects (a & b) in 2 projects (A & B) and calculate the transformation needed to merge the partial project b to the other partial project a, and then apply this transformation to all partial projects that are in project B (in practice these partial projects will be copied out and then transformed, so that the original things do not disappear).

The only difference the type of project (reconstruction or localization) makes is whether a new folder should be created as new localiation project. In total there are 4 cases:

1. **Sub partial project b0 is in reconstruction project B, main partial project a0 is in reconstruction project A:**
    - A new folder of `[project][name]` in config toml will be created as the new localization project.
    - After the transformation is calculated, both a0 and b0 will be copied into the new localization project folder.
    - And the transformation will be applied on the copied sub partial project b0.
2. **Sub partial project b0 is in localization project B, main partial project a0 is in reconstruction project A:**
    - A new folder of `[project][name]` in config toml will be created as the new localization project.
    - After the transformation is calculated, the main partial project a0 will be copied into the new loaclization project folder. And all partial projects (b1-bm) which are in the same parent folder as the sub partial project b0 will also be copied into the new localization project folder.
    - And the transformation will be applied on all of the copied sub partial projects (b0-bm).
3. **Sub partial project b0 is in reconstruction project B, main partial project a0 is in localization project A:**
    - No new project folder will be created, the existing localization project A will be used.
    - The `[project][name]` in config toml must be the same as the localization project A, otherwise the pipeline will stop directly without running any tasks.
    - After the transformation is calculated, only b0 will be copied into the existing localization project A.
    - And the transformation will be applied on the copied sub partial project b0.
4. **Sub partial project is in localization project, main partial project is in localization project:**
    - No new project folder will be created, the existing localization project A will be used.
    - The `[project][name]` in config toml must be the same as the localization project A, otherwise the pipeline will stop directly without running any tasks.
    - After the transformation is calculated, all partial projects (b1-bm) which are in the same parent folder as the sub partial project b0 will be copied into the existing localization project A.
    - And the transformation will be applied on all of the copied sub partial projects (b0-bm).

### Export SfM routes to GPS geojson
Design Doc for Exporting SfM routes to GPS geojson
[link](https://docs.google.com/presentation/d/1RfNZ3YlKSb5bEV-Ez-dvGeCe7_bo6ixQIQe2G3_SdCA/edit#slide=id.p)
