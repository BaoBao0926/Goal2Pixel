
# Goal2Pixel: Grounding Goals to Pixels for Vision-and-Language Naivgaiton

[Project Page](https://baobao0926.github.io/Goal2Pixel/) | [Paper](https://arxiv.org/pdf/2606.01621)

This repository currently contains the training-data rollout and VLM fine-tuning
pipeline for Goal2Pixel. Evaluation code is intentionally not included in this
release snapshot.


## 1. Environment Setup


```bash
# build habitat-sim from source
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
export CUDACXX=/usr/local/cuda/bin/nvcc
export CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
python setup.py install --with-cuda --bullet --headless
cd ..

# install habitat-lab and habitat-baselines
cd habitat-lab
pip install -e habitat-lab  # install habitat_lab
pip install -e habitat-baselines  # install habitat_baselines
cd ..

# For cuda 12.1
# some important packages version
pip install numpy==1.26.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install transformers==4.37.2 deepspeed==0.14.4 accelerate==0.33.0
pip install orjson matplotlib wandb
pip install timm==0.9.12 peft==0.10.0 scipy==1.15.3 datasets==4.0.0

# download pretrained InternVL model
cd InternVL_cleaned
mkdir pretrained/
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL3-2B --local-dir InternVL3-2B
```



## 2.Prepare Training Data

Training data structure is shown in [2.0 Data Structure](#20-data-structure).

You can use [2.1 Download From Huggingface](#21-download-from-huggingface) or [2.2 Rollout All Trainig Data by Scripts](#22-rollout-training-data-by-scripts).

### 2.0 Data Structure

Place the files under the project root with the following structure:

```text
Goal2Pixel/
├── training_data/
│   ├── data_mp3d_r2r/                          # data for R2R-CE
│   │   └── v1-3/                               # Version: v1-3
│   │       ├── annotations/                    # total annotation files record all info for R2R
│   │       │   ├── <scene_id>/                 # Matterport3D scene id, e.g. 17DRP5sb8fy
│   │       │   │   └── <episode_id>.json       # trajectory/episode id, e.g. 10154.json
│   │       ├── train_trajectory/               # train: train split / trajectory: collect for basic info
│   │       │   ├── annotations/                # annotation files record all info
│   │       │   ├── adjusted_annotations/       # adjusted annotation json files
│   │       │   ├── rgb/                        # current observation
│   │       │   ├── rgb_padded_down/            # current observation RGB with directive regions (padding region)
│   │       │   ├── local_occupancy_explore/    # local BEV map, visualization for GOAL2PIXEL
│   │       │   └── combined/                   # only for visualization
│   │       ├── train_pixel/               # train: train split / trajectory: collect for basic info
│   │       │   ├── annotations/
│   │       │   ├── adjusted_annotations/
│   │       ├── train_history/             # history representation assets
│   │       │   ├── annotations/
│   │       │   ├── annotation_mask/
│   │       │   └── his_tra_single_colored/     # use blue dot to draw the ViKeyMem trajectory overlay (history representation)
│   ├── data_mp3d_rxr/                          # similar to data_mp3d_r2r
│   │   ├── rxr_guide_en-US/                    # Dataset: RXR / Mode: Guide / LANGUAGES: EN-US
│   │   │   ├── annotations/
│   │   │   ├── train_trajectory/
│   │   │   ├── train_pixel/
│   │   │   └── train_history/
│   │   └── rxr_guide_en-IN/                    # LANGUAGES: EN-IN
│   │   │   ├── annotations/
│   │   │   ├── train_trajectory/
│   │   │   ├── train_pixel/
│   │   │   └── train_history/
│   └── jsonl/                                  # reconstructed jsonl files
│       ├── r2r/
│       │   ├── RGB.jsonl
│       │   ├── RGB_HisKFSingleColor.jsonl
│       │   └── ...
│       ├── rxr-en-US/
│       └── rxr-en-IN/
└── ...
```

Notes:

- `<scene_id>` is a Matterport3D scene id such as `17DRP5sb8fy`.
- `<episode_id>` is a trajectory/episode id such as `10154`.
- `<step_id>` is a zero-padded time step such as `001`, `002`, `003`.
- For Goal2Pixel training with `RGB_HisKFSingleColor*.jsonl`, the key assets used by the jsonl files are `train_trajectory/rgb_padded_down/...` and `train_history/his_tra_single_colored/...`.
- `annotations/` stores merged per-episode metadata, while `train_trajectory/adjusted_annotations`, `train_pixel/adjusted_annotations`, and `train_history/annotations` are intermediate files produced by the data-generation scripts.
- you may also see extra folders such as `depth/`, `semantic/`, `rgb_padded/`, `top_down_map/`, `full_occupancy_explore*/`, `his_tra_multip_colored/`, and `his_tra_without_colored/`. These are optional depending on which scripts and save flags you use. But these files are not used in GOAL2PIXEL.


### 2.1 Download From Huggingface

Download from [Huggingface](https://huggingface.co/datasets/Muyiaaaa/VLN1/tree/main).

Referring to [2.0 Data Structure](#20-data-structure), you should download all files except the visualization folders.

### 2.2 Rollout Training Data by Scripts

#### 2.2.1 Rollout Data

The rollout pipeline is modular:

- `r2r`, `rxr-en-US`, and `rxr-en-IN` are different dataset/split modes.
- `trajectory`, `pixel`, and `history` are different training-data types.
- You should run each `(mode, data_type)` pair separately, and it is normal to repeat this process multiple times for different datasets or regenerated versions.
- After each rollout, you must run the corresponding `adjust_*.py` script to normalize and fix the generated annotation files before merging them.

```bash
# trajectory data
bash run_write_replay_data.sh [r2r|rxr-en-US|rxr-en-IN] trajectory
python ./scripts/scripts_for_generating_data/adjust_trajectory.py --mode [r2r|rxr-en-US|rxr-en-IN]

# pixel data
bash run_write_replay_data.sh [r2r|rxr-en-US|rxr-en-IN] pixel
python ./scripts/scripts_for_generating_data/adjust_pixel.py --mode [r2r|rxr-en-US|rxr-en-IN]

# history data
bash run_write_replay_data.sh [r2r|rxr-en-US|rxr-en-IN] history
python ./scripts/scripts_for_generating_data/adjust_history.py --mode [r2r|rxr-en-US|rxr-en-IN]
```

#### 2.2.2 Merge Annotation Files

For now, we need construct ```data_mp3d_r2r/v1-3/annotations```; ```data_mp3d_rxr/rxr_guide_en-US/annotations``` ; ```data_mp3d_rxr/rxr_guide_en-IN/annotations```.

Use `python scripts/scripts_for_generating_data/merge_dataset.py --mode [r2r|rxr-en-US|rxr-en-IN]` to merge trajectory, pixel, and history annotations into a total annotation. It also adds extra step-level attributes used by the downstream reconstruction scripts.


#### 2.2.3 Reconstruct into InternVL Format

For VLM training, reconstruct the merged annotations into InternVL/Qwen-style jsonl files using
`python scripts/scripts_for_generating_data/dataset_reconstruction_for_intervl/dataset_reconstruct_for_internvl.py --mode [r2r|rxr-en-US|rxr-en-IN] --template RGB_HisKFSingleColor`.

This writes the final training annotations to `training_data/jsonl/[r2r|rxr-en-US|rxr-en-IN]/RGB_HisKFSingleColor.jsonl`.

#### 2.2.4 Create VLM Config Files

Create `configs_vlm/shell_data/[r2r|r2r_rxr|rxr]/*.json`, refering `configs_vlm/shell_data/r2r_rxr/RGB_HisKFSingleColor_ALL_trajectoryMASK.json`.



## 3. Training

### 3.1 VLM Fine-tuning


```bash
PROJECT_NAME="muyi_train" EXP_NAME="vlm_config_2B_Lora" CUDA_VISIBLE_DEVICES=0 GRADIENT_ACC=4 MAX_STEPS=5000 META_FILE_NAME='r2r_rxr/RGB_HisKFSingleColor_ALL_trajectoryMASK.json' ./run_vlm_finetune.sh

PROJECT_NAME="muyi_train" EXP_NAME="vlm_config_7B" CUDA_VISIBLE_DEVICES=0 GRADIENT_ACC=4 MAX_STEPS=5000 META_FILE_NAME='r2r_rxr/RGB_HisKFSingleColor_ALL_trajectoryMASK.json' ./run_vlm_finetune.sh
```


For a Slurm/Apptainer example, set `SIF` to your local container image and run
`sbatch run_in_apptainer.sh`. The script derives the project directory from its
own location unless `PROJ` is provided.




#### Parameter Explanation

| Parameter | Description |
|----------|-------------|
| `PROJECT_NAME` | Weights & Biases (wandb) project name |
| `EXP_NAME` | Used for **three purposes**:<br>1. Specifies which VLM config file to load from `configs_vlm/*.json`<br>2. Used as the wandb experiment name<br>3. Used as the prefix of the experiment folder under `all_log/my_experiments/*` |
| `CUDA_VISIBLE_DEVICES` | GPU ID(s) used for training |
| `GRADIENT_ACC` | Gradient accumulation steps |
| `MAX_STEPS` | Total number of optimizer update steps |
| `META_FILE_NAME` | Specifies the dataset metadata file located in `configs_vlm/shell_data/*.json` |

#### ⚠️ Important Notes on `GRADIENT_ACC`

When tuning **gradient accumulation**, the value **must be consistent** across the following locations:

```
InternVL_cleaned/internvl_chat/zero_stage3_config_acc*.json
└── gradient_accumulation_steps
configs_vlm/vlm_config_*.json
└── gradient_accumulation_steps
```


The following table shows recommended training settings on H100 GPUs.

| Parameter | GPU Number  | Max Token   | Acc * Avg_Num | optimization step | time(hours)  |
|---------- |-------------| ----------  | ----------    |-------------      | -------------|
| `2B`      | `1`         | `24000`     | `12*15.97`    | `13750`           | `60+h`       |
| `2B`      | `2`         | `30000`     | ` 4*20.90`    | `15500`           | `40h`        |
| `7B`      | `2`         | `6000`      | `24* 4.05`    | `13500`           | `110h`       |
| `7B`      | `4`         | `12000`     | ` 7* 7.71`    | `12200`           | `46h`        |
