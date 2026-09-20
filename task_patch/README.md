
### Downloading HM3D with the download utility
```aiignore
Token ID: 2e317c456e37d999
Token Secret: 2c82e5ff5ff146b543a13b3b6f868cce
python -m habitat_sim.utils.datasets_download --username 2e317c456e37d999 --password 2c82e5ff5ff146b543a13b3b6f868cce --uids hm3d_minival_v0.2 --data-path .
```

### View the whole house interactively in Habitat-Sim
```aiignore
# hm3d / ovon
habitat-viewer --dataset 'data/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json' TEEsavR23oF
```
```aiignore
# mp3d
habitat-viewer --dataset 'data/scene_datasets/mp3d/mp3d.scene_dataset_config.json' r1Q1Z4BcV1o
```

### all_episode_visualize.py 
This script visualizes all episodes in the dataset hm3d, ovon and mp3d accordingly, where **only one scene** is selected.

It is useful for understanding each trajectory in the dataset, including:
- start positions, (red)
- end positions, (blue)
- object goal,  (green)
- top down map.

### dataset_visualize.py 
1. Visualize the **human demonstrated trajectory** in each dataset
2. Visualize the **shortest path** from start to goal in each dataset

### Dataset Filter ([env_wrapper.py](utils/env_wrapper.py))
1. Start position on a small island of navigable space
2. Starting position is neither too close nor too far from a goal
3. The episode must have a valid navigable path
4. Elevation (height) change between points must be below a threshold (same floor)
5. Episodes that have no goal

**_Filrer result:_**
- **ovon train**: 
  - Number of episodes before cleaning: 6911470
  - Number of episodes after single floor cleaning: 6836195
  - Number of episodes after navigable space cleaning: 5318891
  - Number of episodes after geodesic distance and path finding cleaning: 2818955

- **ovon val_seen**: 
  - Number of episodes before cleaning: 3000
  - Number of episodes after single floor cleaning: 2974
  - Number of episodes after navigable space cleaning: 1895
  - Number of episodes after geodesic distance and path finding cleaning: 1302
  - Number of episodes after no goal cleaning: 1302
  - 
- **ovon val_unseen**: 
  - Number of episodes before cleaning: 3000
  - Number of episodes after single floor cleaning: 2954
  - Number of episodes after navigable space cleaning: 1928
  - Number of episodes after geodesic distance and path finding cleaning: 1336
  - Number of episodes after no goal cleaning: 1336
  - 
- **mp3d train**:
  - Number of episodes before cleaning: 70176
  - Number of episodes after replay length cleaning: 69751
  - Number of episodes after single floor cleaning: 52137
  - Number of episodes after navigable space cleaning: 24042
  - Number of episodes after geodesic distance and path finding cleaning: 10254
  - Number of episodes after no goal cleaning: 10254

- **mp3d few**:
  - Number of episodes before cleaning: 1470
  - Number of episodes after replay length cleaning: 1436
  - Number of episodes after single floor cleaning: 1191
  - Number of episodes after navigable space cleaning: 1186
  - Number of episodes after geodesic distance and path finding cleaning: 942
  - Number of episodes after no goal cleaning: 942

- **hm3d train**: 
  - Number of episodes before cleaning: 7196434
  - There is no reply for hm3d
  - Number of episodes after single floor cleaning: 7178762
  - Number of episodes after navigable space cleaning: 3038310
  - Number of episodes after geodesic distance and path finding cleaning: 2107702 (takes 55 mins)
  - Number of episodes after no goal cleaning: 2107702

- **hm3d val**: 
  - Number of episodes before cleaning: 1000
  - Number of episodes after single floor cleaning: 997
  - Number of episodes after navigable space cleaning: 581
  - Number of episodes after geodesic distance and path finding cleaning: 421
  - Number of episodes after no goal cleaning: 421

- **hm3d minival**:
  - Number of episodes before cleaning: 83
  - Number of episodes after single floor cleaning: 83
  - Number of episodes after navigable space cleaning: 34
  - Number of episodes after geodesic distance and path finding cleaning: 28
  - Number of episodes after no goal cleaning: 28

- 
### [all_shortest_path_with_clean.py](all_shortest_path_with_clean.py)
1. Visualize the **shortest path** from start to goal in ovon, mp3d, hm3d dataset
2. Dataset be cleaned by the filter


### [all_human_path_with_clean.py](all_human_path_with_clean.py)
1. Visualize the **human demonstrated trajectory** in mp3d dataset
2. Dataset be cleaned by the filter


### Sychronize with Physical robot
1. sensor resolution, position, hfov, sensor_subtype
2. agent height, radius, sensor_specifications, action_space
3. when build navmesh, need to adjust height, radius, max_climb, cell_height


