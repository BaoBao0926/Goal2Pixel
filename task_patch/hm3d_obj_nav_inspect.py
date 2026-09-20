import gzip
import json
import os

"""
This script is to inspect the trajectory data of objectnav_hm3d_v2 dataset
    Inside of data/datasets/objectnav/hm3d/v2/:
        .
        ├── minival
        │ ├── content
        │ └── minival.json.gz
        ├── train
        │ ├── content
        │ └── train.json.gz
        └── val
            ├── content
            └── val.json.gz
    
"""

"""
Let's first inspect the minival.json.gz | train.json.gz | val.json.gz files
"""

objectnav_hm3d_v2_val = 'data/datasets/objectnav/hm3d/v2/val/val.json.gz'
objectnav_hm3d_v2_minival = 'data/datasets/objectnav/hm3d/v2/minival/minival.json.gz'
objectnav_hm3d_v2_train = 'data/datasets/objectnav/hm3d/v2/train/train.json.gz'

path_list = [objectnav_hm3d_v2_val, objectnav_hm3d_v2_minival, objectnav_hm3d_v2_train]
data_list = []
for path in path_list:
    # get absolute path
    # path = os.path.abspath(path)
    with gzip.open(path, 'rb') as f:
        json_bytes = f.read()
    json_str = json_bytes.decode('utf-8')
    data = json.loads(json_str)
    print(json.dumps(data, indent=4))
    data_list.append(data)
    """
    要寻找的东西以及对应的category_id

    data里面有三个key: 'episodes', 'category_to_task_category_id', 'category_to_scene_annotation_category_id'
        'episodes'是一个list，空集
        'category_to_task_category_id'是一个dict，key是category，value是task category id
        'category_to_scene_annotation_category_id'是一个dict，key是category，value是scene annotation category id

    总共6个object goal categories
        "chair": 0,
        "bed": 1,
        "plant": 2,
        "toilet": 3,
        "tv_monitor": 4,
        "sofa": 5
    """

print("*" * 50)

"""
Let's second inspect the minival content folder

    ├── content
    │ ├── k1cupFYWXJ6.json.gz
    │ ├── TEEsavR23oF.json.gz
    │ ├── wcojb4TFT35.json.gz
    │ └── y9hTuugGdiq.json.gz
    └── minival.json.gz

"""

# 然后我们来看看minival/content里面是什么
folder = 'data/datasets/objectnav/hm3d/v2/minival/content'
# 里面有很多json文件，每个json文件对应一个scene
import os
import json

# for each scene in this folder
for scene in os.listdir(folder):
    scene_path = os.path.join(folder, scene)
    print(scene_path)
    with gzip.open(scene_path, 'rb') as f:
        json_bytes = f.read()
    json_str = json_bytes.decode('utf-8')
    scene_data = json.loads(json_str)
    print(json.dumps(scene_data, indent=4))

    """
    scene_data里面有4个key: 'episodes', 'goals_by_category', 'category_to_task_category_id', 'category_to_scene_annotation_category_id'
        'goals_by_category'是一个dict，key是scene_name + 物品， value是一个list，里面是多个dict，表示这个场景里面的这个物品有
            多个，每一个对应的object id，位置等

        'episodes'是一个list，


        'category_to_task_category_id'是一个dict，key是category，value是task category id （和上面的一样）
        'category_to_scene_annotation_category_id'是一个dict，key是category，value是scene annotation category id （和上面的一样）

    总共6个object goal categories
        "chair": 0,
        "bed": 1,
        "plant": 2,
        "toilet": 3,
        "tv_monitor": 4,
        "sofa": 5
    """

    pass
pass