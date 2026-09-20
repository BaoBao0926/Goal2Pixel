coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # potted plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # dining-table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
}

# semantic pixel return
# (https://github.com/niessner/Matterport/blob/master/metadata/mpcat40.tsv)
mp3d_category = [
    # 'void',  # 0
    'wall',  # 1
    'floor',  # 2
    'chair',  # 3
    'door',  # 4
    'table',  # 5
    'picture',  # 6
    'cabinet',  # 7
    'cushion',  # 8
    'window',  # 9
    'sofa',  # 10
    'bed',  # 11
    'curtain',  # 12
    'chest_of_drawers',  # 13
    'plant',  # 14
    'sink',  # 15
    'stairs',  # 16
    'ceiling',  # 17
    'toilet',  # 18
    'stool',  # 19
    'towel',  # 20
    'mirror',  # 21
    'tv_monitor',  # 22
    'shower',  # 23
    'column',  # 24
    'bathtub',  # 25
    'counter',  # 26
    'fireplace',  # 27
    'lighting',  # 28
    'beam',  # 29
    'railing',  # 30
    'shelving',  # 31
    'blinds',  # 32
    'gym_equipment',  # 33
    'seating',  # 34
    'board_panel',  # 35
    'furniture',  # 36
    'appliances',  # 37
    'clothes',  # 38
    'objects',  # 39
    'misc',  # 40
    ]

# task_cat
category_to_task_category_id = {
    "chair": 0,
    "table": 1,
    "picture": 2,
    "cabinet": 3,
    "cushion": 4,
    "sofa": 5,
    "bed": 6,
    "chest_of_drawers": 7,
    "plant": 8,
    "sink": 9,
    "toilet": 10,
    "stool": 11,
    "towel": 12,
    "tv_monitor": 13,
    "shower": 14,
    "bathtub": 15,
    "counter": 16,
    "fireplace": 17,
    "gym_equipment": 18,
    "seating": 19,
    "clothes": 20,
    "foodstuff": 21,
    "stationery": 22,
    "fruit": 23,
    "plaything": 24,
    "hand_tool": 25,
    "game_equipment": 26,
    "kitchenware": 27
}

category_to_mp3d_category_id = {
    "chair": 3,
    "table": 5,
    "picture": 6,
    "cabinet": 7,
    "cushion": 8,
    "sofa": 10,
    "bed": 11,
    "chest_of_drawers": 13,
    "plant": 14,
    "sink": 15,
    "toilet": 18,
    "stool": 19,
    "towel": 20,
    "tv_monitor": 22,
    "shower": 23,
    "bathtub": 25,
    "counter": 26,
    "fireplace": 27,
    "gym_equipment": 33,
    "seating": 34,
    "clothes": 38,
    "foodstuff": 43,
    "stationery": 44,
    "fruit": 45,
    "plaything": 46,
    "hand_tool": 47,
    "game_equipment": 48,
    "kitchenware": 49
}

# mapping from mp3d category id to goal category id (1-21)
mapping_mpcat40_to_goal21 = {
    3: 1,  # ('chair', 2, task_cat: 0)
    5: 2,  # ('table', 4, task_cat: 1)
    6: 3,  # ('picture', 5, task_cat: 2)
    7: 4,  # ('cabinet', 6, task_cat: 3)
    8: 5,  # ('cushion', 7, task_cat: 4)
    10: 6,  # ('sofa', 9, task_cat: 5)
    11: 7,  # ('bed', 10, task_cat: 6)
    13: 8, # ('chest_of_drawers', 12, task_cat: 7)
    14: 9, # ('plant', 13, task_cat: 8)
    15: 10, # ('sink', 14, task_cat: 9)
    18: 11, # ('toilet', 17, task_cat: 10)
    19: 12, # ('stool', 18, task_cat: 11)
    20: 13, # ('towel', 19, task_cat: 12)
    22: 14, # ('tv_monitor', 21, task_cat: 13)
    23: 15, # ('shower', 22, task_cat: 14)
    25: 16, # ('bathtub', 24, task_cat: 15)
    26: 17, # ('counter', 25, task_cat: 16)
    27: 18, # ('fireplace', 26, task_cat: 17)
    33: 19, # ('gym_equipment', 32, task_cat: 18)
    34: 20,  # ('seating', 33, task_cat: 19)
    38: 21,  # ('clothes', 37, task_cat: 20)
}

C16_categories = [
"chair", "table", "picture", "cabinet", "cushion", "sofa", "fireplace",
"seating", "stool", "shower", "tv_monitor", "towel", "gym_equipment",
"sink", "clothes", "bathtub"
]

C5_categories = [
"counter", "chest_of_drawers", "bed", "toilet", "plant"
]

NO_USED_CATEGORIES = [
    "foodstuff", "stationery", "fruit", "plaything",
    "hand_tool", "game_equipment", "kitchenware"
]
# DETECTION_CATEGORIES = [
#     # 基础目标（ObjectNav常用）
#     "chair", "sofa", "table", "bed", "nightstand", "dresser", "wardrobe",
#     "bookshelf", "cabinet", "desk", "tv", "refrigerator", "oven",
#     "sink", "toilet", "bathtub", "shower", "mirror", "plant",
#     "washing machine",
#
#     # 数据集特有/要求保留（来自 C16/C5）
#     "picture", "cushion", "fireplace", "stool", "towel",
#     "gym_equipment", "clothes", "counter"
#     # 说明：C16 里的 “seating” 在规范映射里并入 "chair"（见下）
# ]

DETECTION_CATEGORIES = [
    # Seating
    "sofa", "chair", "dining_chair", "bar_chair", "stool", "combination_sofa",
    # Sleeping
    "bed", "multifunctional_combination_bed", "pillow",
    # Storage / Cabinets
    "wardrobe", "nightstand", "dresser", "tv_cabinet", "wine_cabinet",
    "bathroom_cabinet", "shoe_cabinet", "entrance_cabinet",
    "decorative_cabinet", "washing_cabinet", "wall_cabinet",
    "sideboard", "cupboard", "bookshelf", "bookcase", "cabinet", "clothes_rack",
    # Tables & Desks
    "table", "coffee_table", "dining_table", "side_table", "dressing_table",
    "desk", "counter", "bar", "dining_table_combination",
    "leisure_table_and_chair_combination",
    # Appliances (Kitchen)
    "refrigerator", "oven", "integrated_stove", "gas_stove", "stove",
    "range_hood", "micro-wave_oven", "dishwasher",
    # Appliances (Other)
    "washing_machine", "air_conditioner", "computer", "tv", "screen",
    # Bathroom
    "sink", "hand_sink", "toilet", "bathtub", "tub",
    "shower", "shower_room", "mirror",
    # Lighting
    "illumination", "lamp", "chandelier", "floor-standing_lamp",
    # Soft & Decorative
    "cushion", "carpet", "curtain", "painting", "wall_decoration", "picture",
    # Plants
    "plant", "plants", "potted_bonsai",
    # Misc / Utility
    "fireplace", "gym_equipment", "clothes"
]

# 同义词映射 (synonym mapping) -> canonical label（规范标签）
SYNONYMS_TO_CANONICAL = {
    # Seating
    "couch": "sofa", "loveseat": "sofa",
    "armchair": "chair", "office chair": "chair", "swivel chair": "chair",
    "lounge chair": "chair", "rocking chair": "chair", "bench": "chair",
    "sofa chair": "sofa", "pew": "chair", "row of theater seats": "chair",

    # Tables
    "end table": "side_table", "wall table": "table",
    "night table": "nightstand", "bedside table": "nightstand",

    # Cabinets
    "chest_of_drawers": "dresser", "drawer chest": "dresser",
    "bookcase": "bookshelf", "cabinet table": "cabinet",
    "kitchen cabinet": "cabinet", "vanity": "cabinet",

    # Electronics
    "tv monitor": "tv", "monitor": "tv", "wall tv": "tv",
    "projector screen": "screen",

    # Kitchen appliances
    "fridge": "refrigerator",
    "microwave oven": "micro-wave_oven", "microwave": "micro-wave_oven",
    "range": "stove",

    # Bathroom
    "washbasin": "sink", "basin": "sink",
    "jacuzzi": "bathtub", "hot tub": "bathtub",

    # Soft/Decor
    "photo": "picture", "photo frame": "picture",
    "pillow": "cushion",   # pillow 可归 cushion（小物品时）或保留 bed → pillow（大类）
    "potted plant": "plant", "flowerpot": "plant", "vase with plant": "plant",

    # Redundant names collapsed
    "plants": "plant", "potted_bonsai": "plant",

    # Ignore (不进入检测)
    "cup": None, "mug": None, "plate": None, "bowl": None,
    "book": None, "magazine": None, "toy": None, "clock": None
}