"""HouseholdSoloProtocol 用のタスク種別定義。"""

TASK_SCHEMAS = {
    "collect_red": {
        "label_ja": "赤い物を集める",
        "category": "red",
    },
    "collect_blue": {
        "label_ja": "青い物を集める",
        "category": "blue",
    },
    "collect_green": {
        "label_ja": "緑の物を集める",
        "category": "green",
    },
    "collect_yellow": {
        "label_ja": "黄色い物を集める",
        "category": "yellow",
    },
}

# item 索引化時に参照する色カテゴリ
HOUSEHOLD_ITEM_CATEGORIES = tuple(schema["category"] for schema in TASK_SCHEMAS.values())

# Phase 3: household_style ごとのサブゴール選択重み（Phase 6 まで style は "committed" 固定でも利用）
HOUSEHOLD_STYLE_PARAMS = {
    "committed": {
        "distance_weight": 2.0,
        "typicality_weight": 0.5,
        "temperature": 0.8,
    },
    "opportunistic": {
        "distance_weight": 3.0,
        "typicality_weight": 0.2,
        "temperature": 0.5,
    },
    "monitoring": {
        "distance_weight": 1.0,
        "typicality_weight": 0.8,
        "temperature": 1.0,
    },
    "interrupted_but_returning": {
        "distance_weight": 1.0,
        "typicality_weight": 1.0,
        "temperature": 1.0,
    },
}

# カテゴリ典型性バイアス（クラス名 → スコア）。未登録は default_typicality を使用
ITEM_TYPICALITY_BY_CLASS = {
    "RedCube": 1.0,
    "BlueCube": 1.0,
    "GreenCube": 1.0,
    "YellowCube": 1.0,
    "RedSphere": 0.85,
    "BlueSphere": 0.85,
    "GreenSphere": 0.85,
    "YellowSphere": 0.85,
    "RedCylinder": 0.8,
    "BlueCylinder": 0.8,
    "GreenCylinder": 0.8,
    "YellowCylinder": 0.8,
    "RedCone": 0.75,
    "BlueCone": 0.75,
    "GreenCone": 0.75,
    "YellowCone": 0.75,
}
DEFAULT_ITEM_TYPICALITY = 0.7

# Phase 4: inspection / hesitation の確率と滞在ステップ数
# 既定はオフ（0 が正）。コードパスは残し、context またはデモ opt-in でオンにする。
HOUSEHOLD_INSPECTION_PARAMS = {
    "inspect_item_prob": 0.0,
    "inspect_board_prob": 0.0,
    "hesitate_prob": 0.0,
    "inspect_item_duration": 40,
    "inspect_board_duration": 35,
    "hesitate_duration": 25,
}

# デモ --inspection 用。item / board は必ず出し、hesitate は出さない。
HOUSEHOLD_INSPECTION_DEMO_ON = {
    "inspect_item_prob": 1.0,
    "hesitate_prob": 0.0,
    "inspect_board_prob": 1.0,
}
