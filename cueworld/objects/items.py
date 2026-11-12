import math

from .base import BaseObject


PRIMITIVE_BBOX_HALF_EXTENTS = {
    "cube": [0.06, 0.06, 0.06],
    "sphere": [0.06, 0.06, 0.06],
    "cylinder": [0.05, 0.05, 0.06],
    "cone": [0.06, 0.06, 0.065],
}

# 形状 → 振る音ベクトル（音色, pitch）。単位円上の4点でノルム1を保証し、無音[0,0]と明確に区別できる。
# オラクル観測（oracle_isolation.md）の形の識別にも流用する。
SHAPE_SHAKE_SOUND = {
    "cube":     [1.0,  0.0],
    "sphere":   [0.0,  1.0],
    "cylinder": [-1.0, 0.0],
    "cone":     [0.0, -1.0],
}

# 色 → 振る効果チャネルベクトル。単位円上の4点。board判定・語彙等の色同一性には影響しない。
# オラクル観測（oracle_isolation.md）の色の識別にも流用する。
COLOR_SHAKE_EFFECT = {
    "red":    [1.0,  0.0],
    "blue":   [0.0,  1.0],
    "green":  [-1.0, 0.0],
    "yellow": [0.0, -1.0],
}

# 形クラスタ中心（振る音ベクトルと同じ方位, 度）
SHAPE_TACTILE_CENTER_DEG = {
    "cube": 0.0,
    "sphere": 90.0,
    "cylinder": 180.0,
    "cone": -90.0,
}

# 色オフセット（度）。全形共通。
COLOR_TACTILE_OFFSET_DEG = {
    "red": -15.0,
    "blue": -5.0,
    "green": 5.0,
    "yellow": 15.0,
}


def tactile_vec_for_concept(color: str, shape_type: str):
    """concept_id ごとの単位円上触覚ベクトル [cos, sin]。"""
    angle_deg = SHAPE_TACTILE_CENTER_DEG[shape_type] + COLOR_TACTILE_OFFSET_DEG[color]
    angle_rad = math.radians(angle_deg)
    return [math.cos(angle_rad), math.sin(angle_rad)]


class PrimitiveObject(BaseObject):
    def __init__(self, color: str, shape_type: str, name: str):
        super().__init__(name)
        self.color = color
        self.shape_type = shape_type
        self.concept_id = f"{color}_{shape_type}"
        self.urdf_file = f"{color}_{shape_type}.urdf"
        self.bbox_half_extents = list(PRIMITIVE_BBOX_HALF_EXTENTS[shape_type])
        self.is_graspable = True
        self.is_containable = False  # contain アフォーダンスは削減済み（§2）
        self.move_type = 1
        self.texture = "smooth"
        self.size = max(self.bbox_half_extents) * 2.0
        # 振る → 音 / 効果チャネル（形・色にそれぞれ固定値を割当）
        self.can_shake = True
        self.shake_sound = list(SHAPE_SHAKE_SOUND[shape_type])
        self.shake_effect_vec = list(COLOR_SHAKE_EFFECT[color])
        self.tactile_vec = list(tactile_vec_for_concept(color, shape_type))
        self.preference = 1.0


class RedCube(PrimitiveObject):
    def __init__(self, name="RedCube"):
        super().__init__("red", "cube", name)


class RedSphere(PrimitiveObject):
    def __init__(self, name="RedSphere"):
        super().__init__("red", "sphere", name)


class RedCylinder(PrimitiveObject):
    def __init__(self, name="RedCylinder"):
        super().__init__("red", "cylinder", name)


class RedCone(PrimitiveObject):
    def __init__(self, name="RedCone"):
        super().__init__("red", "cone", name)


class BlueCube(PrimitiveObject):
    def __init__(self, name="BlueCube"):
        super().__init__("blue", "cube", name)


class BlueSphere(PrimitiveObject):
    def __init__(self, name="BlueSphere"):
        super().__init__("blue", "sphere", name)


class BlueCylinder(PrimitiveObject):
    def __init__(self, name="BlueCylinder"):
        super().__init__("blue", "cylinder", name)


class BlueCone(PrimitiveObject):
    def __init__(self, name="BlueCone"):
        super().__init__("blue", "cone", name)


class GreenCube(PrimitiveObject):
    def __init__(self, name="GreenCube"):
        super().__init__("green", "cube", name)


class GreenSphere(PrimitiveObject):
    def __init__(self, name="GreenSphere"):
        super().__init__("green", "sphere", name)


class GreenCylinder(PrimitiveObject):
    def __init__(self, name="GreenCylinder"):
        super().__init__("green", "cylinder", name)


class GreenCone(PrimitiveObject):
    def __init__(self, name="GreenCone"):
        super().__init__("green", "cone", name)


class YellowCube(PrimitiveObject):
    def __init__(self, name="YellowCube"):
        super().__init__("yellow", "cube", name)


class YellowSphere(PrimitiveObject):
    def __init__(self, name="YellowSphere"):
        super().__init__("yellow", "sphere", name)


class YellowCylinder(PrimitiveObject):
    def __init__(self, name="YellowCylinder"):
        super().__init__("yellow", "cylinder", name)


class YellowCone(PrimitiveObject):
    def __init__(self, name="YellowCone"):
        super().__init__("yellow", "cone", name)


class Board(BaseObject):
    def __init__(self, name="Board"):
        super().__init__(name)
        self.concept_id = "board"
        self.urdf_file = "board.urdf"
        self.move_type = None
        self.is_graspable = False
        self.is_platform = True
        self.accepts_category = None
        self.bbox_half_extents = [0.01, 0.4, 0.5]
        self.objects_on_board = [] # 板の上にある物リスト


class ColorBoard(Board):
    def __init__(self, color: str, name: str):
        super().__init__(name)
        self.color = color
        self.concept_id = f"{color}_board"
        self.urdf_file = f"board_{color}.urdf"
        self.accepts_category = color


class RedBoard(ColorBoard):
    def __init__(self, name="RedBoard"):
        super().__init__("red", name)


class BlueBoard(ColorBoard):
    def __init__(self, name="BlueBoard"):
        super().__init__("blue", name)


class GreenBoard(ColorBoard):
    def __init__(self, name="GreenBoard"):
        super().__init__("green", name)


class YellowBoard(ColorBoard):
    def __init__(self, name="YellowBoard"):
        super().__init__("yellow", name)
