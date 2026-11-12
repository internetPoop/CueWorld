from .base import BaseObject

# NOTE:
# §1 オブジェクト設計の簡素化により、旧オブジェクト定義をここへ退避した。
# 既存の URDF / OBJ 資産ファイルは cueworld/objects/assets/ に残している。

#TODO: 容器オブジェクトに何が入っているのかを属性で持っておく．

class Refrigerator(BaseObject):
    def __init__(self, name="Refrigerator"):
        super().__init__(name)
        self.concept_id = "refrigerator"
        self.urdf_file = "refrigerator.urdf"
        self.is_graspable = False
        self.can_contain = False
        self.is_openable = True
        #蓋のリンクのidは後で取得する．
        self.lid_link_id = None
        self.lid_joint = None
        self.open = False
        self.open_type = "rotate"
        self.contents = []
        self.target_category = "is_food"

class ToyCar(BaseObject):
    def __init__(self, name="ToyCar"):
        super().__init__(name)
        self.concept_id = "car"
        self.urdf_file = "toy_01.urdf"
        self.is_graspable = True
        self.is_containable = True
        self.move_type = 1
        self.is_toy = True


class ContainerClosed(BaseObject):
    def __init__(self, name="ContainerClosed"):
        super().__init__(name)
        self.urdf_file = "container_closed.urdf"
        self.bbox_half_extents = [0.2, 0.2, 0.15]
        self.is_graspable = False
        self.is_openable = True
        self.contents = [] # 中に入っている物リスト
        self.can_place = True
        self.placed_objects = []
        self.lid_link_id = None
        self.lid_joint = None
        self.open_type = "detach"
        #開いたら以下のリストのオブジェクトに分解される
        self.detachable_body_ids = []


class Table(BaseObject):
    def __init__(self, name="Table"):
        super().__init__(name)
        self.concept_id = "table"
        self.urdf_file = "table.urdf"
        self.is_graspable = False
        self.can_place = True
        self.placed_objects = []


class ContainerOpen(BaseObject):
    def __init__(self, name="ContainerOpen"):
        super().__init__(name)
        self.urdf_file = "container_open.urdf"
        self.bbox_half_extents = [0.2, 0.2, 0.15]
        self.is_graspable = False
        self.can_contain = True
        self.contents = [] # 中に入っている物リスト
        self.lid_link_id = None
        self.lid_joint = None
        self.connectable_object_type = "Lid"


class Apple(BaseObject):
    def __init__(self, name="Apple"):
        super().__init__(name)
        self.concept_id = "apple"
        self.urdf_file = "apple.urdf"
        self.bbox_half_extents = [0.05, 0.05, 0.05]
        self.is_food = True
        self.is_graspable = True
        self.is_containable = True
        self.move_type = 1


class Lamp(BaseObject):
    def __init__(self, name="Lamp"):
        super().__init__(name)
        self.concept_id = "lamp"
        self.urdf_file = "lamp_001.urdf"
        self.is_graspable = False



class Ball(BaseObject):
    def __init__(self, name="Ball"):
        super().__init__(name)
        self.concept_id = "ball"
        self.urdf_file = "ball.urdf"
        self.bbox_half_extents = [0.05, 0.05, 0.05]
        self.move_type = 2
        self.is_graspable = True
        self.is_containable = True
        self.is_toy = True


class Rattle(BaseObject):
    def __init__(self, name="Rattle"):
        super().__init__(name)
        self.concept_id = "rattle"
        self.urdf_file = "rattle.urdf"
        self.move_type = 1
        self.is_graspable = True
        self.can_shake = True
        self.is_containable = True
        self.is_toy = True


class Banana(BaseObject):
    def __init__(self, name="Banana"):
        super().__init__(name)
        self.concept_id = "banana"
        self.urdf_file = "banana.urdf"
        self.is_food = True
        self.is_graspable = True
        self.is_containable = True
        self.move_type = 1


class FryingPan(BaseObject):
    def __init__(self, name="FryingPan"):
        super().__init__(name)
        self.concept_id = "pan"
        self.urdf_file = "pan_tefal.urdf"
        self.is_graspable = True
        self.is_containable = True
        self.move_type = 1
        self.can_place = True
        self.placed_objects = []

class Lego(BaseObject):
    def __init__(self, name="Lego"):
        super().__init__(name)
        self.concept_id = "lego"
        self.urdf_file = "lego.urdf"
        self.is_graspable = True
        self.is_containable = True
        self.move_type = 1
        self.is_toy = True
        


class Door(BaseObject):
    def __init__(self, name="Door"):
        super().__init__(name)
        self.concept_id = "door"
        self.urdf_file = "door.urdf"
        self.is_graspable = False
        self.is_openable = True
        self.open = False
        self.open_type = "rotate"
        self.lid_link_id = None
        self.lid_joint = None
        


class TeddyBear(BaseObject):
    def __init__(self, name="TeddyBear"):
        super().__init__(name)
        self.concept_id = "bear"
        self.urdf_file = "teddy_vhacd.urdf"
        self.is_graspable = True
        self.is_containable = True
        self.move_type = 1
        self.is_toy = True


class Box(BaseObject):
    def __init__(self, name="Box"):
        super().__init__(name)
        self.concept_id = "box"
        self.urdf_file = "box.urdf"
        self.shape_type = "box_container"
        self.size = 0.2
        self.bbox_half_extents = [0.15, 0.15, 0.15]
        self.is_graspable = True
        self.texture = "wood"
        self.move_type = 1
        
        # 容器特有のアフォーダンス属性
        self.can_contain = True
        self.contents = [] # 中に入っている物リスト
        self.target_category = "is_toy"



class Plate(BaseObject):
    def __init__(self, name="Plate"):
        super().__init__(name)
        self.concept_id = "plate"
        self.urdf_file = "plate.urdf"
        self.is_graspable = True
        self.can_contain = True
        self.is_containable = True
        self.contents = []
        self.move_type = 1


class LaundryBasket(BaseObject):
    def __init__(self, name="LaundryBasket"):
        super().__init__(name)
        self.concept_id = "basket"
        self.urdf_file = "laundry_basket.urdf"
        self.bbox_half_extents = [0.25, 0.25, 0.3]
        self.is_graspable = True
        self.can_contain = True
        self.contents = [] # 中に入っている物リスト
        self.target_category = "is_laundry"


class RedCube(BaseObject):
    def __init__(self, name="RedCube"):
        super().__init__(name)
        self.concept_id = "red"
        self.urdf_file = "red_cube.urdf"
        self.shape_type = "box"
        self.size = 0.15
        self.is_graspable = True
        self.is_containable = True
        self.texture = "smooth"


class BlueCube(BaseObject):
    def __init__(self, name="BlueCube"):
        super().__init__(name)
        self.concept_id = "blue"
        self.urdf_file = "blue_cube.urdf"
        self.shape_type = "box"
        self.size = 0.15
        self.is_graspable = True
        self.is_containable = True
        self.texture = "smooth"


class GreenCube(BaseObject):
    def __init__(self, name="GreenCube"):
        super().__init__(name)
        self.concept_id = "green"
        self.urdf_file = "green_cube.urdf"
        self.shape_type = "box"
        self.size = 0.15
        self.is_graspable = True
        self.is_containable = True
        self.texture = "smooth"


class Lid(BaseObject):
    def __init__(self, name="Lid"):
        super().__init__(name)
        self.concept_id = "lid"
        self.urdf_file = "lid.urdf"
        self.is_graspable = True
        self.connectable_object_type = "ContainerOpen"


class WashingMachine(BaseObject):
    def __init__(self, name="WashingMachine"):
        super().__init__(name)
        self.concept_id = "washing_machine"
        self.urdf_file = "washing_machine.urdf"
        self.is_graspable = False
        self.can_contain = False
        self.is_openable = True
        self.open = False
        self.lid_link_id = None
        self.lid_joint = None
        self.open_type = "rotate"
        self.contents = [] # 中に入っている物リスト
        self.target_category = "is_laundry"


class Jenga(BaseObject):
    def __init__(self, name="Jenga"):
        super().__init__(name)
        self.concept_id = "jenga"
        self.urdf_file = "jenga.urdf"
        self.is_graspable = True
        self.is_containable = True
        self.move_type = 1
        self.is_toy = True


class HingedBox(BaseObject):
    def __init__(self, name="HingedBox"):
        super().__init__(name)
        self.concept_id = "box"
        self.urdf_file = "hinged_box.urdf"
        self.is_graspable = False
        self.can_contain = False
        self.is_openable = True
        self.open = False
        self.lid_link_id = None
        self.lid_joint = None
        self.open_type = "rotate"
        self.contents = [] # 中に入っている物リスト


class CupSmall(BaseObject):
    def __init__(self, name="CupSmall"):
        super().__init__(name)
        self.concept_id = "cup"
        self.urdf_file = "cup_small.urdf"
        self.is_graspable = True
        self.can_contain = True
        self.is_containable = True
        self.move_type = 1
        self.contents = [] # 中に入っている物リスト


class Towel(BaseObject):
    def __init__(self, name="Towel"):
        super().__init__(name)
        self.concept_id = "towel"
        self.urdf_file = "towel.urdf"
        self.is_graspable = True
        self.is_containable = True
        self.move_type = 1
        self.is_laundry = True


class Shelf(BaseObject):
    def __init__(self, name="Shelf"):
        super().__init__(name)
        self.concept_id = "shelf"
        self.urdf_file = "shelf.urdf"
        self.is_graspable = False
        self.can_contain = True
        self.contents = [] # 中に入っている物リスト


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


class ToyBoard(Board):
    def __init__(self, name="ToyBoard"):
        super().__init__(name)
        self.concept_id = "toy_board"
        self.urdf_file = "board_toy.urdf"
        self.accepts_category = "is_toy"


class LaundryBoard(Board):
    def __init__(self, name="LaundryBoard"):
        super().__init__(name)
        self.concept_id = "laundry_board"
        self.urdf_file = "board_laundry.urdf"
        self.accepts_category = "is_laundry"


class FoodBoard(Board):
    def __init__(self, name="FoodBoard"):
        super().__init__(name)
        self.concept_id = "food_board"
        self.urdf_file = "board_food.urdf"
        self.accepts_category = "is_food"
