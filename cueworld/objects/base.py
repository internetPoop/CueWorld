import numpy as np
from typing import Dict, Any, Optional, Tuple

class BaseObject:
    def __init__(self, name: str):
        self.name = name
        self.id = None  # PyBulletのbody_id
        self.urdf_file = None  # サブクラスで定義するURDFファイル名
        
        # 触覚で何が感じられるか
        self.texture = None
        # 見た目(モデル)
        self.visual_shape = None
        # 色カテゴリ
        self.color = None
        # 形のカテゴリ
        self.shape_type = None
        # 動きのタイプ(?)．Noneの場合は，動かず，位置固定．また，初期化の際も位置リセットは最初以外は行わない．
        self.move_type = None
        #掴めるか．graspを判定する際に使用．
        self.is_graspable = False
        #食べ物か．eatの際の判定に使用する．
        self.is_food = False
        # 選好の重み（報酬計算のみ。観測には出ない）
        self.preference = 0.0
        #物の中に入れることができるか．物を手放すとき使用する．
        self.is_containable = False
        #危険か．
        self.dangerous = False
        #振る遊びができるか．オブジェクトが動いたときに音出す．
        #環境の最初で，この属性がTrueのものは音を鳴らす．
        self.can_shake = False
        #他のオブジェクトを中に入れられるか．開閉でパーツが分かれる場合，閉じているときは中に入れることができないとする(self.contentsのみになる)．
        self.can_contain = False
        #開けしめ可能か
        self.is_openable = False
        # 何に入っているか
        self.contained_in = None
        #サイズのカテゴリ
        self.size = None
        #バウンディングボックスの半長 (x, y, z)
        self.bbox_half_extents = [0.1, 0.1, 0.1]
        #連結(何かを置くことができる)対象のオブジェクトタイプ
        self.is_platform = False
        #板の上にある物リスト
        self.objects_on_board = []
        # 何に置かれているか
        self.put_on = None
        # 開閉
        self.open = True

        # 家事プロトコル用の属性
        self.target_category = None  # コンテナが収納対象とするアイテムの属性名 (例: "is_food", "is_toy")．旧コンテナ型 FSM 向け
        self.accepts_category = None  # board が受け付ける item の属性名 (例: "is_toy")．is_platform な board 向け
        self.is_toy = False          # おもちゃかどうか
        self.is_laundry = False      # 洗濯物かどうか

        # アフォーダンス: 振る → 音 / 効果チャネル
        # 振ったときに出る音（2次元固定ベクトル: 音色, pitch）。無音は [0.0, 0.0]。形カテゴリに紐付く。
        self.shake_sound = [0.0, 0.0]
        # 振ったときに出る音以外の効果チャネル（2次元固定ベクトル）。無効果は [0.0, 0.0]。色カテゴリに紐付く。
        self.shake_effect_vec = [0.0, 0.0]
        # 触覚ベクトル（2次元固定ベクトル）。非接触時の観測は [0.0, 0.0]。board 等は既定のまま。
        self.tactile_vec = [0.0, 0.0]
