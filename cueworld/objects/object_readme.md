# オブジェクトの定義方法
各オブジェクトの見た目のモデルはassetsフォルダにあります．
各オブジェクトの設定はitems.pyに定義しています．
## items.pyについて
各オブジェクトの設定クラスがあります．base.pyのBaseObjectを継承しています．
現在の items.py では，オブジェクトは次のカテゴリに大別されます．
*   **可動プリミティブ (`PrimitiveObject`)**
    *   **色**: `red`, `blue`, `green`, `yellow`
    *   **形**: `cube`, `sphere`, `cone`, `cylinder`
    *   **種類**: 4 色 × 4 形 = 16 クラス（例: `RedCube`, `BlueSphere`, `GreenCone`, `YellowCylinder`）
    *   **特徴**: すべて `is_graspable=True`, `is_containable=False`, `can_shake=True`, `move_type=1` を持ち，識別は `concept_id`（例: `red_cube`）・`color`・`shape_type` で行います．
    *   **アフォーダンス（§2 削減後）**: 振る → 音 / 効果チャネルのみ。`shake_sound`（形由来の2次元固定ベクトル）と `shake_effect_vec`（色由来の2次元固定ベクトル）をクラスが保持する。`is_containable` は無効化済み。
        *   `SHAPE_SHAKE_SOUND`: cube=[1,0], sphere=[0,1], cylinder=[-1,0], cone=[0,-1]
        *   `COLOR_SHAKE_EFFECT`: red=[1,0], blue=[0,1], green=[-1,0], yellow=[0,-1]
*   **色ボード (`ColorBoard`)**
    *   **種類**: `RedBoard`, `BlueBoard`, `GreenBoard`, `YellowBoard`
    *   **特徴**: `is_platform=True` の固定オブジェクトで，`accepts_category` に一致する色のプリミティブを受け付けます．
*   **汎用ボード (`Board`)**
    *   **種類**: `Board`
    *   **特徴**: 色カテゴリを持たない汎用プラットフォームです．

旧来の `ToyCar`, `Banana`, `Box`, `Refrigerator` などの定義は `items_legacy.py` に退避しています．関連する URDF / OBJ 資産は assets フォルダに残していますが，現在の items.py からはロードしません．