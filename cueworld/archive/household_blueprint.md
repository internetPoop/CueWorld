# HouseholdSoloProtocol 設計ブループリント（仮）

> **ステータス:** Phase 0 用の仮ドラフト。実装前の作業メモであり、確定仕様ではない。
>
> 親ドキュメント: [plan.md](plan.md)

---

## 0. このドキュメントの位置づけ

`plan.md` Phase 0 の成果物として、以下を 1 か所にまとめた。

| Phase 0 チェック項目 | 本ドキュメントの該当セクション |
|---------------------|------------------------------|
| タスク種別テーブル | §1 |
| オブジェクト資産の棚卸し | §2 |
| 旧 FSM → 新 FSM 対応表 | §3 |
| 発話テンプレート方針 | §4 |
| board 実装方針（Phase 1.5 前提） | §5 |
| 未決事項の仮決め | §6 |

---

## 1. タスク種別定義（仮）

### 1.1 `TASK_SCHEMAS` 草案

Phase 2 で `household_solo.py` に導入する想定の dict。キーは `global_intent`（タスク種別）とする。

```python
TASK_SCHEMAS = {
    "tidy_toys": {
        "label_ja": "おもちゃを片付ける",
        "item_attr": "is_toy",           # item 側のブール属性名
        "board_attr": "accepts_category",  # board 側の属性名
        "board_value": "is_toy",         # board_attr の期待値
    },
    "laundry": {
        "label_ja": "洗濯物を集める",
        "item_attr": "is_laundry",
        "board_attr": "accepts_category",
        "board_value": "is_laundry",
    },
    "store_items": {
        "label_ja": "物をしまう",
        "item_attr": "is_food",          # 仮: is_storable ではなく既存属性を流用
        "board_attr": "accepts_category",
        "board_value": "is_food",
    },
}
```

### 1.2 タスク選択の責務分担（仮）

タスク候補の列挙は **プロトコルの `start()` ではなく `BaseEnv` 側でインデックス化** する。プロトコルはキャッシュを読むだけにし、`active_objects` の全走査を繰り返さない。

#### なぜ `BaseEnv` 側か

| 観点 | `start()` 毎の全走査 | `BaseEnv` でキャッシュ |
|------|---------------------|----------------------|
| 呼び出し頻度 | プロトコル切替のたびに走査 | board はロード後 1 回、item は reset 後 1 回 |
| 責務 | プロトコルが環境内部を知りすぎる | 環境が「何が置けるか」を提供、プロトコルは選択だけ |
| 他プロトコル | 同様の走査が重複しうる | 共通インデックスを再利用可能 |

#### 更新タイミング（重要）

`BaseEnv` は `__init__` 時点ではまだオブジェクトをロードしていない（初回 `reset()` → `_load_urdfs()` でロード）。よって:

| インデックス | 構築タイミング | 理由 |
|-------------|--------------|------|
| board 一覧 | **`_load_urdfs()` 完了後**（初回 reset 内） | board は `move_type=None` の固定物。エピソード間で body_id は不変 |
| item 一覧 | **`_select_active_objects()` 完了後**（毎 reset） | 可動 item は reset ごとに `active_objects` が変わる |

> ユーザー案の「`init` 時」は、実装上は **`_load_urdfs()` 後の初回構築** に相当する。`__init__` 直後ではなく、環境がオブジェクトを持った直後と理解する。

#### `BaseEnv` に追加する属性（仮）

```python
# board: accepts_category -> [body_id, ...]  （ロード後に構築、以降不変）
self.household_boards_by_category: Dict[str, List[int]] = {}

# item: item_attr -> [body_id, ...]  （毎 reset 後に再構築）
self.household_items_by_attr: Dict[str, List[int]] = {}

# 上記を TASK_SCHEMAS と突き合わせたタスク候補（毎 reset 後に再構築）
# 要素: {"task_type": str, "board_id": int, "item_ids": List[int]}
self.household_task_candidates: List[Dict[str, Any]] = []
```

#### `BaseEnv` に追加するメソッド（仮）

```python
def _index_household_boards(self) -> None:
    """_load_urdfs() 後に 1 回呼ぶ。is_platform かつ accepts_category を持つ board を索引化。"""
    self.household_boards_by_category = {}
    for body_id, obj in self.all_objects.items():
        category = getattr(obj, "accepts_category", None)
        if obj.is_platform and category is not None:
            self.household_boards_by_category.setdefault(category, []).append(body_id)

def _index_household_items(self) -> None:
    """_select_active_objects() 後に毎 reset 呼ぶ。active な item を属性別に索引化。"""
    self.household_items_by_attr = {}
    for body_id in self.active_objects:
        obj = self.all_objects[body_id]
        if obj.is_platform:
            continue
        for attr in ("is_toy", "is_laundry", "is_food"):  # TASK_SCHEMAS と同期
            if getattr(obj, attr, False):
                self.household_items_by_attr.setdefault(attr, []).append(body_id)

def _build_household_task_candidates(self) -> None:
    """board 索引と item 索引から (task_type, board, items) 候補を組み立てる。"""
    # TASK_SCHEMAS は household_solo 側に置くか、共有定数モジュールに切り出す（Phase 2 で決定）
    ...
```

**呼び出し箇所（仮）:**

```
_load_urdfs()
  └─ _index_household_boards()          # 追加

_select_active_objects()  （各 reset）
  └─ _index_household_items()           # 追加
  └─ _build_household_task_candidates() # 追加
```

#### プロトコル `start()` の流れ（修正版）

1. `self.env.household_task_candidates` を読む（**走査しない**）
2. 空なら `completed` で終了
3. `random.choice(candidates)` で 1 タスクを選ぶ
4. `global_intent`, `target_board_id`, `candidate_items` を保存し FSM を開始

Phase 5 以降、配置済み item の除外や `put_on is not None` のフィルタは、`_build_household_task_candidates()` 側で行う（`start()` ではなく reset 時に反映）。

**Phase 1 の暫定:** `BaseEnv` への索引追加は省略可。`household_solo.py` 内で `tidy_toys` 1 種をハードコード。索引化は **Phase 2 と同時** に `base.py` へ入れる。

### 1.3 初期タスクセット（仮決め）

| 項目 | 仮決め |
|------|--------|
| 初期リリースのタスク種別数 | **3 種すべて**（`tidy_toys`, `laundry`, `store_items`） |
| Phase 1 で動かす種別 | **`tidy_toys` のみ**（既存 `Board` を暫定 board として使用） |
| Phase 1.5 完了時点 | 3 種すべての (item, board) ペアが環境に存在 |

---

## 2. オブジェクト資産の棚卸し

### 2.1 属性の役割分担（仮）

| 属性 | 所在 | 用途 | 備考 |
|------|------|------|------|
| `target_category` | 旧コンテナ（`Box`, `LaundryBasket` 等） | コンテナが「何を入れるか」を示す属性名 | **新プロトコルでは不使用**（旧 FSM 専用） |
| `accepts_category` | board（新規） | board が「何を置いてよいか」を示す属性名 | Phase 1.5 で `BaseObject` に追加 |
| `is_toy` / `is_laundry` / `is_food` | item | タスクマッチング用ブール | 既存 |

> `target_category` と `accepts_category` は意味が対称だが、旧コードとの混同を避けるため board 側は別名にする（仮決め）。

### 2.2 item 一覧

| クラス | `is_toy` | `is_laundry` | `is_food` | `is_graspable` | `move_type` | タスク候補 | 状態 |
|--------|----------|--------------|-----------|----------------|-------------|-----------|------|
| `ToyCar` | ✓ | | | ✓ | 1 | `tidy_toys` | 利用可 |
| `Ball` | ✓ | | | ✓ | 2 | `tidy_toys` | 利用可 |
| `Rattle` | ✓ | | | ✓ | 1 | `tidy_toys` | 利用可 |
| `Lego` | ✓ | | | ✓ | 1 | `tidy_toys` | 利用可 |
| `TeddyBear` | ✓ | | | ✓ | 1 | `tidy_toys` | 利用可 |
| `Jenga` | ✓ | | | ✓ | 1 | `tidy_toys` | 利用可 |
| `Box` | | | | ✓ | 1 | — | 旧コンテナ。item にはしない |
| `Towel` | | ✓ | | ✓ | 1 | `laundry` | 利用可（唯一） |
| `Apple` | | | ✓ | ✓ | 1 | `store_items` | 利用可 |
| `Banana` | | | ✓ | ✓ | 1 | `store_items` | 利用可 |

**item 不足リスト（仮）:**

| 優先度 | 内容 | 対応 |
|--------|------|------|
| 低 | `laundry` の item が `Towel` のみ | Phase 1.5 では追加不要。必要なら後から `Sock` 等 |
| 低 | `store_items` の item が 2 種のみ | 同上。`Plate` 等は `is_food` 未設定のため対象外 |
| — | `is_food` 以外の「しまう」対象 | 当面は `store_items` = 食品の片付けと定義（仮） |

### 2.3 board 一覧

| クラス | `is_platform` | `accepts_category` | URDF | タスク候補 | 状態 |
|--------|---------------|-------------------|------|-----------|------|
| `Board` | ✓ | **なし** | `board.urdf` | — | Phase 1 暫定 board として流用可。属性追加が必要 |
| `Shelf` | ✗ | なし | `shelf.urdf` | — | `can_contain` のみ。**新設計では不採用**（仮） |
| `ToyBoard` | — | `"is_toy"` | `board_toy.urdf`（案） | `tidy_toys` | **未実装** |
| `LaundryBoard` | — | `"is_laundry"` | `board_laundry.urdf`（案） | `laundry` | **未実装** |
| `FoodBoard` | — | `"is_food"` | `board_food.urdf`（案） | `store_items` | **未実装** |

**旧コンテナ（新プロトコルでは不使用）:**

| クラス | `target_category` | 理由 |
|--------|-------------------|------|
| `Box` | `"is_toy"` | 開閉・投入フロー前提 |
| `LaundryBasket` | `"is_laundry"` | 同上 |
| `WashingMachine` | `"is_laundry"` | 開閉必須 |
| `Refrigerator` | `"is_food"` | 開閉必須 |

### 2.4 配置フロー（環境側の前提）

board への配置は `base.py` の `_update_grasp` が担う。

1. 親が item を把持した状態で board 近くに移動
2. 手を開く（`hand_shape != "closed"`）
3. AABB オーバーラップで近傍オブジェクトを検索
4. `is_platform=True` のオブジェクトが見つかれば `objects_on_board` / `put_on` を更新

**プロトコル側の完了判定（仮）:** `current_item_id in env.all_objects[target_board_id].objects_on_board`

**注意:** カテゴリ不一致の item も物理的には board に置けてしまう。正しいマッチングはプロトコルの候補選択で担保する（環境側バリデーションは Phase 1.5 では入れない）。

### 2.5 デモ配置案（仮）

Phase 1.5 以降の `demo_household_solo.py` 想定。3 board を 1 シーンに同時配置（仮決め）。

```
arena: grid, arena_half_extents=(1.6, 1.2), arena_rows=4, arena_cols=5

fixed_placements (row=0 / row=max):
  ToyBoard     x=-1.2, y=+yh-0.05   # 北壁
  LaundryBoard x= 0.0, y=+yh-0.05
  FoodBoard    x=+1.2, y=+yh-0.05

object_counts (可動, 中間行):
  ToyCar: 1, Towel: 1, Apple: 1, Ball: 1, Banana: 1
```

Phase 1 暫定（1 ペアのみ）:

```
fixed_placements:
  Board  x=0.0, y=+yh-0.05   # 既存 Board クラス

object_counts:
  ToyCar: 1, Ball: 1
```

---

## 3. FSM 設計: 旧 → 新 対応表

### 3.1 現行 FSM（`household_solo.py`）

| 状態 | proximity | 動作 | 遷移先 |
|------|-----------|------|--------|
| `init` | 0.0 | — | `moving_to_container_for_open` or `moving_to_item` |
| `moving_to_container_for_open` | 0.1 | コンテナへ接近 | `opening_container` |
| `opening_container` | 0.2 | 開ける | `moving_to_item` |
| `moving_to_item` | 0.4 | item へ接近 | `grasping_item` / `moving_to_container_for_close` |
| `grasping_item` | 0.5 | 把持 | `moving_to_container_for_put` |
| `moving_to_container_for_put` | 0.7 | コンテナへ接近 | `putting_in` |
| `putting_in` | 0.8 | 投入 | `moving_to_item` |
| `moving_to_container_for_close` | 0.9 | コンテナへ接近 | `closing_container` |
| `closing_container` | 0.95 | 閉める | `completed` |
| `completed` | 1.0 | — | 終了 |

**保持するインスタンス変数（現行）:**

- `target_container_id`
- `target_items`（list, FIFO）
- `current_item_id`

### 3.2 新 FSM（Phase 1 最小版）

| 状態 | proximity（仮） | 動作 | 遷移先 |
|------|----------------|------|--------|
| `init` | 0.0 | — | `selecting_task` |
| `selecting_task` | 0.05 | （`start()` で完了） | `moving_to_item` |
| `moving_to_item` | 0.3 | item へ接近 | `grasping_item` |
| `grasping_item` | 0.5 | 把持 | `moving_to_board` |
| `moving_to_board` | 0.7 | board へ接近（把持維持） | `placing_on_board` |
| `placing_on_board` | 0.85 | 手を開く（board 上に配置） | `check_done` |
| `check_done` | 0.9 | 完了判定 | `moving_to_item` or `completed` |
| `completed` | 1.0 | — | 終了 |

**新インスタンス変数（仮）:**

| 変数 | 型 | 説明 |
|------|-----|------|
| `global_intent` | `str` | `tidy_toys` / `laundry` / `store_items` |
| `target_board_id` | `int` | 配置先 board の body_id |
| `candidate_items` | `list[int]` | 未処理 item の body_id リスト |
| `current_item_id` | `int \| None` | 現在扱っている item |
| `items_placed` | `int` | 配置完了数（Phase 5 以降で使用） |
| `household_style` | `str` | Phase 6 以降で使用。Phase 1 では `"committed"` 固定 |

**削除する状態（Phase 1 で）:**

- `moving_to_container_for_open`
- `opening_container`
- `moving_to_container_for_put`
- `putting_in`
- `moving_to_container_for_close`
- `closing_container`

**削除するインスタンス変数:**

- `target_container_id` → `target_board_id` に置換

### 3.3 Phase 3 以降で追加する状態（参考・仮）

| 状態 | 導入 Phase | 概要 |
|------|-----------|------|
| `inspecting_item` | 4 | item を見るだけ。接近しない |
| `inspecting_board` | 4 | board を見るだけ |
| `hesitating` | 4 | 手を伸ばさず短時間停止 |
| `monitoring_infant` | 6 | 幼児方向を向く |
| `interrupted` | 6 | 注意が逸れた短い中断 |

### 3.4 旧 → 新 マッピング早見表

| 現行 | 新（Phase 1） | 備考 |
|------|--------------|------|
| `target_container_id` | `target_board_id` | コンテナ → board |
| `target_items` | `candidate_items` | 名前変更。選択ロジックは Phase 3 で変更 |
| `moving_to_item` | `moving_to_item` | 維持 |
| `grasping_item` | `grasping_item` | 維持 |
| `moving_to_container_for_put` | `moving_to_board` | board へ接近 |
| `putting_in` | `placing_on_board` | `put_in_object` → 手を開いて `is_platform` 判定 |
| `opening_container` / `closing_container` | — | 削除 |
| `queue_utterance` | — | 削除 |

---

## 4. 発話テンプレート方針

### 4.1 方針（仮決め）

| 項目 | 決定 |
|------|------|
| 家事中の親の発話 | **行わない** |
| `start()` での `queue_utterance` | **削除** |
| `_determine_action` の発話処理 | **削除**（基底クラスの `parent_utterance=False` 設定も検討） |
| `language/templates.py` の `household_solo` エントリ | **ファイルからは削除しない**。コメントで「現行未使用」と明記 |
| 将来の発話復活 | 意図ラベル用の非言語 cue（ジェスチャ等）を先に検討。音声は Phase 7 以降のオプション |

### 4.2 現行テンプレート（参照用・削除対象の呼び出し）

`language/templates.py` の `household_solo`:

| キー | 用途（現行） | 新プロトコル |
|------|-------------|-------------|
| `cleanup` | 片付け開始時 | 不使用 |
| `wash` | 洗濯開始時 | 不使用 |
| `soliloquy` | 独り言 | 不使用 |

### 4.3 コード変更箇所（Phase 1）

`household_solo.py`:

- `start()` L68–71: `queue_utterance("wash" / "cleanup")` を削除
- `_determine_action()` L200–207: 発話アクションブロックを削除
- （任意）`__init__` で `self.parent_utterance = False` を設定

---

## 5. board 実装方針（Phase 1.5 前提・仮決め）

### 5.1 採用方針

| 項目 | 仮決め | 理由 |
|------|--------|------|
| board の実装形態 | **タスク別サブクラス**（`ToyBoard`, `LaundryBoard`, `FoodBoard`） | `fixed_placements` の `class` キーで明示的に区別でき、URDF も分けやすい |
| 汎用 `Board` クラス | **残すが非推奨**。Phase 1 の暫定用途のみ | 後方互換 |
| `Shelf` の流用 | **不採用** | `is_platform=False`。専用 board に統一 |
| URDF | **色違いコピーで十分**（形状は共通 `board.urdf` ベース） | 実装コスト最小 |
| `store_items` の item 属性 | **`is_food` を流用** | 新属性 `is_storable` は当面導入しない |
| デモ配置 | **3 board を 1 シーンに同時配置** | タスク選択のランダム性を 1 エピソードで検証可能 |

### 5.2 サブクラス定義（仮）

```python
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
```

`Board` 基底に `accepts_category = None` を追加し、汎用 `Board` は Phase 1 暫定用として残す。

### 5.3 URDF 色分け（仮）

| ファイル | ベース | 視覚的区別 |
|---------|--------|-----------|
| `board_toy.urdf` | `board.urdf` | ストライプ色を暖色系（例: 赤・橙・黄） |
| `board_laundry.urdf` | `board.urdf` | 寒色系（例: 青・水色・白） |
| `board_food.urdf` | `board.urdf` | 緑系（例: 緑・黄緑・白） |

---

## 6. 未決事項と仮決め一覧

| # | 論点 | 仮決め | 確定タイミング |
|---|------|--------|--------------|
| 1 | board 実装方針 | タスク別サブクラス | Phase 1.5 開始時 |
| 2 | URDF の差別化 | 色違いコピー | Phase 1.5 |
| 3 | `Shelf` の扱い | 不採用 | Phase 1.5 |
| 4 | `store_items` の属性 | `is_food` 流用 | Phase 2 |
| 5 | デモ配置 | 3 board 同時配置 | Phase 1.5-D |
| 6 | 初期タスク種別数 | 3 種 | Phase 2 |
| 7 | `household_style` 分布 | Phase 1–5 は `"committed"` 固定。Phase 6 で均等確率 | Phase 6 |
| 8 | inspection 中の動作 | **首・胴の回転のみ**（`approach_object` 不使用） | Phase 4 |
| 9 | 未処理 item の再候補 | **次回 `start()` で再候補に含める**（`put_on is None` の item） | Phase 5 |
| 10 | `parent_utterance` フラグ | `False` に設定 | Phase 1 |
| 11 | Phase 1 暫定ペア | 既存 `Board` + `is_toy` item | Phase 1 |
| 12 | 完了判定（Phase 1） | 1 個配置で `completed`（全候補回収はしない） | Phase 1 |
| 13 | `persistence` | 現行どおり `5.0` 固定。style 連動は Phase 6 | Phase 1 |
| 14 | タスク候補の索引化 | **`BaseEnv` 側で管理**（board: ロード後、item/候補: 毎 reset） | Phase 2 |
| 15 | `TASK_SCHEMAS` の置き場所 | 当面 `household_solo.py`。索引構築時に参照するなら共有モジュールへ切り出し検討 | Phase 2 |

---

## 7. Phase 0 完了チェックリスト

- [x] タスク種別テーブル草案（§1）
- [x] オブジェクト棚卸しと不足リスト（§2）
- [x] 旧 FSM 状態一覧と削除対象（§3.1, §3.2）
- [x] 新 FSM 状態一覧（§3.2）
- [x] 旧 → 新 対応表（§3.4）
- [x] 発話テンプレート無効化方針（§4）
- [x] board 実装方針の仮決め（§5）
- [ ] 関係者レビュー後、`plan.md` Phase 0 のチェックボックスを更新
- [ ] 仮決めのうち変更があれば本ドキュメントを更新

---

## 8. 次のアクション（Phase 1 へ）

1. `household_solo.py` から発話・開閉・投入を削除
2. §3.2 の新 FSM を実装（`tidy_toys` 暫定ハードコード）
3. 既存 `Board` + `ToyCar`（または `Ball`）で 1 個配置 → `completed` を確認
4. `cueworld/demo/demo_household_solo.py` を board ベース配置に合わせて更新

**Phase 2 で追加（`base.py`）:**

5. `_index_household_boards()` / `_index_household_items()` / `_build_household_task_candidates()` を実装（§1.2 参照）
6. `household_solo.start()` を `env.household_task_candidates` 参照に切り替え
