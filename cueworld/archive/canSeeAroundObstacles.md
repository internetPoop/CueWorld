# 親の「遮蔽物を避けて視線を確保する」実装案（グリッド利用型）

環境(`base.py`)に存在するアリーナグリッド（`self.cells`, `self.arena_rows`, `self.arena_cols`）の概念を利用し、対象が直接見えない（障害物がある）場合は周辺のグリッドセルを探索して、見える位置へ移動する仕組みを実装します。

## 実装計画

### 1. グリッド探索＆判定用メソッドの追加 (`parent_tools.py`)
`ParentTools` クラスに、以下の処理を行うヘルパーメソッドを追加します。

**`get_visible_target_position(self, agent, target_pos, target_body_id)`**
1. **直接の視線チェック**:
   まずは現在の顔（または胴体）の位置から `target_pos` へ向けて `p.rayTest` を行います。
   障害物がなく遮られていなければ、そのまま現在の位置を維持（または何もせず現在のターゲット方向を見る）で完了とします。
2. **自分とターゲットの所属グリッド計算**:
   現在の `agent` の座標と `target_pos` を元に、それぞれが属するグリッドの座標(行`r`, 列`c`)またはインデックスを計算します。（`base.py` の `arena_half_extents`, `cell_w`, `cell_h` などを参照して逆算）
3. **周辺グリッドの探索 (層状の探索)**:
   自分のいるグリッドから探索半径を1マス、2マス...と徐々に広げながら、候補となるグリッドのセル中心座標を取得します。
   - **除外条件**: そのグリッドがターゲットのいるグリッドと同じ場合は除外します。
   - **視線チェック**: 候補グリッドのセルの中心（高さは親の目の高さ程度を想定）から `target_pos` に向けて `p.rayTest` を放ちます。
   - もし視線が通る（ターゲットあるいは何もヒットしない）セルが見つかれば、そのセルの中心座標を目標位置（ウェイポイント）として返します。
4. **探索打ち切り**:
   すべてのセルを探しても見つからない場合は現在地を返します。

### 2. 回避行動の生成メソッドの追加 (`parent_tools.py`)
上記の座標に向けて移動するアクションを作るメソッドを実装します。

**`move_to_visible_cell(self, agent, target_pos, target_body_id, action=None)`**
1. `get_visible_target_position` を呼び出し、目標座標（迂回先のグリッド中心座標、またはNone）を取得します。
2. 迂回先セル座標が見つかった場合、そちらへ向くためのヨー回転を計算し、`forward` に値をセットしてそこへ移動するアクションを返します。
3. もし見つからなかった場合、または直接視線が通る場合は、従来通りのアクション（単純にターゲットを見るだけ、など）を行います。

### 3. 具体的な追加部品 (`parent_tools.py`)
- **`check_ray_visibility(start_pos, target_pos, target_id, ignore_ids)`**: `p.rayTest` のラップ関数。
- **`get_grid_rc_from_pos(pos)`**: 座標から (row, col) を返す関数。
- **`get_cell_center(r, c)`**: (row, col) からセルの中心XYを返す関数。
- **`find_visible_cell_for_target(agent, target_pos, target_body_id)`**: グリッド探索の本体。
- **`approach_for_visibility(agent, target_bdoy_id, ...)`**: これらを統合して見えなければ移動、見えれば注視などのアクションを返すメソッド。

---
以上の内容で `parent_tools.py` への実装を開始しますが、よろしいでしょうか？

## 4. objectとinfantの違いに対応する修正案

現状の `approach_for_visibility` は `target_body_id` （整数）を受け取り、オブジェクトであることを前提（`_obj_pos` や `approach_object`, `gaze_object` を使用）とした実装になっています。しかし、オブジェクトと幼児では注目すべき位置（オブジェクトはbody、幼児は `eyes_link`）が異なるため、これに対応する必要があります。

以下の3つの対応案が考えられます。

### 案1: メソッドをオブジェクト用と幼児用に完全に分離する（推奨）
既存の `approach_object` / `approach_infant` や `gaze_object` / `gaze_agent` の設計思想に合わせて、回避付きアプローチも2つのメソッドに分けます。

- **`approach_object_for_visibility(self, agent, oid, action=None, ...)`**
  現在追加したものの名前を変更。オブジェクト（`oid`）を受け取り、位置は `self.env._obj_pos(oid)` で取得。見えたら `approach_object` や `gaze_object` を呼ぶ。
- **`approach_infant_for_visibility(self, agent, infant, action=None, ...)`**
  幼児をターゲットにするメソッド。エージェント辞書（`infant`）を受け取り、位置は `p.getLinkState(infant["body"], infant["eyes_link"])` から取得。見えたら `approach_infant` や `gaze_agent` を呼ぶ。
- **メリット**: 既存コードの設計（引数の取り方など）との一貫性があり、コードが最もシンプルでバグが起きにくいです。

### 案2: 1つのメソッド内で型の違い（やフラグ）によって動的に処理を分岐する
メソッドの引数を `approach_for_visibility(self, agent, target, action=None, ...)` とし、引数 `target` の型で処理を分ける。

- `isinstance(target, dict)` なら幼児と判断し、リンク位置を計算して `approach_infant` 系を呼び出す。
- `isinstance(target, int)` ならオブジェクトIDと判断して `approach_object` 系を呼び出す。
- **メリット**: 利用側が1つのメソッドを呼ぶだけで済みます。
- **デメリット**: 内部の分岐が多くなり、引数 `target` の意味が文脈で変わるため、ドキュメントや型ヒントが煩雑になります。

### 案3: 迂回移動処理を分離し、最終アプローチ処理は引数で渡す（高階関数）
迂回先のセルに移動する部分だけを共通のヘルパー関数として抽出し、見えている場合の「接近・注視関数（オブジェクト用 or 幼児用）」を引数として渡す。

- **メリット**: 回避用のコード（セルの中心へ向かう処理など）の重複をなくせます。
- **デメリット**: 呼び出し時の引数の渡し方が複雑になり、関数を渡す構成に少し癖が出ます。

---
既存ソースの `gaze` や `approach` に関するメソッドの分かれ方からすると、**案1（メソッドの分離）** が素直で保守しやすいと考えられます。案のご希望や、他に検討したい方針があれば教えてください。
