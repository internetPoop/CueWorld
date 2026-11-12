# HouseholdSoloProtocol 再設計プラン

## 背景と目的

`cueworld` は幼児の意図理解・共同注意・模倣などの社会的スキル獲得をシミュレーションで検証する環境である（[env.md](docs/env.md)）。`protocol_manager` も確率的選択や部分的なシグナル解釈を前提にしている。

現行の `household_solo.py` は、コンテナの開閉 → 対象物を順番に回収 → 閉める、という**完全計画型の FSM** になっている。幼児から見ると「次にどの物を取るか」まで読み切れてしまい、意図理解の coarse-to-fine な発達を観察しにくい。

**狙うべき振る舞い:**

> 親の行動には局所的一貫性があるが、幼児からは途中までしか逆算できない

つまり親は **intention-consistent（意図は一貫）** だが **action-variable（行動は揺れる）** に振る舞う。幼児が学ぶのは single next action の予測ではなく、**latent intention schema** の推定である。

---

## 現状の問題点

| 観点 | 現行 `household_solo.py` |
|------|--------------------------|
| タスク構造 | コンテナ開閉 + 全アイテム回収の直線 FSM |
| 対象選択 | `target_items[0]` を固定順で処理 |
| 完遂条件 | 対象物をすべて片付けるまで継続 |
| 発話 | 開始時に `wash` / `cleanup` を発話 |
| 幼児シグナル | `infant_signals` を参照しない |
| 物理操作 | `open_close_object`, `put_in_object` など複合アフォーダンス |

---

## 設計方針

### 1. 操作の単純化（board ベース）

家事は「オブジェクトを board（固定可能な場所）に置く」という抽象操作に統一する。

- 開閉・投入などの複合アフォーダンスは**いったん使わない**
- 環境側の `is_platform` / `objects_on_board` / `put_on` 機構を利用する
- 家事中の親の発話は**行わない**

### 2. タスク種別の抽象化

ターゲットのオブジェクトと board は、以下のような**タスク種別（規則性）**の中から選び、その規則に従ってマッチングする。

| タスク種別 | 意図（global_intent） | item 側の属性 | board 側の属性（目標） |
|-----------|----------------------|--------------|----------------------|
| `tidy_toys` | おもちゃを片付ける | `is_toy` | `accepts_category = "is_toy"` |
| `laundry` | 洗濯物を集める | `is_laundry` | `accepts_category = "is_laundry"` |
| `store_items` | 物をしまう | `is_food` 等 | `accepts_category = "is_food"` |

> 具体的なオブジェクトクラス名ではなく、`accepts_category`（board）と item のブール属性でマッチングする。
>
> **前提:** この抽象化に入る前に、各タスク種別に対応する **item と board のオブジェクト実装**が揃っている必要がある（→ Phase 1.5）。

#### タスク選択方針（状況依存・ランダム）

`protocol_manager` が家事プロトコル（`household_solo`）を選んだあと、**どの家事タスク種別（`global_intent`）を実行するか**はランダムだが、**そのときの環境の状況だけ**を見て決める。

| 原則 | 内容 |
|------|------|
| 候補は「今できること」だけ | `TASK_SCHEMAS` 全種から無条件に抽選しない |
| 状況の情報源 | 主に **現在 `active` な可動オブジェクト**（固定 board を除く）と、シーン内の board |
| 成立条件 | 各タスク種別について、(対応 board が存在) **かつ** (未配置の対応 item が 1 個以上) |
| 不可能なタスク | **スキップ**（候補リストに入れない） |
| 抽選 | 成立した候補のうちから `random.choice` |
| 候補ゼロ | プロトコルは即 `completed`（実行可能な家事がない） |

**「現在の状況」の具体例（`start()` / 毎 `reset` 時点）:**

- **可動 item:** `active_objects` のうち `move_type != None` のもの。すでに `put_on` されている・board 上にあるものは対象外
- **board:** `fixed_placements` 等で配置された `is_platform` オブジェクト。`accepts_category` でタスク種別と対応
- **マッチング:** `TASK_SCHEMAS` の `item_attr` / `board_value` で (board, item群) のペアを組み立て、ペアが 1 件以上ある種別だけを候補にする

**実装上の分担（Phase 2）:**

```
reset 時（BaseEnv）
  → _index_household_boards()      # シーン内 board（固定物）
  → _index_household_items()       # 現在 active な可動 item
  → _build_household_task_candidates()  # 成立する (task_type, board, items) だけ列挙

start 時（HouseholdSoloProtocol）
  → household_task_candidates が空ならスキップ（completed）
  → 非空ならその中からランダムに 1 タスクを選択
```

> エピソード途中で item が減ったり配置済みになった場合の再評価は、原則 **次回 `start()` 時**（Phase 5 以降でエピソード内ループ時にも更新する余地あり）。

#### 現状のオブジェクト資産と不足（2025-07 時点の棚卸し）

| 役割 | タスク種別 | 既存クラス | 状態 |
|------|-----------|-----------|------|
| item | `tidy_toys` | `ToyCar`, `Ball`, `Rattle`, `Lego`, `TeddyBear`, `Jenga` 等（`is_toy=True`） | 十分 |
| item | `laundry` | `Towel`（`is_laundry=True`）のみ | 最小限（追加は任意） |
| item | `store_items` | `Apple`, `Banana`（`is_food=True`） | 最小限 |
| board | `tidy_toys` | `Board`（`is_platform=True`） | **属性なし** — どのタスク用か区別できない |
| board | `laundry` | なし（`LaundryBasket` / `WashingMachine` は旧コンテナ型） | **未実装** |
| board | `store_items` | `Shelf`（`can_contain` のみ、`is_platform=False`） | **未対応** — `place_on_board` 機構が使えない |

`Board` クラスと `board.urdf` は存在するが、タスク種別との対応属性・複数種の board・デモ用配置が未整備。

### 3. 3 層の意図構造

```
global_intent   … 数十ステップ維持される大きな意図（片付け / 洗濯 / しまう）
local_subgoal   … 文脈で切り替わる短い意図（候補を選ぶ / 近づく / 見る / 戻る）
micro_action    … 実際の接近・把持・board への配置
```

### 4. 行動の非決定論性（4 つの技法）

#### (a) foraging 的な探索

- 「今見えている候補から 1 つ選ぶ」
- 近いものを優先するが、たまにカテゴリ典型性の高いものを選ぶ
- 幼児視点では「拾う対象が毎回少し揺れる」

#### (b) inspection / hesitation 状態

親が対象物や board を見る、周囲を見回す、手を伸ばしかけてやめる、別の物に向き直る、といった短い中間状態。ノイズではなく「選別している」振る舞いとして意図を示すが、可視化は不完全にする。

#### (c) 完遂条件の緩和

「全部終える」ではなく「十分やった」で終了する。

- n 個片付けたら一旦終了
- 時間が来たら終了
- 2–3 個やったら別プロトコルへ戻る

#### (d) household_style による実行モード

| モード | 振る舞い |
|--------|----------|
| `committed` | 比較的一直線に進める |
| `opportunistic` | 近い物から拾う |
| `monitoring` | 幼児をたまに見ながら進める |
| `interrupted_but_returning` | 一瞬注意が逸れるが家事に戻る |

`persistence` で意図の持続を表現しつつ、局所行動の予測不能性を入れる。

### 5. 幼児シグナルとの相互作用

親はソロだが、幼児が近づいたり見たりすると:

- 一瞬だけ見返す
- 少し待つ
- 別の物を選ぶ

ことがある（`monitoring` / `interrupted_but_returning` モードで顕在化）。

### 6. 研究的に目指す性質

| 性質 | 内容 |
|------|------|
| 近接的予測は難しい | 次にどの物を取るかは読みにくい |
| 中域の意図は読める | 「片付け」「洗濯」「しまう」はわかる |
| 結果から逆算できる | 数ステップ後に見ると「ああ、あれをしていたのか」となる |
| 発達段階に応じた理解単位 | 動き → サブゴール → 全体意図 |

**推奨の組み合わせ:** 大目標は固定 + サブゴール選択は確率的 + 途中に inspection/hesitation + 完遂条件は緩める

---

## 目標アーキテクチャ（FSM 概要）

```
[start]
  → select_task_type()          # 状況に応じた候補から global_intent をランダム決定（不可能ならスキップ）
  → select_household_style()    # 実行モードを決定
  → [loop]
      → (optional) inspect / hesitate
      → select_subgoal()        # 確率的に候補を選ぶ
      → approach_item
      → grasp_item
      → approach_board
      → place_on_board
      → check_completion()      # 十分やった？ or 続ける？
  → [completed]
```

`state_proximities` は global_intent の進捗（何個片付けたか / 経過時間）と local_subgoal の段階を組み合わせて定義する。

---

## 実装ロードマップ

### Phase 0: 設計確定と下準備

**目的:** 実装前にインターフェースとデータ構造を固める。

**成果物ドキュメント:** [household_blueprint.md](household_blueprint.md)（仮ドラフト）

- [x] タスク種別テーブル（`task_type` → item 属性 → board の `accepts_category`）を定義 → blueprint §1
- [x] オブジェクト資産の棚卸し → blueprint §2
- [x] `household_solo.py` の旧 FSM 状態一覧を文書化し、削除対象（開閉系）を明示 → blueprint §3
- [x] `language/templates.py` の `household_solo` 発話テンプレートを無効化方針で確認 → blueprint §4
- [x] Phase 1.5 の board 実装方針（サブクラス vs 汎用 Board）を決定 → blueprint §5
- [x] 関係者レビュー・仮決めの確定

**成果物:** タスク種別定義（コード内定数 or 設定 dict）、旧→新の状態対応表、オブジェクト不足リスト（→ `household_blueprint.md` に集約）

---

### Phase 1: 操作の単純化（board ベース FSM）

**目的:** 最小限の新 FSM で end-to-end に動かす。非決定論性はまだ入れない。

- [x] 発話（`queue_utterance`, `_determine_action` 内の発話処理）を削除
- [x] コンテナ開閉・投入ロジックを削除
- [x] `approach_object` → `grasp_object` → `approach_object(board)` → 手を開いて `place_on_board` の直線 FSM を実装
- [x] `start()` で **暫定的に 1 種類の (item, board) ペア**をハードコードで選ぶ（タスク種別の抽象化は Phase 1.5 / 2 へ）
- [x] `state_proximities` を新状態に合わせて更新
- [x] デモスクリプト or 簡易テストで 1 タスクが完走することを確認

**依存:** 環境の `is_platform` / `objects_on_board` が期待通り動くこと（`base.py` の `_update_grasp`）

**完了基準:** 1 種類の (item, board) ペアで、1 個のオブジェクトを board に置いて `completed` になる（タスク種別の抽象化は Phase 1.5 / 2 で行う）

---

### Phase 1.5: 必要オブジェクトの実装

**目的:** Phase 2 のタスク種別抽象化に必要な item / board を環境に揃える。プロトコル側の `TASK_SCHEMAS` より先に、**物理オブジェクトとマッチング属性**を実装する。

**背景:** 現行の `items.py` には汎用 `Board` はあるが、タスク種別ごとの board・マッチング属性・デモ配置が未整備。`Shelf` は `is_platform=False` のため board ベースの配置フローに使えない。Phase 1 では既存の 1 ペアで動作確認し、本フェーズで 3 タスク種別すべてが成立するオブジェクト群を用意する。

#### 1.5-A: 共通インターフェースの定義

- [ ] `BaseObject`（`objects/base.py`）に board 用属性を追加
  - `accepts_category: Optional[str]` — この board が受け付ける item の属性名（例: `"is_toy"`）
  - 既存の `target_category`（コンテナ向け）との役割分担をコメントで明示
- [ ] board 判定ヘルパーの方針を決める（例: `is_platform and accepts_category is not None`）

#### 1.5-B: board クラスの実装

方針: 既存 `Board` を拡張するか、タスク別サブクラスを作る。視覚的に区別できるよう URDF の色・ラベルを変える。

| クラス名（案） | `accepts_category` | URDF | 備考 |
|--------------|-------------------|------|------|
| `ToyBoard` | `"is_toy"` | `board_toy.urdf`（または `board.urdf` + 色変更） | おもちゃ片付け用 |
| `LaundryBoard` | `"is_laundry"` | `board_laundry.urdf` | 洗濯物集め用 |
| `FoodBoard` | `"is_food"` | `board_food.urdf` | しまう／食品用（`store_items`） |

- [ ] 上記 3 クラスを `objects/items.py` に追加（または `Board` に `board_type` 引数を持たせる）
- [ ] 各 board に `is_platform = True`, `move_type = None`, `bbox_half_extents` を設定
- [ ] URDF を `objects/assets/` に追加（既存 `board.urdf` の色違いコピーでも可）
- [ ] `Shelf` を board として使う場合は `is_platform = True` と `accepts_category` を追加する方針を決め、採用時のみ実装

#### 1.5-C: item 側の確認・補充

| タスク種別 | 必要な item 属性 | 対応 board | 既存 item | 追加要否 |
|-----------|----------------|-----------|----------|---------|
| `tidy_toys` | `is_toy` | `ToyBoard` | `ToyCar`, `Ball`, `Rattle`, `Lego`, `TeddyBear`, `Jenga` | 不要（既存で可） |
| `laundry` | `is_laundry` | `LaundryBoard` | `Towel` | 任意（`Sock` 等を追加する場合のみ） |
| `store_items` | `is_food` | `FoodBoard` | `Apple`, `Banana` | 任意 |

- [ ] 各 item に `is_graspable = True`, `move_type != None` が設定されていることを確認
- [ ] （任意）洗濯物・食品のバリエーション item を追加

#### 1.5-D: 環境への組み込み

- [ ] `base.py` の `_load_urdfs` が新クラス名を解決できることを確認（`importlib` による動的ロードのため、クラス名を `items.py` に追加すれば可）
- [ ] デモ用 `fixed_placements` に 3 種の board を配置するレイアウト例を用意
  - 例: `cueworld/demo/demo_household_solo.py` または `demo_initialize_pos.py` を更新
  - row=0 / row=max の固定行に board を置き、中間行に可動 item を配置
- [ ] `object_counts` で各タスクの item が最低 1 個ずつスポーンすることを確認

#### 1.5-E: 配置フローの動作確認

- [ ] `is_platform` 経由の `place_on_board`（`base.py` `_update_grasp`）が新 board で動作する
- [ ] board 上に置いた後、`objects_on_board` / `put_on` が正しく更新される
- [ ] 誤ったカテゴリの item を近づけてもプロトコル側で選ばれないことを確認（Phase 2 前の手動テストでも可）

**主な変更ファイル:** `objects/base.py`, `objects/items.py`, `objects/assets/*.urdf`, `cueworld/demo/demo_household_solo.py`

**完了基準:**

1. 3 タスク種別それぞれについて、(item, board) のペアが環境に存在する
2. 各 board は `accepts_category` を持ち、対応する item 属性と一致する
3. デモ配置で board に item を 1 個置ける（プロトコル未接続でも手動操作 or 簡易スクリプトで確認可）

---

### Phase 2: タスク種別の抽象化

**目的:** Phase 1.5 で揃えたオブジェクトを前提に、ハードコードされた探索を再利用可能なタスクスキーマに整理する。**タスク選択はランダムだが、現在の状況（active な可動 item・board の有無）で成立するものに限定し、不可能な種別はスキップする**（→ 設計方針 §2「タスク選択方針」）。

**依存:** Phase 1.5 完了（board / item が揃っていること）

- [x] `TASK_SCHEMAS` dict を導入（`cueworld/household_schemas.py`）
- [x] `BaseEnv` で `household_task_candidates` を毎 `reset` 時に構築（現状を反映した成立候補のみ）
- [x] `start()` は `household_task_candidates` からランダム選択（空なら `completed` でスキップ）
- [x] 旧コンテナベースの `target_category` 探索ロジックを削除
- [x] 複数タスク種別が同時に可能な場面でランダム選択が機能することを確認

**完了基準:**

1. `tidy_toys`, `laundry`, `store_items` の 3 種別すべてが、**条件が揃うシーンで**選ばれて実行できる
2. 対応 item や board が無い種別は候補に入らず、選ばれない
3. 成立候補が 0 件のときプロトコルが即終了する

---

### Phase 3: 確率的サブゴール選択（foraging）

**目的:** 一直線な `target_items[0]` をやめ、意図は保ちつつ対象選択を揺らす。

- [x] `select_next_item(candidates, style)` 関数を実装
  - 距離ベースのソフトマックス
  - カテゴリ典型性によるバイアス
  - `household_style` による重み付け
  - household_styleの確率は固定（デフォルト `committed`、Phase 6 で確率割当）
- [x] `_enter_moving_to_item()` で `select_next_item` を呼ぶ（開始時・grasp/place 失敗時の再選択）
- [x] デバッグログに選択理由（距離 / 典型性 / style）を出力
- [x] `item_selection_history` / `last_item_selection` を `debug_info` に記録

**完了基準:**

1. 同一シード・同一初期配置でも、複数回実行で拾う順序が変わる
2. 同じ `global_intent` は維持される

**テスト:** `cueworld/tests/test_household_foraging.py`

---

### Phase 4: inspection / hesitation 状態

**目的:** 中間状態を入れて「選別している」振る舞いを作る。

- [x] `inspecting_item`, `inspecting_board`, `hesitating` 状態を追加
- [x] インスペクションのときの親の動きは首を動かすことで行う（`gaze_object`, `approach_object` 不使用）
- [x] 遷移条件: サブゴール選択後に確率で inspection / hesitation に入る、一定ステップで本処理へ
- [x] grasp 成功後に確率で `inspecting_board` へ
- [x] `_determine_action` で接近せず首・視線だけ動かす
- [x] `state_proximities` に inspection 段階を反映
- [x] `inspection_history` を `debug_info` に記録

**完了基準:**

1. ログ上で inspection が挿入される
2. 全体の完了時間が延びる
3. `global_intent` は変わらない

**テスト:** `cueworld/tests/test_household_inspection.py`

---

### Phase 5: 完遂条件の緩和

**目的:** 「全部終える」から「十分やった」へ。

- [ ] `completion_policy` を導入（`min_items`, `max_items`, `max_steps`, `prob_early_stop`）
- [ ] `check_completion()` で n 個達成 or タイムアウト時に `completed` へ遷移
- [ ] 未処理の候補が残っていても終了できることを確認
- [ ] `protocol_manager` 側で次プロトコルへの切り替えが自然に行われるか確認

**完了基準:** 候補が 5 個あっても 2–3 個で終了するケースが発生する

---

### Phase 6: household_style と幼児シグナル連動

**目的:** `infant_signals` を使った振る舞いの揺れを入れる。

- [ ] `start()` で `household_style` を確率的に割り当て
- [ ] `_manage_state` で `infant_signals`（接近、視線、発声等）を参照
- [ ] `monitoring` モード: 一定確率で幼児方向を向く中間状態
- [ ] `interrupted_but_returning` モード: シグナル強度が閾値を超えたら短い中断 → 元のサブゴールに復帰
- [ ] `persistence` を style ごとに変動させる

**完了基準:** 幼児が近づいたときに親の行動が一時的に変化し、その後家事に戻る

---

### Phase 7: 検証・チューニング・記録

**目的:** 研究利用に耐える品質に仕上げる。

- [ ] 複数シードでの行動ログ収集（選択順序、inspection 頻度、完了個数の分布）
- [ ] `debug_info` に `global_intent`, `household_style`, 選択履歴を記録
- [ ] 既存テスト（`protocol_manager` 関連）の更新
- [ ] パラメータ（確率、閾値、完了ポリシー）のデフォルト値を調整
- [ ] 必要に応じてデモスクリプト `cueworld/demo/` を更新

**完了基準:** 意図図式（中域で読める・近接では読めない）が定性的に確認でき、既存テストが通る

---

## フェーズ依存関係

```
Phase 0 ──→ Phase 1 ──→ Phase 1.5 ──→ Phase 2
                                          │
                                ┌─────────┼─────────┐
                                ▼         ▼         ▼
                           Phase 3   Phase 4   Phase 5
                                │         │         │
                                └─────────┼─────────┘
                                          ▼
                                     Phase 6
                                          ▼
                                     Phase 7
```

- Phase 0 → 1 → 1.5 → 2 は**直列必須**
  - Phase 1: 1 ペアで FSM が動く土台
  - Phase 1.5: タスク種別ごとのオブジェクトが揃う
  - Phase 2: オブジェクトを `TASK_SCHEMAS` で抽象化
- Phase 3–5 は Phase 2 完了後に**並行可能**（独立した機能追加）
- Phase 6 は Phase 3–5 のいずれかが入っていると効果を確認しやすい
- Phase 7 は全体完了後

---

## 各フェーズの見積もり（目安）

| Phase | 規模 | 主な変更ファイル |
|-------|------|-----------------|
| 0 | 小 | `plan.md`, `objects/items.py`（調査） |
| 1 | 中 | `household_solo.py` |
| 1.5 | 中 | `objects/base.py`, `objects/items.py`, `objects/assets/`, `cueworld/demo/` |
| 2 | 小〜中 | `household_solo.py` |
| 3 | 中 | `household_solo.py` |
| 4 | 中 | `household_solo.py` |
| 5 | 小 | `household_solo.py` |
| 6 | 中 | `household_solo.py`, `infant_signal.py`（参照のみ） |
| 7 | 小〜中 | テスト, デモ, `plan.md` 更新 |

---

## 参考: 現行コードの削除対象

Phase 1 で削除・置換する現行要素:

- 状態: `moving_to_container_for_open`, `opening_container`, `moving_to_container_for_put`, `putting_in`, `moving_to_container_for_close`, `closing_container`
- 発話: `start()` 内の `queue_utterance("wash" / "cleanup")`
- アクション: `open_close_object`, `put_in_object`

---

## 未決事項

### Phase 0 / 1.5 で決める（オブジェクト関連）

1. board の実装方針: タスク別サブクラス（`ToyBoard` 等）vs 汎用 `Board` + コンストラクタ引数
2. URDF: 色違いコピーで十分か、形状も変えるか
3. `Shelf` を board として流用するか、専用 board クラスに統一するか
4. `store_items` の item 属性を `is_food` に固定するか、別属性（`is_storable`）を新設するか
5. デモ配置: 3 種の board を 1 シーンに同時配置するか、タスクごとにシーンを分けるか

### Phase 0 で決める（プロトコル関連）

6. タスク種別の初期セット（2 種 vs 3 種）→ **3 種**（確定）
7. `household_style` の確率分布（固定 vs `protocol_manager` から渡す）
8. inspection 中の親の動作（首だけ vs 腕を伸ばす vs 静止）→ **首だけ**（Phase 4 で確定）
9. 完了後の未処理オブジェクトを次回の `start()` で再候補にするか → **はい**（`put_on is None` の item を毎 reset 再評価）
10. タスク選択の状況依存範囲 → **毎 `reset` 時の `household_task_candidates` 構築時**（設計方針 §2 参照）

## 既知の問題

- `Apple` は把持が安定せず `store_items` で失敗しやすい。デモでは `Banana` を使用
- 掴むのが安定しないオブジェクトを修正する．(大きさを大きくするなど)
- 変な物体が誕生しているのが気になる
