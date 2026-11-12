# 環境まわりの修正・検討メモ（指示用）

親ドキュメント: [docs/env.md](docs/env.md)

各項目は、そのまま作業指示プロンプトとして使える粒度で書いている。
未決事項は「検討」と明記し、方針が固まったものは「方針」として書く。

## 推奨実装順

| 順 | 項目 | 複雑さ | 理由 |
|----|------|--------|------|
| ✅ | §1 オブジェクト簡素化 | —（完了） | 把持エラーの土台。これ以降のプロトコル・アフォーダンス検証が不安定なままになる |
| ✅ | §2 アフォーダンス削減 | —（方針・振る実装済み） | 残す相互作用の前提。詳細は §2 |
| ✅ | §4 幼児ボディ | —（完了） | 非 `*_2` URDF・同サイズ・`torso_link` 色分け済み。到達範囲は親と同寸法前提で §3 に進める |
| ✅ | §3 手渡しの要否 | —（完了） | 到達圏 `place_object_at`。`eval_object_giving_place` OVERALL PASS |
| ✅ | §5 プロトコル再定義 | —（カタログ確定・個別仕上げ済み） | 5本の FSM・デモ・テストあり。連鎖スモークは §6 |
| **now** | §6 遷移行列 | **中〜高** | A–E・H 完了。バッチ F は **方針確定（dig 済）**。次は F の実装 |

### 複雑さの目安
| レベル | 目安 | イメージ |
|--------|------|----------|
| **低** | 半日〜1日 | 方針がほぼ決まっており、変更箇所が局所的 |
| **中** | 数日 | 複数ファイルにまたがるが、設計の大枠は見えている |
| **高** | 1週間前後〜 | 設計判断＋複数プロトコルの仕上げ・検証が必要 |

### 進め方の注意
- §1→§2 は連続作業（物体資産の変更が両方にまたがる）※§1 は完了済み
- §4 は完了済み（親子同サイズ・`torso_link` 色分け）
- §5 の個別プロトコルは仕上げ済み。§6 はバッチ A–E・H 完了。バッチ F（JA↔giving の連鎖・除外リスト・board）は **dig 済・未実装**

---

## 1. オブジェクト設計の簡素化 ✅ 完了

### 目的
オブジェクト形状が複雑すぎて、親の把持判定エラーが増えている。把持の複雑さは取り扱いたい問題ではない．複雑な形状・大きさ変化を抑え、色と基本形だけで区別できるようにする。

### 方針
- オブジェクトはプリミティブ立体のみにする
  - 立方体 / 球体 / 円錐 / 円柱
- 識別は **色** と **プリミティブな形** を主とする（複雑な形状・大きさ変化は使わない）
- `board`（プラットフォーム）は継続して使用する
- 実装方針
  - 4色 `red / blue / green / yellow` × 4形状で 16 クラスを用意する
  - `concept_id` は `red_cube` のような **色+形の複合 ID** にする
  - `household_solo` の分類軸は色ベース（`collect_red` など）にする
  - 旧オブジェクト定義は `items_legacy.py` に退避し，資産ファイルは残す

### 検討
- 色や形に応じた特徴付け（例: 色ごとにアフォーダンスや音の違い）を入れるか
  - → §2 側で扱う（色ごとの差分を入れるかはアフォーダンス設計と一体）

### 注意
- オブジェクト種類を参照しているプロトコルなどに関しては何かしらの書き換えが必要かも．
- 修正後，各プリミティブ形状に対して親が掴む行為がうまくいくかを定量的に評価してほしい．

### 評価結果
- `scripts/experiment/eval_grasp_primitives.py` を追加し，把持成功率を形状別に評価した
- 実行環境: コンテナ `d994f05625a7`
- 条件: 各形状 30 trial，`ParentTools.grasp_object()` を用いて親が把持
- 結果
  - cube: 30 / 30 成功，平均 12.90 step
  - sphere: 30 / 30 成功，平均 12.73 step
  - cylinder: 30 / 30 成功，平均 12.93 step
  - cone: 30 / 30 成功，平均 13.07 step
- 形状間の大きな偏りは見られなかったため，寸法・摩擦の追加調整は不要と判断した

---

## 2. オブジェクトのアフォーダンス削減

**複雑さ: 中**（削減・配線が主。新規物理挙動をちゃんとやるなら中の上）

### 目的
アフォーダンスを少数に絞り、インタラクションの意図を明確にする。contain / open 等は優先度を下げる。
**研究上の焦点は共同注意**（視線・指差しが集まりやすい変化と、三項関係でデモできる行為）。

### 方針（残す）

| アフォーダンス | 効果 | パラメータの割当 | メモ |
|----------------|------|------------------|------|
| **振る → 音** | 音が出る | **形 → 音色・pitch** | デモ可能な1行為の核。「物体で何かする → 一緒に見る／渡す」 |
| **行為 → 色変化** | 見た目の色が変わる | （効果自体が色チャネル） | 振る／触る等の**行為の結果**として変化。「何をしたか」と saliency がセット。注目ハイライト演出は足場かけ側とし、ここでは採用しない |

**振るの定義**: そのオブジェクトを把持している状態で、オブジェクトの座標が変化する（しきい値ありでもよい）。
**オブジェクトの出す音をどう扱うか**: 親や幼児の発話とは別にものすごく簡易な2次元ベクトルで表現．(何も音がしないときは[0.0, 0.0])
**出す音はどこで定義する?**: オブジェクトクラスが保持．固定値．

**例外**: **board のみ**載せる（くっつける）可。判定は **色一致**（`accepts_category`）。household 用で、共同注意とは別枠。

### 後回し・採用しない
- 触ると揺れる / 触ると音 — 振る・色変化と役割が被る。偶然接触のノイズにもなりやすい
- 形ごとに物理が違う（球だけ転がる、円錐だけ倒れる等） — ターゲット移動で共同注意が不安定／デモが形分岐する。※振る音の形差は含めない（上表で採用）
- contain / open — 優先度低のまま

### 色・形の役割分担

識別は §1 どおり **色 × 形** の複合 ID（例: `red_cube`）。「何の物体か」と「効果がどう違うか」は分け、次のように割る（board・色変化が色に寄るため、振る音は形に載せて偏りを是正する）。

| 特徴 | 識別・カテゴリ上の手がかり | 効果・判定への割当 |
|------|----------------------------|--------------------|
| **色** | board マッチング、語彙・カテゴリ、色変化の saliency（視線・指差し向き） | board 判定、行為→色変化 |
| **形** | 物体同一性・もう一つのカテゴリ軸、把持の見え方 | **振る音の音色・pitch** |

- **形で変えてよい**: 同じ行為（振る）の効果パラメータ（音色）。行為は一本のまま。
- **形で変えにくい（共同注意向け）**: できる行為や物理結果そのもの（転がる／倒れる等）。
- **振る音を色に載せる案**は非推奨（board＋色変化＋音色がすべて色になり、形がラベルだけになる）。

### 確定した設計判断（実装済み）

- **色変化は不採用**。`color` は `concept_id`・語彙・board の `accepts_category` に使われるため、カテゴリ色を動かすと同一性判定と board 判定が壊れる。代わりに**振る → 効果チャネル**（色由来の 2 次元抽象ベクトル）を独立モダリティとして追加した。
- **トリガーは振る一本**。振ると音（形由来）と効果チャネル（色由来）が同時に出る。行為が一本なのでデモが形・色で分岐しない。
- **contain / open はフラグ無効化のみ**（`is_containable=False`）。env のコードパスは残す。コード削除は §5 とセット。

役割分担:
- 色 → board 判定（`accepts_category`）、語彙・カテゴリ、**`object_effect` チャネルのベクトル**
- 形 → 物体同一性、**`object_sound` チャネルのベクトル（音色・pitch）**

### 注意
これが関連するファイルが別にないか注意

### 評価結果

- `scripts/experiment/eval_shake_sound.py` を追加し，振る検出精度と偽陽性率を評価した
- 実行環境: コンテナ `d994f05625a7`
- 条件: 全 16 クラス × 5 trial，`shake_object()` を 40 step 実行 / 歩行のみ 40 step
- 結果（`shake_move_threshold=0.002`）
  - 発火率（振る）: **0.88**（正弦波の頂点/底付近の 3 step/周期がしきい値以下になるため 100% でないことは想定内）
  - sound ベクトル一致率: **1.0**（発火時は常に形から期待される固定値と一致）
  - effect ベクトル一致率: **1.0**（発火時は常に色から期待される固定値と一致）
  - 偽陽性率（歩行のみ）: **0.0000**（歩行では一切発火しない）
- `shake_move_threshold=0.002` を採用（歩行時の最大変位が 0 に対し、振る時の変位は 0.003〜0.010 m/step の範囲）

---

## 3. 物体の手渡し（object giving）の要否

**複雑さ: 方針決めは低 / 実装は中**

### 目的
「親 ↔ 幼児で直接手渡しする」必要があるかを再検討する。

### 方針
- **直接の `hand_over` は採用しない**
- object giving は「親が対象を幼児の到達範囲に配置し、幼児の把持（または明確な拒否）を待つ」プロトコルとして再定義する
- プロトコルの役割（三項・命令的・regulation）は残す。変えるのはモータ実現のみ（`hand_over` → 接近＋到達圏内への place）
- 成功条件は「置いた位置が幼児到達圏内」＋「幼児が把持」（現状の `infant["held_obj"] == target` と整合）
- 置く位置は幼児正面・床上・衝突しないなど一意にし、共同注意のターゲット曖昧さを増やさない
- `mouth_feeding`（口元運び）は当面スコープ外
- §5 の「渡し合う」も、双方が相手の到達圏に置く形で足りる（往復の意味は保たれる）

### 根拠
- 取りたい現象は物理的な受け渡しではなく、「相手に物を使える状態にして関わりを促す」こと
- 現状の成功条件もすでに幼児の把持側にあり、手から手への遷移を見ていない
- `hand_over` は二体同期の把持競合・落下・タイミングずれの温床で、§1 で簡素化した把持の難しさを再導入しやすい
- 「振って見せる → 届くところに置く → 幼児が取る／振る」は共同注意＋ regulation として読みやすい

### 現状と見積もりの根拠
- 現状 `ObjectGivingProtocol` は `hand_over` による直接受け渡し前提（実装は仮）
- 「置くだけ」にすると: 到達範囲の定義（§4 のボディ寸法と連動）＋ place 動作＋成功条件の書き換えが必要
- 方針は確定。変更箇所は主に `object_giving` と親モータ周りに閉じる
- §5 の object giving / オブジェクト遊び定義に直結するため、実装タイミングは §5 とセットになりやすい
- 実装は §4（到達半径）確定後に着手する

---

## 4. 幼児ボディの見た目・サイズ ✅ 完了

**複雑さ: 低**

### 目的
親子の視覚的区別と、実装上の扱いやすさを両立する。

### 方針（実装済み）
- **食べる動作は使わない**ため、`cue_robot_*_2.urdf` は使わず、`cue_robot_parent.urdf` / `cue_robot_infant.urdf` を使用
- **基本サイズは親子同じ**（幼児 URDF を親と同寸法に揃えた）
- 視覚的区別は **`torso_link` の色**のみ（親: 黄 `0.8 0.8 0` / 幼児: 青 `0.2 0.45 0.9`）
- `_create_agent(..., scale=1.0)` → `p.loadURDF(..., globalScaling=scale)` を追加（将来の一様スケール用。当面は両方 `1.0`）
- 非 `*_2` に `jaw_link` がないため、`hand_over(mouth_feeding=True)` は `jaw_link is None` なら手元へフォールバック

### 検討メモ: `_create_agent` でスケール指定
- PyBullet `globalScaling` で link 比率を保ったまま全体スケール可能（上記 API で利用可）
- 今回の同サイズ化は URDF 寸法揃えで行い、スケール差分は使っていない

---

## 5. 必要なプロトコルの再定義

**複雑さ: 高**（項目ごとのばらつきあり。全体として最大コスト）

### 目的
「本当に必要なプロトコル」だけを残し、それぞれがきちんと使えるレベルまで仕上げる。

### 方針（確定カタログ）

残すプロトコルは次の **5つ**（dig で確定）。

#### インタラクションプロトコル
| 名称（キー） | 関わりの型 | メモ | 個別の複雑さ |
|--------------|------------|------|--------------|
| `dyadic` | 二項・social | 旧 `imitation` をリネーム。物体なしの声・身体模倣 | **中〜高**（リネーム＋仕上げ） |
| `joint_attention` | 三項・social | cue に `shake` 追加可（fetch〜把持〜振る）。置くは含めない | **中〜高**（shake cue 追加＋仕上げ） |
| `object_giving` | 三項・regulation | §3: 到達圏へ place。モードなし。既持時は fetch 明示スキップを §3 改修で | **中**（§3 改修が本体） |

#### オブジェクト遊び（独立プロトコルにしない）
- 表現: `joint_attention`（shake）→ `object_giving` → 幼児が振って手放し置く → … の繰り返し
- 社会性の計上: プロトコル履歴からシーケンス検出し、コミュニケーション指標へ事後加算（詳細・選択への履歴利用は §6）
- 旧 `object_play` は復活させない

#### ソロプロトコル
| 名称（キー） | 挙動 | 個別の複雑さ |
|--------------|------|--------------|
| `observance` | 見守り。インタラクションへ高遷移は §6 | **低〜中** |
| `household_solo` | 家事。インタラクション非遷移は §6 | **低〜中** |

#### カタログ外（§5 実装対象外）
`physical_contact` / 旧 `object_play` / `dyadic_play`（peekaboo）。ファイル残置可。
距離規制の独立プロトコル（「近くに来て」）も **追加しない**（下記追記 dig）。

### 現状と見積もりの根拠
- カタログ本数と責務は dig で確定。残りは各プロトコルの仕上げと §3／shake cue／§6
- §2 の振る／音・effect と §3 の「到達圏への配置」方針は確定済み。object giving 実装は §3 改修がゲート
- オブジェクト遊びは独立単位を持たず、JA・giving 仕上げ後の連鎖確認＋§6 の履歴検出に分散

### 進め方（方針）

ボトルネックは実装量より、**プロトコル単位の動画による定性確認ループ**。§5 全体を一気にプランニングして実装し切るのは避ける。

**「カタログ確定 → プロトコル単位の縦スライス」**で進める。

```
① カタログ確定 ← 完了
② インタラクションを1本ずつ
   薄く計画 → 実装 → 定量スモーク → 動画定性 → 修正 → 受け入れ
③ ソロは軽く触る／据え置き（遷移は §6）
④ §6 は最後（履歴による遊び検出・継続優遇を含む）
```

### 分割粒度（方針）

| 粒度 | 例 | 評価 |
|------|----|------|
| 粗すぎ | §5 全部を1本のプラン | 動画フィードバックが遅すぎて手戻り大 |
| **ちょうどよい** | **プロトコル1本＝1サイクル** | 計画コスト小、定性の単位と一致 |
| 細すぎ | approach / grasp / gesture を別プラン | 計画オーバーヘッドが大きい。動画は結局全体を見る |

**作業単位**（カタログ確定後）:

| 順 | 単位 | 内容 |
|----|------|------|
| ✅ | カタログ確定 | 完了（下記 dig 節） |
| 1 | object giving（§3） | 到達圏 place・既持スキップ。遊び連鎖の前提 |
| 2 | joint attention | `shake` cue（fetch〜把持含む）＋既存指差し／注視の仕上げ＋動画 |
| 3 | dyadic | `imitation` → `dyadic` リネーム＋仕上げ＋動画 |
| 4 | 連鎖スモーク | JA↔giving のオブジェクト遊び様シーケンスを動画確認（独立プロトコルは作らない） |
| 5 | observance / household | 必要なら最小。遷移ガードは §6 |

各単位の「計画」は次だけ決めれば十分（着手時にその場でよい）。

- 成功条件（何が起きれば終わり／成功か）
- 親のフェーズ列（3〜5個程度）
- 今回やらないこと
- 動画で見るチェックリスト（3〜5項目）

### 分担イメージ

- **人間**: カタログの最終判断、各プロトコルの動画定性、受け入れ判定
- **実装側**: 単位ごとの薄い計画 → 実装 → 定量スモーク（コンテナ `d994f05625a7`）→ 動画書き出し

実装プランは**次に触る1プロトコル分だけ**にする。

### カタログ確定（dig 完了）

#### 確定カタログ（5つ）
`dyadic` / `joint_attention` / `object_giving` / `observance` / `household_solo`

#### dyadic
- **本体**: 現行 `ImitationProtocol`（物体なしの声・身体模倣: voice / arm_wave / head_nod / hand_open_close）。物体は扱わない（確定）
- **`DyadicPlayProtocol`（peekaboo）**: 廃止（カタログ外・§5 実装対象外）。ファイル残置は可（確定）
- **キー／クラス名**: `imitation` → `dyadic` / `DyadicProtocol` へリネーム（§5 の dyadic 単位で実施）（確定）
- **物体行為（振る等）**: `dyadic` には入れない。振るデモ／物体行為の模倣は三項側（確定）

#### 削るもの
- **`physical_contact`**: カタログ外・§5 対象外。ファイル残置可（確定）
- **旧 `object_play`（contain 寄り）**: カタログ外・§5 で復活させない。ファイル／デッド重みの掃除は実装時または後で可（確定）
- **`dyadic_play`（peekaboo）**: 上記 dyadic 節どおり廃止（確定）

#### オブジェクト遊び / 三項の組み合わせ
- **表現**: `joint_attention` → `object_giving` → 幼児が振って見せて手放し置く → `joint_attention`（親が振って見せる）→ `object_giving` の繰り返し（組み合わせ。独立プロトコルにはしない）（確定）
- **前提**: 遊び連鎖の前に、`object_giving` を §3（到達圏への配置、直接 `hand_over` しない）に合わせて改修する
- **分類**: `object_giving` に事前モードは持たない。オブジェクト遊び様シーケンスは **直近プロトコル履歴から検出し、コミュニケーション関連指標へ事後加算**（基本は指標・ログ）（確定）
- **遷移**: 遊び継続の優遇はプロトコル選択側（`infant_signal` ＋プロトコル履歴）。詳細は **§6**（確定）
- **不採用**: JA に振る・置くを吸収（スケール混在）／giving の事前モード（起動時に区別不能）
- **振るデモ**: `joint_attention` の cue（`cue_type="shake"`）。成功条件は共同注意。置く・渡すは含めない（確定）
- **JA-shake の把持**: fetch → grasp → shake → 共同注意チェックまで含む（確定）

カタログ確定の dig はここまで。以降は実装単位に進む。細部（検出パターンの厳密定義等）は着手時または §6 で掘る。

### 追記 dig: 距離規制プロトコル（完了）

- **核の現象**: 幼児起点の「近くに来て」（二項・規制）（確認済み）
- **カタログ追加**: **しない**。距離要求は既存プロトコル選択のバイアスや、JA／giving 開始時の approach で吸収。二項規制の穴は当面許容（確定）
- **遠ざかる**: 独立プロトコルにしない。拒否は既存の `detect_rejection` 中断で足りる。一歩下がる演出が必要なら各プロトコル失敗時の共通クリーンアップ（後回し可）（確定）
- **含意**: 確定カタログ5つのまま。距離まわりでカタログを増やさない

---

## 6. プロトコル遷移判定の見直し

**複雑さ: 中〜高**（骨格はある。行列の再設計・チューニングが本体）

### 目的
各プロトコルへの遷移が意図どおり起きるよう、遷移計算（行列）を見直す。

### 現状の意図（§5 と連動）
- observance → 高確率でインタラクション系へ遷移
- household → インタラクション系へは遷移しない
- インタラクション系同士・オブジェクト遊びの遷移条件を明確にする

### 現状と見積もりの根拠
- `protocol_manager.py` にシグナル重み行列・ペナルティ・response/request リストは既にある
- §5 でプロトコル集合が変わるたびに行・列と重みを作り直す必要がある
- 実装そのものより、「意図どおり遷移するか」の調整・検証コストが大きい
- household 非遷移・observance 高遷移はルール追加としては比較的軽いが、インタラクション同士の条件明確化は検証込みで重い

---

## 参考（env.md からの補足）

- テーブルを挟んで物を操作する環境にしていない理由: 指差しと物掴みが曖昧になることを避けるため。幼児の動きを制限すれば、実質的に近い条件にもできる。
- 環境の中核は `BaseEnv`（`base.py`）。その上に `infant_signal` / `protocol_manager` / 各プロトコルが乗る構成。


---

## §3・§5 の整理（横断 / プロトコル別）

§3（手渡し）と §5（プロトコル再定義）の確定内容を、実装・指示用に再配置したもの。詳細・経緯は上文の §3・§5 を参照。

### A. プロトコル種類に関係なく影響する内容

#### カタログと削るもの
- **残す5つ**: `object_giving` / `joint_attention` / `dyadic` / `observance` / `household_solo`
- **カタログ外**（§5 実装対象外、ファイル残置可）: `physical_contact` / 旧 `object_play` / `dyadic_play`（peekaboo）
- **距離規制の独立プロトコルは追加しない**（「近くに来て」「遠ざかる」とも）。距離要求は選択バイアスや各プロトコル内 approach で吸収。拒否は既存 `detect_rejection` 中断。一歩下がる演出は後回し可の共通クリーンアップ
- 二項・規制の穴（近くに来て）は当面許容

#### 物の受け渡し・配置（モータ方針）
- **直接の `hand_over` は採用しない**
- 「渡す／渡し合う」は **相手の到達圏へ床上配置（place）** で実現する（往復も同様）
- 置く位置は幼児正面・床上・衝突しないなど一意にし、共同注意のターゲット曖昧さを増やさない
- 成功の核は「使える状態にする」ことであり、手から手への物理遷移ではない
- `mouth_feeding`（口元運び）は当面スコープ外
- 到達半径は §4（親子同寸法）前提

#### オブジェクト遊び（独立プロトコルにしない）
- 表現: `joint_attention`（shake）→ `object_giving` → 幼児が振って手放し置く → … の繰り返し
- `object_giving` に事前モード（regulatory / social_play）は持たない（起動時に区別不能）
- 社会性の計上: **プロトコル履歴からシーケンス検出し、コミュニケーション指標へ事後加算**（指標・ログが主）
- 遊び継続の優遇: プロトコル選択側（`infant_signal` ＋履歴）。詳細は **§6**
- JA に「振る＋置く」を吸収しない（JA のスケール混在を避ける）

#### 進め方・評価
- ボトルネックは動画による定性確認。§5 全体を一気に実装しない
- **プロトコル1本＝1サイクル**（薄く計画 → 実装 → 定量スモーク → 動画定性 → 受け入れ）
- 評価環境: コンテナ `d994f05625a7`
- 動画
- **§6（遷移行列・履歴検出・継続優遇）はプロトコル仕上げの後**

#### 共有モータ・横断クリーンアップ（実装時の共通部品）
- 到達圏 place（giving と、渡し合い連鎖で共用）
- `shake_object`（JA の shake cue から使用。§2 済み）
- 旧プロトコル／デッド重み（`object_play` 等）の掃除は実装時または後で可
- contain / open のコードパス削除は §5 とセット（フラグ無効化は §2 済み）

---

### B. プロトコルについての内容（実装順）

推奨順: **1 object_giving（§3）→ 2 joint_attention → 3 dyadic → 4 連鎖スモーク → 5 ソロ（最小）→ §6**

#### 1. `object_giving`（三項・regulation）— ゲート

- 役割（三項・命令的・regulation）は残す。変えるのはモータ実現
- **`hand_over` → 接近＋幼児到達圏内への place**
- 成功条件: 置いた位置が幼児到達圏内 ＋ 幼児が把持（`infant["held_obj"] == target` と整合）／または明確な拒否
- 遊び連鎖の前提のため、JA shake より先に仕上げる
- 変更箇所の主座: `object_giving` と親モータ周り

#### 2. `joint_attention`（三項・social）

- **`cue_type="shake"` を追加**: fetch → grasp → shake → 共同注意チェックまで含む
- 成功条件は共同注意のまま。**置く・渡すは含めない**（place は giving）
- 振るデモ／物体行為はこちら（`dyadic` には入れない）

#### 3. `dyadic`（二項・social）

- 本体: 現行 `ImitationProtocol`（voice / arm_wave / head_nod / hand_open_close）。**物体は扱わない**
- キー／クラス: `imitation` → `dyadic` / `DyadicProtocol` へリネーム
- `DyadicPlayProtocol`（peekaboo）は廃止（カタログ外）
- 物体を振っての模倣は入れない（三項側）
- 複雑さ: 中〜高（リネーム＋仕上げ）

#### 4. 連鎖スモーク（独立プロトコルは作らない）

- JA（shake）↔ `object_giving` のオブジェクト遊び様シーケンスを動画確認
- 履歴検出・指標加算・選択優遇の本実装は §6（ここでは振る舞いが繋がることの確認が主）

#### 5. `observance` / `household_solo`（ソロ）

- 本体は据え置きまたは最小タッチ
- observance → インタラクション高遷移、household → インタラクション非遷移は **§6**
- 複雑さ: 低〜中

#### （参考）カタログに載せないもの
- `physical_contact` / 旧 `object_play` / `dyadic_play` / 距離規制プロトコル — 上記 A 参照

## 追記（§6 前に判明した修正点・優先順）

これまでの編集・動画確認で出た追記。**§6（遷移判定）に入る前に**、下記の順で潰す／織り込む。
既存の実装順（giving → JA → dyadic）と依存関係を揃えた。

### 推奨順（一覧）

| 順 | 項目 | 優先度 | 複雑さ | 依存・タイミング |
|----|------|--------|--------|------------------|
| **A** | 把持距離・`reach_radius`（旧3） | **最高・実装済** | 中 | 物体操作すべての土台。giving / escalate / JA の前提 |
| **B** | 発話テンプレ簡素化（旧1） | **高・実装済** | 低 | 言語基盤。escalate の声掛け・発話増の前提 |
| **C** | 親の1音素長の統一（旧2） | **高・実装済** | 低 | B とセット（1音素=6） |
| **D** | プロトコル内の親発話タイミング増（旧4） | **中高・実装済** | 低〜中 | B・C の後。**E と一体設計**（giving の RETRY／ESCALATE に閉じる） |
| **E** | `object_giving` escalate 見直し（旧5） | **中高・実装済** | 中 | **動画確認は人間**。A・B・C 済。D は本項に包含 |
| **H** | `show_object` 目標点の固定（ESCALATE 目視バグ） | **中・実装済** | 低〜中 | E 実装済の修正。**F の前提** |
| **F** | shake 動作の掲げ感（旧6） | **中・実装済** | 低〜中 | H の後。JA 仕上げ（`compute_show_position` 共有） |
| **G** | dyadic 模倣の手の動き拡大（旧7） | **中・実装済** | 低〜中 | H/F の寿命モデルに揃える。dyadic 仕上げ時 |
| — | object_sound 同時発生（base.py） | 後回し・普通 | — | 今は触れない．§6 以降でも可 |
| — | プロトコルごとのスケール調整 | 後回し・低 | — | 今は触れない。§6 以降でも可 |

```
A 把持距離・reach_radius
    ↓
B・C 発話テンプレ簡素化 ＋ 音素長統一（並行可）
    ↓
D＋E giving escalate 見直し（発話増は E に包含）  ※ A・B・C 必須
    ↓
H show_object 目標固定（E のバグ修正）
    ↓
F shake 掲げ（JA） → G 模倣モータ（dyadic）
    ↓
§6 遷移判定
```

### A. 把持距離・`reach_radius`（旧3）— 最高

giving の place・escalate の再把持・JA の fetch/grasp の前提のため **最先**。dig で方針確定・**実装済み**。

#### 方針（dig 確定）

**スコープ**
- 含む: `grasp_distance` / `reach_radius` / `place_dist` / 把持用 `approach_dist` / giving の `stand_dist`
- 含む（配線）: JA の**把持** `approach_dist` → `env.grasp_approach_dist` に置換
- 含めない（JA 仕上げ時）: `shake_stand_dist`（見せる距離。演出ノブ）

**正の置き場所**
- **env を正**とする。エージェント／物体ロード後にまとめて計算
- 属性例: `env.grasp_distance` / `env.reach_radius` / `env.place_dist` / `env.grasp_approach_dist` / `env.stand_dist` / `env.measured_floor_reach_xy`
- プロトコルはそれを読む（距離の直書き禁止）。コンストラクタ明示指定があれば上書き可
- 親子同寸法前提のため `reach_radius` は共有1本（計測は片方でよい）

**`grasp_distance`**
- 意味: palm 基準の把持検出しきい値（到達幾何とは別スケール）
- 式: `palm_half + max_object_half + grasp_margin` → **環境の1定数**（既定 `grasp_margin=0.12` → 実測 ≈0.30）
- 注意: しゃがめないため床付近の掌 z 下限は ≈0.33。margin 過小は `not_detected` 増

**到達・接近・配置（`grasp_distance` からは導出しない）**
```
measured_floor_xy   = FK(床置き掴み姿勢) の base→palm XY
reach_radius        = measured_floor_xy × reach_factor          # 0.85
place_dist          = reach_radius - place_margin               # 0.10
grasp_approach_dist = min(reach_radius + δ, measured - 0.02)    # δ=0.05（実測腕長でクランプ）
stand_dist          = place_dist + grasp_approach_dist
```
- `FLOOR_REACH_JOINTS`: shoulder_pitch=-1.5, elbow=-0.2, pitch_wrist=0.3（palm z≈0.33, XY≈0.48）
- `reach_radius + δ` が実測腕長を超えると place/grasp 不能のため、`measured - 0.02` で上限クランプ

#### 実装・検証結果（コンテナ `d994f05625a7`）

代表値（現行 URDF・プリミティブ）:
- `grasp_distance≈0.30`, `measured_xy≈0.484`, `reach_radius≈0.412`, `place_dist≈0.312`, `grasp_approach_dist≈0.462`, `stand_dist≈0.773`

| 検証 | 結果 |
|------|------|
| `eval_grasp_primitives`（5 trial/形状） | 全形状 success_rate **1.0** |
| `eval_object_giving_place`（fixed/passive/reject） | **OVERALL PASS** |
| デモ mp4 | `results/3d_test/demo_object_giving_{fixed,passive}[_third].mp4`（目視確認は人間） |

### B. 発話テンプレ簡素化（旧1）— 高

escalate の声掛け・発話増（D/E）の前提。dig で方針確定・**実装済み**。

#### 方針（dig 確定）

**テンプレの形**
- 生音素タプルは廃止。テンプレは **概念ID列**（`CONCEPT_TO_PHONEMES` と整合）
- 複合発話は **2スロット**：機能語 → TARGET（例: `["praise", "TARGET"]`）
- 語順は全テンプレで **機能語先行**に統一（日本語表面順には縛らない。表示は `CONCEPT_TO_HUMAN_WORD`）
- 3スロット（「だね」等の閉じ語）や枠込み専用概念は採らない

**ラベリング・語彙追加**
- `declare_object` → `["declare", "TARGET"]`。語彙に **`declare`**（`["p6","p5","p0","p1","p7"]`，表示「これ」）を追記
- dyadic の `prompt` → `["prompt"]`。語彙に **`prompt`**（`["p4","p2","p2","p6","p2"]`，表示「ねえ」）を追記
- 既存コードは固定し、**追記のみ**（`optimize_vocabulary.py --append`。`--write` による全語再最適化はしない）

**単語間ギャップ**
- 単語境界で `none` を挿入（ソフト区切り）。学習観測向け。親声のセグメント分割は要求しない
- 定数 `INTER_WORD_GAP_FRAMES`（既定 **1**）。セグメント用 `offset_silence_frames` とは別物

**スコープ（実装時）**
- 含む: 現行 `queue_utterance` パス — `object_giving` / `joint_attention` / `dyadic`（テンプレ発話）
- 含めない: カタログ外（`object_play` / `dyadic_play` / `physical_contact`）・household 発話。テンプレエントリは削除
- dyadic 喃語 `_queue_phoneme_pattern` は C 側の「触らない」に合わせ今回対象外

**テンプレ対応表（正本）**

| キー | 新テンプレ | 備考 |
|------|-----------|------|
| `declare_object` | `["declare", "TARGET"]` | 語彙に `declare` 追記 |
| `declare_color` | **削除** | 未使用 |
| `present` | `["douzo", "TARGET"]` | 既存 `douzo` を流用 |
| `praise_with_target` | `["praise", "TARGET"]` | |
| `recast_with_target` | `["recast", "TARGET"]` | |
| `correct_with_target` | `["correct", "TARGET"]` | |
| `praise` / `recast` / `correct` | `["praise"]` 等 | 語彙の固定長5へ |
| `prompt` | `["prompt"]` | 語彙に `prompt` 追記 |
| `target_only` | `["TARGET"]` | 残す |
| 上記以外（play/household/…） | **削除** | スコープ外 |

**escalate への含意**
- 新声掛けは「機能語を語彙に追記 → `["新語", "TARGET"]`」で足せる

**併せて修正**
- JA informative の単語一致判定: `best_word == self.target_word`（日本語表示語）→ `best_word == c_id`（concept_id）。認識器の辞書キーと揃える

**触り先**
- `language/templates.py` / `manager.py` / `vocabulary.py`（+ HUMAN_WORD）
- `protocols/object_play.py` / `dyadic_play.py`（廃止テンプレ呼び出し削除）
- `protocols/joint_attention.py`（上記単語比較）

#### 実装・検証結果（コンテナ `d994f05625a7`）

| 検証 | 結果 |
|------|------|
| `test_language_templates`（7） | **OK**（2スロット+gap=61、duration統一、min_h≥4） |
| `unittest discover cueworld/tests` | 67 tests（`test_household_inspection` 1件は発話非関連の既存フレーク） |
| `eval_voice_recognition` | k=1 top1=0.99 / fa=0.00；silence/random accept=0 |
| デモ mp4 | `results/3d_test/demo_object_giving_fixed[_third].mp4`、`demo_ja_shake_fixed[_third].mp4`（目視確認は人間） |

### C. 親の1音素長の統一（旧2）— 高

B とセット（発話タイミング・テンプレ長の単位）。dig で方針確定・**実装済み**。

#### 方針（dig 確定）

**時間アンカー**
- **1単語 = モータ1周期（30ステップ）** → **1音素 = 6フレーム**
- 根拠: `MOTOR_CYCLE_STEPS=30`、語彙固定長5、`voice_plan` の「固定長5（30フレーム）」
- 2スロット発話長 ≈ `5×6 + gap + 5×6`（gap=1 なら 61）≈ JA looking/pointing 提示（60）に収まる

**定数の置き場所**
- 言語層で **独立定数**: `PHONEME_DURATION_FRAMES=6`（+ B の `INTER_WORD_GAP_FRAMES=1`）＠`language/manager.py`
- モータ周期から式で導出しない（`language/` が `dyadic.py` に依存しない）。対応関係はコメント／本メモに残す
- テンプレ側の per-phoneme duration フィールドは廃止し、manager が一括付与

**今回触らない**
- dyadic 喃語の「1音素=30・間15」（`_queue_phoneme_pattern`）。バブリング用の引き伸ばしとして別スケールのまま。dyadic 仕上げ時に見直す

### D. プロトコル内の親発話タイミング増（旧4）— 中高

B・C の後。E（escalate）と重なるため **E と一体で dig・方針確定・実装済み**。JA／dyadic の happy-path 発話密度は今サイクル対象外。

#### 方針 — E の発話節に包含

giving 内の追加／維持ポイントのみ（語彙・テンプレの新規追加はしない。既存 `present` を使う）:

| タイミング | 発話 | 備考 |
|------------|------|------|
| 初回 `approach`→`placing` | `present` | 現行どおり |
| `waiting_infant` **RETRY** | `present` | 現行どおり。維持必須 |
| `waiting_infant` **ESCALATE 判定時** | **なし** | RETRY で既に `present` 済みのため重ねない |
| ESCALATE 経路の **`showing` 進入** | `present` | 「見せながらどうぞ」 |
| ESCALATE 経路の再 `placing` 進入 | **新規 `present` しない** | `showing` でキュー済み。言い残しは `utterance_queue` が状態を跨いで継続再生 |
| 成功 `reaction` | `praise_with_target` | 現行どおり |

詳細・モータとの対応は **E** を正本とする。検証結果は E にまとめて記載。

### E. `object_giving` escalate 見直し（旧5）— 中高

**動画で確認（確認は人間）**: `results/3d_test/demo_object_giving_passive[_third].mp4`

現状の ESCALATE（`waiting_infant` で配置破棄→未把持のまま `approach`）は関与強化として薄い。dig で方針確定・**実装済み**。依存 A・B・C は実装済。

#### 方針

**スコープ**
- 含む: `waiting_infant` の **ESCALATE** 専用経路（再把持→掲げる→再 place）と、それに伴う D の発話配置
- 含む: `parent_tools.show_object` 新設
- 含めない: `fetching`／`approach`／`placing` の ESCALATE（RETRY と同処理のまま）
- 含めない: 初回ハッピーパスへの `showing` 挿入、JA／dyadic の発話増、語彙新規追加、F（shake 掲げ感）

**ESCALATE 状態列（失敗後のみ）**
```
waiting_infant (ESCALATE, 発話なし)
  → fetching → approach → showing → placing → waiting_infant
```
- 初回ハッピーパスは現状どおり: `fetching → approach → placing`（`showing` なし）
- 現行バグ寄り挙動（未把持のまま `approach`）をやめる
- `escalated` フラグは showing → placing で落とす（1回きり）
- `showing` 中に幼児が把持したら `reaction`（成功扱い）

**`show_object`（parent_tools）**
- 把持済み前提。**接近は含めない**（現行 `shake_object`／`place_object_at` と同型の薄いプリミティブ）
- 戻り: `(action, reached)`（`place_object_at` に合わせる）
- 手先ターゲット:
  - **XY**: 親子を結ぶ線上・幼児から `show_dist`（幼児寄り）
  - **Z**: 幼児 `eyes_link` の高さ
- 実行時の Z 段階下げフォールバックはしない。届くことは **定数調整＋動画／eval** で担保
- 視線はツールに埋め込まず、プロトコル側で **幼児へ `gaze_agent`**

**`show_dist`**
- **プロトコルを正**。初期値は `env.place_dist` 相当
- `getattr(env, "show_dist", None)` があれば上書き（デモ／評価注入用）。reach からの自動派生はしない（演出ノブ）

**`showing` の終了**
- 成功: ショー目標へ十分接近 **かつ** 最低 step 経過 → `placing`
- 言い残し: `utterance_queue` は `change_state` で消えないため、`placing` でも `_emit_utterance` が継続
- 例外: 時間切れでも届かなければ **到達扱いせず `placing` へ**（escalate 全体を落としない）。定常設計では届く前提

**発話（D と共通・再掲）**
- RETRY: `present`／ESCALATE 待ち: なし／`showing`: `present`／ESCALATE 再 `placing`: 新規 `present` 抑制（フラグ等）

**触り先（実装時）**
- `parent_tools.py`（`show_object`）
- `protocols/object_giving.py`（状態・発話・`show_dist`）
- 必要なら `protocols/object_giving_escalate.md` を現状→新仕様に更新
- 検証: 既存 `eval_object_giving_place` ＋ escalate 経路のデモ mp4（passive 等）。目視は人間

#### 実装・検証結果（コンテナ `d994f05625a7`）

| 検証 | 結果 |
|------|------|
| `eval_object_giving_place`（fixed/passive/reject） | **OVERALL PASS**（passive: `showing_seen=True`, `show_reached=True`, `held_during_showing=True`） |
| `unittest discover cueworld/tests` | 67 tests（`test_household_inspection` 1件は発話非関連の既存フレーク） |
| デモ mp4 | `results/3d_test/demo_object_giving_passive[_third].mp4`、`demo_object_giving_fixed[_third].mp4`（目視確認は人間。fixed は showing 非経由で completed） |
### H. `show_object` 目標点の固定（ESCALATE 目視）— 中・実装済

**動画で確認（確認は人間）**: `demo_object_giving.py --infant_pattern passive` → `results/3d_test/demo_object_giving_passive[_third].mp4`

確認した結果
- `show_object` のバグ: 毎ステップ `eyes_link` を目標にすると、目が動くたびに手が不安定になる（`hand_reach_location` は目標 1cm 超で IK リセット）

#### 方針（実装済み）

**共通原則（H / F / G）**
- 目標点の**寿命はプロトコル**、計算・到達のプリミティブはツール（`place_pos` / `place_object_at` と同型）
- **ライブの `eyes_link` 追従はしない**（計算瞬間の Z をスナップショット）
- ツール内に掲げ／振りのステートマシンや目標キャッシュを持たせない（再進入でフェーズが腐るため撤回）

**`compute_show_position`（parent_tools・新設）**
- XY: 現行どおり親子線上・幼児から `show_dist`
- Z: **計算時**の幼児 `eyes_link` 高さ（頭とほぼ同高。`head_link` 追加はしない）

**プロトコル（`object_giving`）**
- `approach → showing` 進入時に毎回 `show_pos = compute_show_position(...)` を新規計算
- `showing` 退出時（`placing` / `reaction` / `fetching` / 失敗）に `show_pos = None`
- 落下→再 `showing` も進入時に再計算（親子位置が変わっているため）
- 毎ステップ `show_object(..., target_pos=show_pos)`（固定点へ到達するだけ）

**`show_object`**
- 渡された固定 `target_pos` へ `hand_reach_location`＋把持維持。毎ステップ自ら目を読まない

**触り先**
- `parent_tools.py`（`compute_show_position` / `palm_reach_error` / `show_object` 改修）
- `protocols/object_giving.py`（`show_pos` 寿命）
- 検証: `eval_object_giving_place`（passive）＋デモ mp4。目視は人間

**実装順**: **H → F → G**（掲げ点ヘルパが共有土台）

#### 実装・検証結果（コンテナ `d994f05625a7`）

| 検証 | 結果 |
|------|------|
| `eval_object_giving_place`（fixed/passive/reject） | **OVERALL PASS**（passive: `showing_seen=True`, `show_reached=True`, `held_during_showing=True`） |
| デモ mp4 | `results/3d_test/demo_object_giving_passive[_third].mp4`（目視確認は人間） |


#### 結果
- 手の動きは良い．
- 一方で，頭の動きが不安定

### F. shake 動作の掲げ感（旧6）— 中・実装済

**動画で確認（確認は人間）**

- `parent_tools` の物体を振る動作が正しいか
- shake はもう少し掲げる感じのほうがよいのでは
- **JA 仕上げサイクルで触る**
- デモ: `demo_ja_shake.py --infant_pattern fixed --views infant,third`

確認した結果
- もう少し掲げるべき．　幼児エージェントの頭の高さくらい．(show objectを参考にする．)

#### 方針（実装済み）※ H 依存

**スコープ**
- 含む: JA `cue_type="shake"` の `shaking` で、H と同じ掲げ点へ手を上げてから振る
- 含む: 届くよう **`shake_stand_dist` を `stand_dist` 相当へ下げる**（`show_dist≈place_dist` のまま接近 0.9 だと腕長外になりやすい）
- 含めない: `shake_object` 内の raise→shake ステートマシン（旧 archive 型は採らない）

**プロトコル（`joint_attention`）**
- `shaking` 進入時に `show_pos = compute_show_position`（giving と同じ `show_dist` 系）を固定。退出で破棄
- 毎ステップ: 固定点へ IK 維持＋手首振り（薄い合成。内部 SM なし）
- **到達後**に振り専用タイマを `shake_steps` 分回してから `checking`（到達待ちなしの現行タイマはやめる）
- `checking` ESCALATE → 再 `shaking` は進入時に `show_pos` 再計算（プロトコル寿命なのでフェーズ腐りなし）

**`shake_object`**
- 固定 `target_pos` 前提の薄いプリミティブ（IK 維持＋手首ピッチ）。`target_pos=None` は現行の手首振りのみ（既存 eval/テスト互換）。フェーズフラグは持たない

**触り先**
- `parent_tools.py`（`shake_object`）
- `protocols/joint_attention.py`（`show_pos` / 到達後タイマ / `shake_stand_dist`）
- 検証: `demo_ja_shake.py`（fixed）。目視は人間

#### 実装・検証結果（コンテナ `d994f05625a7`）

| 検証 | 結果 |
|------|------|
| `demo_ja_shake.py --infant_pattern fixed` | `final_state=completed`, `reached_shaking=True`, `reached_checking=True`, `shake_sound_firings=85` |
| デモ mp4 | `results/3d_test/demo_ja_shake_fixed[_third].mp4`（目視確認は人間） |

#### 結果
- 頭の動きが不安定
### G. dyadic 模倣の手の動き拡大（旧7）— 中・実装済

**動画で確認（確認は人間）**

- 模倣タスクにおける親の手の動きなどが小さすぎる。もう少し改良が必要
- **dyadic 仕上げサイクルで触る**
- デモ: `demo_dyadic.py --scenario arm_wave|hand_open_close --views infant,third`

確認した結果
- 手を目の高さくらいのところに持ってきてそこで左右に振るほうがいいのでは？(開け閉めも同様)

#### 方針（実装済み）

**スコープ**
- 含む: 親の `arm_wave` / `hand_open_close` を「幼児目線高さ付近へ掲げてから」ジェスチャ
- 含めない: 幼児デモ用モータの見た目変更（デフォルト動作は現行維持）
- 含めない: `head_nod` の掲げ IK（腕と無関係）

**`gesture_pos`（プロトコルが保持）**
- `imitating` / `performing_parent_cue` 進入時に計算・保持。`_reset_motor_state`／退出で破棄
- Z: その瞬間の幼児 `eyes_link`
- XY: **親 torso 前方の固定距離**（giving の `compute_show_position` は流用しない。二項は自前の見せ位置）
- 前方距離の具体値は実装＋動画で調整（演出ノブ `gesture_forward_dist=0.35`）

**毎ステップ（親・腕チャネル時）**
1. `hand_reach_location(gesture_pos)` で掲げを維持
2. 続けてオーバーレイのみ:
   - `wave_arm(..., gesture=True)` → **`shoulder_yaw` のみ**左右往復（他は IK のまま）
   - `animate_hand_open_close(..., gesture=True)` → **指のみ**開閉
3. 引数なし（デフォルト）は現行の関節角ポーズのまま → **幼児デモ・既存テスト互換**

**タイマ**
- `arm_wave` / `hand_open_close` の duration は **掲げ到達後**のみ進行（F の shake と同型）。タイムアウトで到達扱い
- `head_nod` / voice は従来どおり進入から進行

**触り先**
- `parent_tools.py`（`gesture=True` オプション）
- `protocols/dyadic.py`（`gesture_pos`・先に IK・到達後 timer）
- 検証: `demo_dyadic.py`（`arm_wave` / `hand_open_close`）。目視は人間

#### 実装・検証結果（コンテナ `d994f05625a7`）

| 検証 | 結果 |
|------|------|
| `demo_dyadic.py --scenario arm_wave` | `final_state=completed` |
| `demo_dyadic.py --scenario hand_open_close` | `final_state=completed` |
| `unittest discover cueworld/tests` | **70 tests OK** |
| デモ mp4 | `results/3d_test/demo_dyadic_{arm_wave,hand_open_close}[_third].mp4`（目視確認は人間） |


### 後で（今は触れない）優先度普通

- オブジェクトサウンドなどが同時に発生したらどうなるか（`base.py`）

### 後で（今は触れない）優先度低い

- プロトコルごとのスケールについての調整（集計時にその大きさに応じて加算。スケールは理想的なステップ数で行った場合を考える）


## 追記(2回目)
- タイムアウト関連の見直し
- action_repeat=1で使うことを前提になっているか確認して，そうでないならそうなるように調整．
- 対象が動いている場合に注意が成功できない(ja_shakeなどで，目的のオブジェクトが動く)
- dyadicで，幼児の腕や手の動きを真似るとき，幼児の腕や手の位置に関わらず真似るわけだけど，それはおかしい気がする．親にとってサインになるには目線くらいの高さに幼児が手を持ってきている必要があるのでは？(もう少し手の位置の観点で顕著なものだけを真似るべき)
- 幼児が行った動きを親が真似るのはあるけど，幼児が親を真似るのはできる？これも実装しよう．デモも作ろう．幼児の動きがどれくらいズレていても許容されるのかも定量的にチェックしたい．
- 共同注意のシナリオで，幼児が始動するパターンはある？これのデモも作ろう．
- 体の関節の1stepの速さをどのくらいにするかをちゃんと決めたい

### 追記(2回目) — 実装順・まとまり（計画準備）

項目をそのまま並列に潰すと、**時間単位（1 env.step が物理的に何を意味するか）が動くたびにタイムアウト・到達待ち・デモ duration を何度もやり直す**ことになる。先に時間軸を固定し、その後プロトコル別の振る舞いを JA / dyadic の2トラックで進める。

#### 項目の仮ラベル

| ID | 内容 | 主座（触りそうなところ） |
|----|------|--------------------------|
| **T0** | `action_repeat=1` 前提の確認・統一 | `base.py`、各テスト／デモの `action_repeat=`、`infant_signal.dt` |
| **T1** | 関節 1step 速度の決定 | `base.py` の joint `step_size` / mappings |
| **T2** | タイムアウト見直し | 各 `protocols/*.py` の `state_timer > N`、`*_timeout_steps`、duration |
| **J1** | 動く対象でも注意成功（ja_shake 等） | `joint_attention` の checking／視線ターゲット判定、`infant_signal` の gaze target |
| **J2** | JA 幼児始動パターン＋デモ | `joint_attention` の `infant_initiative` 分岐（既存 TODO）、デモスクリプト |
| **D1** | dyadic: 手の高さなど位置で salience を絞る | `DyadicProtocol.get_salient_channels`、必要なら `infant_signal` |
| **D2** | 幼児が親を真似る＋デモ＋許容ズレの定量 | `dyadic` 親始動側（`waiting_infant_response` 等）、デモ、eval |

#### 依存関係（なぜこの順か）

```
T0 (AR=1) ──┬── 一緒に決める ──┐
T1 (関節速) ─┘                   │
                                 ▼
                         T2 (タイムアウト再較正)
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                      ▼
        JA トラック                            dyadic トラック
         J1 → J2                                D1 → D2
```

- **T0 と T1 はセット**: `env.step` 1回あたりの実効変位＝`action_repeat × step_size`。片方だけ直すと、既存の `shake_steps` / `MOTOR_CYCLE_STEPS=30` / 掲げ到達タイムアウト（dyadic の AR=1 前提コメント含む）が全部ズレる。
- **T2 は T0・T1 の後**: タイムアウトは「何 step 待つか」の定数群。時間単位が固まってから横断で見直す。T0 調査の途中で「AR=4 時代のマジックナンバー」が出ても、**定数の本書き換えは T2 に寄せる**（調査メモだけ T0 に残す）。
- **J1 → J2**: 動く物体で checking が落ちる問題を残したまま幼児始動デモを足すと、成功判定が曖昧なままデモ検証になる。J1 はバグ寄り・局所、J2 はオートマトン拡張。
- **D1 → D2**: 「何をサインとみなすか」（高さゲート）を先に決めないと、D2 の「どれだけズレていても許容するか」の定量基準がぶれる。親始動の応答マッチも同じ saliency／位置前提を共有しうる。
- **JA トラックと dyadic トラックは T2 以降なら並行可**（触るファイルがほぼ分かれる）。ただし時間定数を再度いじる変更は避ける。

#### 推奨バッチ（一緒に実装する単位）

| バッチ | 含む | 一緒にする理由 | 複雑さ目安 | 備考 |
|--------|------|----------------|------------|------|
| **① 時間土台** | T0＋T1 | 実効速度の定義が同一。監査→方針→パラメータ確定まで一気に | 中 | **方針 dig 確定・実装待ち**（下記）。household 等の AR=4 残存はゲート後に揃える |
| **② 時間定数横断** | T2 | ①の数値に合わせてプロトコル横断で揃える作業 | 中 | **方針 dig 確定・実装待ち**（下記）。①ゲート後。計測→`k=2×T_task`。①と同一 PR にしない |
| **③a JA 注意安定** | J1 | 単独で閉じるバグ修正＋必要ならシグナル側 | 低〜中 | **方針 dig 確定・実装待ち**（下記）。`demo_ja_shake` で回帰 |
| **③b dyadic 顕著性** | D1 | ③a と並行可。salience ゲートのみ | 低〜中 | **方針 dig 確定・実装待ち**（下記）。G 済みの上に幼児側掲げ条件。PM 強度反映は D1 後続 |
| **④a JA 幼児始動** | J2 | ③a の後。デモ新設 | 中 | `infant_initiative` でオートマトン分岐（コード上 TODO あり） |
| **④b 幼児→親模倣** | D2 | ③b の後。デモ＋定量 eval | 中〜高 | **方針 dig 確定・実装待ち**（下記）。実行時は現行のゆるい反応成功のまま。B 層 match は注釈／eval |

```
① T0+T1（AR=1 と関節速度）
    ↓
② T2（タイムアウト見直し）
    ↓
├─ ③a J1（動く対象の注意） → ④a J2（JA 幼児始動＋デモ）
└─ ③b D1（手高さ salience） → ④b D2（幼児が親を真似る＋デモ＋定量）
```

#### 計画 dig に持ち込む論点

- **① T0+T1**: **方針確定**（下記「バッチ① 方針」）。実装待ち
- **② T2**: **方針確定**（下記「バッチ② 方針」）。実装は①ゲート後。計測→`T_task`→定数反映の順
- **J1**: **方針確定**（下記「バッチ③a J1 方針」）。実装待ち
- **D1**: **方針確定**（下記「バッチ③b D1 方針」）。実装待ち。PM dyadic 強度へのマージン反映は D1 後続
- **D2**: **方針確定**（下記「バッチ④b D2 方針」）。実装は③b（D1）後。τ は動画目視後に設定
- **J2**: **方針確定**（下記「バッチ④a J2 方針」）。実装待ち

#### 今はやらない（追記2回目の外）

- memo 末尾の「後で」項目（object_sound 同時発生、プロトコルスケール）は引き続き後回し
- §6 遷移行列は、上記バッチが一段落してからのまま

### バッチ①（T0+T1）方針 

#### スコープ
- 含む: `action_repeat=1` への統一、`BaseEnv.action_dt=0.1`、`infant_signal.dt` の接続、親子 `*_step` の意味付け（現行値を初期値として維持）、起動しうるデモ／`cueworld/tests` の AR=1 化
- 含めない（T2）: プロトコル横断の `state_timer` / `*_timeout_steps` / duration の本較正
- 例外: コアデモが AR=1 化だけで即死する場合の **時間定数の最小パッチ** のみ T0/T1 で可
- 廃れたデモ: 無理にメンテしない（触るなら AR=1。使わないなら archive 扱い）

#### 時間の意味（T0 / T1 共通）
- **1 `env.step` ≒ 人間の約 0.1 秒分の動き**（行動粒度 ≈ 10Hz）
- PyBullet の `1/240` s は接触・落下用の **内部刻み**。行動ティックと一致させない
- 正の定数: **`env.action_dt = 0.1`**（物理 timestep から導出しない）
- `infant_signal.dt` は `env.action_dt` を読む（現行の `1/240 * action_repeat` はやめる）

#### T0 — `action_repeat`
- **全面 AR=1**（現行カタログ経路・起動しうるデモ・テスト）
- **`action_repeat != 1` は `ValueError`**（実験用の可変 AR は今は持たない）
- 正とする世界は「現行 AR=1 仕上げ」だが、**先にゲートで確認してから**正と確定する

**ゲート（確認してから正とする）**
- `demo_object_giving`（fixed / passive）
- `demo_ja_shake`（fixed）
- `demo_dyadic`（arm_wave / hand_open_close）
- 上記が AR=1 で完走＋目視 OK
- `unittest discover cueworld/tests` が通る（household 等の AR=4 残は **ゲート後**に AR=1 化して再実行）

#### T1 — 関節 1step（`*_step`）
- **親** = 成人の 0.1s 相当を正。**現行 `parent_*_step` デフォルトを初期値**とする
- **幼児** = 別表で小さめ（分離維持。現行幼児デフォルトを出発点）
- 大きな再設計はしない。**明らかにおかしい関節だけ後で触る**
- シグナル側の速度スケール（`hand_movement_size` の `/2.0`、`vmax=1.0` 等）も当面そのまま。`dt=0.1` 化で壊れたらだけ直す

#### 触り先（実装時）
- `base.py`（`action_repeat` 拒否、`action_dt`、必要なら `*_step` コメント整備）
- `infant_signal.py`（`dt ← env.action_dt`）
- AR=4 残存: `cueworld/tests`（household 等）、起動しうる `cueworld/demo/*`
- 検証: ゲートデモ ＋ unittest（コンテナ `d994f05625a7`）

#### 実装順（①内）
1. `action_dt` 追加 ＋ `infant_signal.dt` 接続 ＋ AR≠1 拒否
2. 起動しうるデモ／テストを AR=1 に揃える
3. ゲート実行（コア3デモ目視＋unittest）→ 問題があれば最小パッチ（即死タイムアウト or 明らかにおかしい step / スケール）
4. household 等の残りを AR=1 化して unittest 再緑

### バッチ②（T2）方針

前提: バッチ①どおり **`1 env.step ≒ 0.1s`（`action_dt=0.1`）**。コード上の閾値は **すべて step 数**で持つ。

**実秒について**: 設計・較正は **step／計測した課題時間（`T_task`）** で行う。実秒換算を設計の主定規にはしない（あまり必要がない）。ただし後で参照する可能性があるので、**実装時は各 timeout 定数のコメントに `≒ Xs`（`steps × action_dt`）を書いておく**。

#### スコープ
- 含む（T2 本対象）: **失敗に倒す／進行を強制する上限** — 明示的な `*_timeout_steps`、および `state_timer > N` のマジックナンバー（到達待ち・幼児応答待ち・親モータの張り付き防止）
- 含めない: 成功に必要な **最低表示・動作時間**（`show_min_steps`、`shake_steps`、motor `duration_steps`、`release_settle_steps`、`gaze_match_timer` 下限など）
- 初回対象プロトコル: **object_giving / joint_attention / dyadic**（household 等は後回し）
- 依存: **①（T0+T1）ゲート後**に本較正。①中の即死パッチは例外として可

#### 三種の型（要不要と発火結果）

| 型 | いつ付けるか | 発火時 | API |
|----|--------------|--------|-----|
| **失敗系** | **幼児の行動待ち**がある状態 | `handle_timeout` → RETRY / ESCALATE / CHANGE_TARGET / FAILED | `handle_timeout` 専用 |
| **エラー対策** | **親のみ**で決まり、失敗すると **進行が止まる**状態 | **即 `FAILED`**（RETRY/ESCALATE/CHANGE_TARGET なし） | **`handle_timeout` を使わない**別経路（例: `fail_on_error_timeout()`） |
| **前進系** | **親のみ**で、未達でもシナリオ上は進めてよい到達待ち | **強制的に次状態へ／到達扱い**（失敗にしない） | 既存の到達 TO 分岐を維持・整理 |

`failure_transition.md` 優先度4の「親のみの行動ではエスカレーション不要」と整合。現行 giving の fetch/approach/placing に刺さっている `handle_timeout` は **失敗系として不適切** → エラー対策へ移す。

#### 失敗系（幼児待ち）

**状態一覧（これだけ）**
- giving: `waiting_infant`
- JA: `checking`, `listening`
- dyadic: `waiting_infant`, `waiting_infant_response`

**段間隔の役割分け**
- **短待ち**: JA `checking`（cue 直後の反応ゲート）
- **本待ち**: 残り4状態
- 数値は共通秒固定ではなく、まず **`timeout_steps = 2 × ceil(max(中央値, 平均))`**（方針 B）。無理なら共通フロア付き（方針 C）へフォールバック
- 理想スタートで計測 T が潰れる状態はフロア T を置く（`checking` / `waiting_infant_response`）

**倍率**: 初回は **`k=2`**

**`T_task` の決め方（計測）**
- 成功エピソードのみ。失敗系に入ってから成功条件成立までの滞在 step（`state_timer` または step 差）
- seed **3本**の **中央値と平均の大きい方**（`ceil`）→ `T_task`
- 計測中の当該 timeout は無効化せず **≈800**（ハング終了用）
- 仮置き表は使わず、計測値で確定する（理想スタートで過小な場合はフロア）

**初回計測マトリクス（4ラン種 × seed 3）**
| 測る状態 | ラン |
|----------|------|
| giving `waiting_infant` | `demo_object_giving --infant_pattern fixed` |
| JA `checking` + `listening` | `demo_ja_shake --infant_pattern fixed`（`intent_type=simple`） |
| dyadic `waiting_infant` | `demo_dyadic` 幼児始動（例: `arm_wave`） |
| dyadic `waiting_infant_response` | `demo_dyadic --scenario parent_cue` |

- JA listening の `T_task` は **simple 用**。informative/sharing は後回し（当面同じ timeout 流用可）

**`max_retries`**
- 対象あり（giving / JA）: **`max_retries=3`**（RETRY→ESCALATE→CHANGE_TARGET→FAILED）
- dyadic（対象なし）: **`max_retries=2`**（RETRY→ESCALATE→FAILED。CHANGE_TARGET に到達しない）
- 注意: 現行 `handle_timeout` は `retry_count > max_retries` を先に見るため、`max_retries=2` では CHANGE_TARGET に到達しない

#### エラー対策（親・進行阻害 → 即 FAILED）

**付ける状態**
- giving: `fetching`, `approach`, `placing`（置き前の張り付き）、手放し後の **到達圏外**（イベント。滞在計測対象外、発火時即 FAILED）
- JA: `fetching`, `approach`（**shake / 非shake 両方**。非shake のコメントアウト TO はエラー対策として復活）

**発火**: 即 `failure_state` + finish。社会的 ESCALATE（showing 等）はここから起動しない

#### 前進系（親到達 → 強制次へ）

**状態**
- giving `showing`（届かなくても `placing`）
- JA `shaking` 到達待ち（届かなくても `checking`）
- dyadic `gesture_reach`（届かなくても到達扱い）

**発火**: 失敗にせず次へ。タイムアウト **あり**（なしだと未達で永久待ちになりうる）

#### 数値の定規（三種共通）

- **`timeout_steps = 2 × ceil(max(中央値, 平均))`**（step 基準。実秒で予算を決めない）
- コード上は step。コメントに **`T`（med/mean）・`k=2`・`≒ Xs`（`× action_dt`）** を残す（実秒は設計必須ではなく、後参照用）
- 理想スタートで T が潰れる状態（JA `checking`、dyadic `waiting_infant_response`）は計測値を信じず **フロア T** を置く
- 名前付き `*_timeout_steps` に寄せ、インライン `state_timer > 800` を減らす

**親側（エラー対策・前進系）の計測**
- 失敗系と同じコアデモに **相乗り**（追加デモなし）。seed 3・`T=ceil(max(med,mean))`・`k=2`
- 計測中の親側安全上限は **≈300**（親行動は短時間の想定。失敗系計測用の ≈800 とは別）
- giving fixed → fetch/approach/placing/showing 滞在
- JA fixed → fetch/approach/shake 到達
- dyadic 腕シナリオ or `parent_cue` → gesture 到達

**フォールバック**: `k=2` で誤 escalate／即死が多い → 方針 C（共通フロア）または `T_task` 見直し

#### 触り先（実装時）
- `protocols/base.py`（エラー即 FAIL の共通ヘルパ。`handle_timeout` は失敗系専用と明記）
- `protocols/object_giving.py`（fetch/approach/placing をエラー対策へ、`waiting_infant` のみ失敗系）
- `protocols/joint_attention.py`（checking/listening 失敗系、fetch/approach エラー対策、shake 到達は前進系）
- `protocols/dyadic.py`（幼児待ちは失敗系、gesture 到達は前進系）
- 計測: コア3デモに状態滞在のログ／サマリを足す（または薄い計測ラッパ）
- 検証: コンテナ `d994f05625a7`、fixed/cooperative の成功＋passive での失敗系 escalate、親ハングでのエラー即 FAIL

#### 実装順（②内）
1. 型の分離実装（fetch 等から `handle_timeout` 除去、エラー別経路、前進系は強制遷移のまま整理）
2. 計測用に一時的に大きめ TO（失敗系≈800、親側≈300）＋滞在ログ
3. 4ラン種 × seed 3 で中央値・平均取得 → `T=ceil(max(med,mean))` → `timeout_steps=2T`
4. `max_retries` を失敗系方針どおり揃え、ゲートデモ＋unittest
5. 厳しければフロア（方針 C）または個別 `T_task` 再計測

#### T2 計測結果（seeds 42/43/44、成功エピソード、`T=ceil(max(med,mean))`、`k=2`）

ログ: `results/t2_dwell_measure.log`

| 定数 | med / mean | T | timeout=2T | 備考 |
|------|------------|---|------------|------|
| giving `fetch` | 58 / 53.3 | 58 | 116 | |
| giving `approach` | 17 / 20 | **26** | **52** | pure なら T=20→40だが escalate 再接近(~43)対策で max_sample も採用 |
| giving `placing` | 42 / 33.3 | 42 | 84 | |
| giving `waiting_infant` | 15 / 16.7 | 17 | 34 | |
| giving `show_timeout` | (12) | 12 | 24 | fixed に showing 無し → JA `shake_reach` 流用 |
| JA `fetch` / `approach` | 同上 | 58 / 26 | 116 / 52 | |
| JA `shake_reach` | 12 | 12 | 24 | seed42 のみ実到達 |
| JA `checking` | 1 / 1 | **30** | **60** | 理想視線スタートで計測過小 → フロア T=30 |
| JA `listening` | 50 / 58.3 | 59 | 118 | |
| dyadic `waiting_infant` | 28 / 28 | 28 | 56 | |
| dyadic `waiting_infant_response` | 1 / 1 | **28** | **56** | 即応答の理想スタートで過小 → `waiting_infant` と同フロア |
| dyadic `gesture_reach` | 10 / 10 | 10 | 20 | |

### バッチ③a（J1）方針 — 動く対象でも注意成功

dig 確定（2026-08）。前提: T2 ゲート後でも **J1 単独 PR で可**（タイムアウト定数は触らない）。

#### 問題

- `ja_shake` 等で親が対象を把持・振ると、物体が動き視線レイが **親の腕／手／胴** に当たりやすい
- `checking` / `listening` の成功条件 `infant_target == target_id` が落ちる（`gaze_targets` の argmax が親 body になる）
- 現行の `get_infant_target` remap（親 body → 対象 id）は **顔注視も成功** しうる
- 現行の `detect_rejection` 把持例外（`parent_holding_target` で視線拒否スキップ）は分布ノイズ対策だが、本質修正の代わりにはならない

#### スコープ

- **含む**: 親または幼児が **対象を把持中**、視線が腕〜手に遮られても `infant_target == target_id` になること
- **含まない**: 未把持の移動体（転がる球等）。必要なら別 ID
- **含まない**: 振り中（`shaking`）の成功判定変更。判定は従来どおり `checking` / `listening` 以降

#### 成功条件の意味（ハイブリッド C）

- **通常**: 視線レイが対象メッシュに当たる幾何ヒットのみ
- **把持中のみ**: 同一視を許可 — 視線が **把持側エージェントの腕〜手クラスタ** に当たっても、対象を見ているとみなす
- **プロトコル側 remap は廃止**（`joint_attention.get_infant_target` の親 body 読み替えを削除）。`gaze_targets` の分布だけを信じる

#### 同一視で許す部位（`hold_gaze_links`）

親・幼児 **対称**。エージェント辞書（`base.py`）に明示定義する。

| 含む | 除外 |
|------|------|
| `upper_arm`, `lower_arm`, `elbow`, wrist, `hand`/`palm`/指/`grasp_link`/`point_link`（あれば） | `shoulder*`, `face_links`（目・頭・唇）, 胴・base |

- **幼児が物を持っているとき**も、幼児側の `hold_gaze_links` ヒットを **`infant["held_obj"]` スロットへ振替**（親把持と同型）
- 振替は **シグナル側**（`infant_signal._get_targets_by_cone_raycast` 後処理）で行う

#### クレジット振替の合成規則（D）

把持中（`agent["held_obj"] == oid`）かつ、レイがその agent の `hold_gaze_links` にヒットしたとき:

1. 許可リンクの精度 → **`held_obj` スロットへ移す**（親／幼児 body スロットには入れない）
2. 対象メッシュへの直ヒット → 対象スロットのまま
3. 最終: `gaze_targets[held_obj] = max(直ヒット, 振替分)`
4. **`face_links` ヒットだけ** 親／幼児 body スロットに残す（checking 中の「顔だけ見る＝成功」を防ぐ）

#### 視線拒否（把持例外の代替）

- **まず把持例外を削除**して 2 段拒否を試す
- **1 段目（候補）**: 現行と同型 — `total_valid_shift < -0.95` かつ `late_valid < 0.15`
- **2 段目（確定）**: 候補に加え **`avoidance_intensity > τ`** または **`head_turn_away_intensity > τ`**
  - `head_turn_away_intensity`: 幼児 **行動コマンド履歴**（関節角ではなく action）から同符号持続を測る
  - **水平回転**: `torso_joint` + `head_pan_joint` を **1 チャネルとして合算**（水平方向の回転として扱う）
  - **垂直回転**: `head_tilt_joint` を別チャネル
  - いずれかが直近で同符号に偏っていれば ON（閾値・同符号率は seed 42–44 で計測）
- **フォールバック**: 2 段拒否で **`demo_ja_shake fixed` が誤拒否** または **`reject` が拒否検知できない** 場合、**把持例外（`parent_holding_target` で視線拒否スキップ）に戻す**。remap 復活はしない

#### 触り先（実装時）

| ファイル | 変更 |
|----------|------|
| `cueworld/base.py` | 親・幼児に `hold_gaze_links` を定義 |
| `cueworld/infant_signal.py` | クレジット振替、幼児行動履歴、`head_turn_away_intensity` |
| `cueworld/protocols/joint_attention.py` | `get_infant_target` remap 削除 |
| `cueworld/protocols/base.py` | 把持例外削除 → 2 段拒否（ダメなら例外復帰） |
| テスト | 下記ゲート |

#### 受け入れ基準（ゲート）

- `demo_ja_shake --infant_pattern fixed`: `reached_checking=True`, `final_state=completed`, 誤拒否なし
- `demo_ja_shake --infant_pattern reject`: 拒否検知または `failure`
- `demo_ja_shake --infant_pattern passive`: 従来どおり失敗系 escalate（回帰）
- object_giving 等、幼児把持シナリオでも `held_obj` スロットが安定すること（該当デモがあれば目視）
- コンテナ `d994f05625a7`、既存 unittest 再緑

#### 実装順（③a 内）

1. `hold_gaze_links` 定義（親・幼児対称）
2. シグナル側クレジット振替（親把持・**幼児把持**の両方）
3. JA remap 削除
4. 2 段拒否＋行動履歴。ゲートで誤検知があれば **把持例外に戻す**
5. 閾値（τ, 同符号率等）を seed 42–44 で計測して確定


### バッチ③b（D1）方針 — dyadic 手高さ salience

dig 確定（2026-08）。前提: T2 ゲート後でも **③a（J1）と並行可**。タイムアウト定数は触らない。G（親の掲げジェスチャ）済みの上に、幼児側の「掲げ」をサイン条件にする。

#### 問題

- 幼児始動 dyadic で、親は `get_salient_channels` の周期性・動き大きさだけで `arm_wave` / `hand_open_close` を選ぶ
- 手が低くても真似するため、親にとってのサイン（目線付近への掲げ）になっていない
- 親側 G は幼児目線高さへ掲げてからジェスチャするが、幼児側の salience には対称の高さ条件がない

#### スコープ

- **含む**: `arm_wave` / `hand_open_close` の salience への高さ hard gate、`infant_signal` の連続マージン、デモ正例の幼児掲げ、単体の負例テスト
- **含まない**: `voice` / `head_nod` への位置条件
- **含まない**: 前方（torso 前方・親見せ位置）条件 — 高さのみ
- **含まない（D1 本対象外・後続）**: `protocol_manager` の dyadic 起動強度へのマージン反映（下記「D1 後続 B」）
- **含まない（D2）**: `_detect_infant_response` の高さ条件、許容ズレの定量

#### ゲート（チャネル選択）

- **対象**: `arm_wave` / `hand_open_close` のみ
- **幾何**: 世界座標の下限のみ — `z_hand > z_eyes − tol`（頭上は OK）
- **時間**: 瞬間値ではなく、直近窓の手 Z **中央値**
- **強度**: hard gate — 帯外なら当該腕チャネルの salience を 0（選外）。soft 乗算にはしない
- **適用箇所**: `DyadicProtocol.get_salient_channels` のみ（起動判定は現行のまま）

#### シグナル（`infant_signal`）

客観量を正規化シグナルに出す（模倣解釈の hard gate はプロトコル側）。

- キー案: `hand_height_margin`
- 定義: `margin = clip((z_hand − (z_eyes − tol)) / tol, 0, 1)`
  - `z_hand`: 直近 **10 step** の手 Z **中央値**（世界座標。既存 `hand_positions` が torso 相対なら、比較用に eyes との差／世界 Z を別途取る）
  - `z_eyes`: 同時点の幼児 `eyes_link` Z
- **D1 hard gate**: `margin > 0`（＝ Q5 の下限条件と一致）
- 動きシグナル（`hand_movement_*` / `hand_open_close_periodicity`）自体は高さでマスクしない（`object_play` 等への波及を避ける）

#### 初期ノブ

| ノブ | 初期値 | 備考 |
|------|--------|------|
| 窓長 N | 10 step | `hand_movement_regularity` が 10 step 以上を見るのと揃える |
| 代表値 | 中央値 | 振りの上下フリッカー抑制 |
| `tol` | 0.15 m | 目線より 15cm 下まで許容。デモ目視でずれたらだけ動かす |

#### デモ

- **デモ専用ヘルパ**で幼児を目線高さ付近へ `hand_reach_location` 掲げ → `wave_arm` / `animate_hand_open_close(..., gesture=True)`
- 既定のモータ API（`gesture=False` 関節ポーズ）は維持（他用途を動かさない）
- 更新対象: `demo_dyadic` の `arm_wave` / `hand_open_close` / `voice_arm_wave` 正例
- デモに負例シナリオは作らない

#### 検証

- **単体負例**: `margin=0`（または未掲げ相当）なら腕チャネルが `get_salient_channels` で選ばれない
- **デモ正例**: 掲げありで `final_state=completed`（目視は人間）
- コンテナ `d994f05625a7`、既存 unittest 再緑

#### 触り先（実装時）

| ファイル | 変更 |
|----------|------|
| `cueworld/infant_signal.py` | `hand_height_margin`（窓中央値・eyes 比較） |
| `cueworld/protocols/dyadic.py` | `get_salient_channels` で腕チャネル hard gate |
| `cueworld/demo/demo_dyadic.py` | デモ専用掲げヘルパ＋正例シナリオ |
| `cueworld/tests/test_dyadic_signals.py` 等 | 負例・正例単体 |

#### 実装順（③b 内）

1. `hand_height_margin` を `infant_signal` に追加（窓 10・中央値・`tol=0.15`）
2. `get_salient_channels` で `arm_wave` / `hand_open_close` を `margin > 0` のときだけ残す
3. 単体テスト（負例必須）
4. `demo_dyadic` 掲げヘルパ＋正例更新 → ゲート実行

#### D1 後続 B（本バッチの直後〜別 PR 可）

- 同じ `hand_height_margin` を `protocol_manager` の dyadic 強度にも載せる
- 低い手振りでは dyadic が立ちにくくする（動きシグナルのグローバルマスクはしない）
- D1 本体（salience ゲート）とは PR を分けてよい


### バッチ④a（J2）方針 — JA 幼児始動＋デモ

dig 確定（2026-08）。前提: **J1 後**に着手。J1 で JA 後半の視線成功判定を安定化してから、`joint_attention` の `infant_initiative` 分岐とデモを足す。

#### 問題

- `JointAttentionProtocol` には `self.infant_initiative` があるが、`_manage_state()` では未使用で **親始動 JA と同一の流れしかない**
- 幼児始動 JA を別オートマトンで新設すると、既存の `approach / looking / checking / listening` と重複しやすい
- 一方で、幼児始動では **親が対象を提示する側ではなく、幼児の対象に追従して共有確認する側** になるので、発話と target 決定の解釈を一部変える必要がある
- さらに、**幼児の発話開始は `checking` 中にすでに起こりうる** ので、`checking` と `listening` を明示的に分ける意味が薄い

#### スコープ

- **含む**: `joint_attention.py` の **派生版フロー**としての幼児始動 JA、target 決定の小拡張、scripted デモ 1 本
- **含まない**: 親応答としての `shake`、本格的な `pointing` cue、別プロトコル新設、学習用 reward 再設計
- **含まない**: 自然発生寄りの adaptive デモ。初版デモは再現性優先

#### フロー方針（親始動 JA の派生版）

- **全面的新規オートマトンにはしない**
- 幼児始動でも、既存ステートを再利用して **`approach -> looking -> post_shared_attention`**（仮名）に入る
- **`checking` と `listening` は 1 状態に統合**する（J2 を機に、親始動 JA 側も同じ整理を許容）
- 解釈だけを変える:
  - `approach`: 親が幼児と対象共有しやすい位置へ入る
  - `looking`: 親が **幼児の示した対象** を見る
  - `post_shared_attention`: 共有成立の確認、幼児発話の待機、親 fallback 発話、賞賛/リキャストをまとめて扱う
- 初期スコープでは、親の応答は **gaze 中心**。`shake` は入れない。`pointing` は将来拡張候補

#### 成功条件

- 成功は **親が幼児の対象へ一定時間追従注視し、その間幼児も同じ対象を維持** すること
- まずは IJA の核である「幼児が始め、親が追従した」を成功条件にする
- `post_shared_attention` 内での発話完了を **成功条件の必須要件にはしない**

#### 発話方針

- 初版では **幼児の発話を基本的に待つ**
- 幼児始動では、親始動 JA のような即 `declare_object` は **行わない**
- 幼児の発話開始が共有確認中に起こりうるので、**共有確認と発話待機は同一状態内で扱う**
- `post_shared_attention` で一定時間待っても幼児側から発話がなければ、**親が `declare_object` を発話する fallback は可**
- その後の賞賛系応答は維持候補だが、主役はあくまで「追従して共有したこと」

#### ターゲット決定

- `select_target` / `get_infant_target` は **gaze のみから小拡張**する
- ルールは:
  1. **指差しが十分強く安定**していれば `hand_targets` を優先
  2. それ以外は `gaze_targets` を使う
- つまり **pointing は override、通常は gaze**
- 初版で常時重み付き融合は行わない

#### デモ方針

- `cueworld/demo/` に **scripted な J2 デモ**を新設する
- 初版デモはかなり固定的にする:
  1. 幼児が親を見る
  2. 幼児が対象を見る／指す
  3. 親がその対象へ追従
  4. 幼児が対象維持
- プロトコル自体は `set_fixed_protocol("joint_attention", context)` 系の既存流儀に乗せる
- 幼児行動の adaptive 化は後回し

#### 触り先（実装時）

| ファイル | 変更 |
|----------|------|
| `cueworld/protocols/joint_attention.py` | `infant_initiative` 分岐、`checking` / `listening` 統合、発話位置調整、成功条件の分岐 |
| `cueworld/protocols/base.py` | 必要なら `select_target()` の拡張（または JA 側 override） |
| `cueworld/infant_signal.py` | 既存 `hand_targets` / `point_intensity` / `hand_target_stability` の利用確認 |
| `cueworld/demo/*` | J2 scripted デモ新設 |

#### 受け入れ基準（ゲート）

- 幼児始動デモで、**幼児が示した対象へ親が追従し `completed` まで到達**する
- 幼児発話が共有確認中に始まっても、**同一後半状態のまま処理できること**
- 幼児発話が無い scripted パターンでは、後半状態で親 `declare_object` fallback を入れられること
- pointing を使う scripted パターンで、target 決定が gaze-only より安定すること
- コンテナ `d994f05625a7` で対象デモ／関連テスト確認

#### 実装前に残る小決め

- 実装確定（2026-08）:
  1. `checking` と `listening` は親始動 JA 側も含めて `post_shared_attention` に統合し、内部で `confirming` / `responding` の 2 フェーズを持たせる
  2. 「指差しが十分強く安定」は **`point_intensity > 0` かつ `hand_target_stability >= 0.6` かつ `gesture_targets` 最大質量 `>= 0.5`** とする
  3. 後半状態での `declare_object` fallback は **60 step** 待ってから 1 回だけ入れる
  4. fallback は **成功条件と独立**。成功はあくまで `shared_hold_steps=50` の追従共有維持で判定する


### バッチ④b（D2）方針 — 幼児→親模倣（デモ＋定量）

前提: **③b（D1）後**に着手。高さ salience ゲートが幼児始動側で効いたうえで、親始動の応答 match 注釈と強度格子デモ／eval を足す。

**実装確定（2026-08）**: 調査で旧 engagement（`arm_ext` / `orientation_to_parent`）が縮退していたため、実行時の反応検知も salience ベースに再定義した。peak 窓は `waiting_infant_response` 進入からプロトコル終了まで。チャネル別 peak dict → argmax。

#### 問題

- 親始動 dyadic（`performing_parent_cue` → `waiting_infant_response`）はあるが、旧 `_detect_infant_response` はバブリング／手の動き／腕伸展／orienting の **ゆるい OR** で、親 cue とのチャネル一致や強度は見ない
- しかも `arm_ext` は静止時 1.0、`orientation_to_parent` はデモ常時注視で常に真 → `waiting_infant_response` が 1 step で抜け、ピーク計測不能
- 追記(2回目)の「幼児が親を真似る＋許容ズレの定量」は、engagement 成功を壊さずに **よりよい成功（B 層）** として測りたい
- 既存 `demo_dyadic --scenario parent_cue` はランダム 2 要素 cue で格子評価に使えない

#### スコープ

- **含む**:
  - engagement を salience ベースに再定義（babbling、または max salience ≥ `SALIENCE_GLOBAL_THRESH`）→ praise → `completed`
  - B 層 `response_matched` は **注釈／eval のみ**（親の praise・状態分岐は変えない）
  - `compute_channel_salience` / peak match / `MATCH_QUANTITY_THRESHOLDS` を `DyadicProtocol` に置く
  - `context["parent_channels"]` で親 cue 固定（無ければ現行 `_sample_parent_channels()`。RETRY でも維持）
  - `parent_tools` の `wave_arm` / `nod_head` / `animate_hand_open_close` に `amplitude_scale`
  - デモ: モータ 3 チャネル × 強度 `{1.0, 0.5, 0.25}` の動画
  - 薄い定量: 同格子の CSV（engagement / peak_channel / peak_score / matched@仮τ）
- **含まない（今回）**:
  - 実行時の段階的 praise／`completed_matched` 等の状態分岐（将来 B/C）
  - voice 複合 cue、ランダム 2 要素 cue の格子
  - 高さの sweep（腕は D1 と同じ二値ゲートのみ）
  - ideal peak 相対での τ フィット、位相ロック
  - 誤チャネルを主表にした sweep

#### 成功の二層

| 層 | 意味 | 実行時 | 記録 |
|----|------|--------|------|
| **engagement** | いずれかのチャネル salience ≥ 0.2（または babbling） | `_detect_infant_response` → `completed` | `final_state=completed` |
| **matched（B 層）** | よりよい成功（模倣寄り） | **変えない** | `response_matched` 等 |

#### match 定義

1. `compute_channel_salience` で全チャネル salience を計算（腕は `hand_height_margin > 0` の hard gate 込み）
2. **チャネル別に peak を保持**し、その中で **argmax** → `peak_channel` / `peak_score`
3. `response_matched = 1` ⇔ `peak_channel ∈ active_channels` かつ `peak_score ≥ τ[peak_channel]`
4. 比較量は **salience スコアそのもの**（生シグナルの別閾値は当面使わない）
5. ピーク窓は `waiting_infant_response` 進入から **プロトコル終了まで**（acknowledging / completed 中も更新。RETRY でリセット）

#### 閾値 τ

- **絶対閾値**（ideal 相対割合では決めない）
- 当面の仮値: チャネル別キーで **すべて 0.2**（`SALIENCE_GLOBAL_THRESH` と同値）。仮である旨を定数コメントに明記
- 強度違いデモ動画を目視確認したあと、`MATCH_QUANTITY_THRESHOLDS` をチャネル別に書き換える
- 高さは sweep しない。腕チャネルは D1 どおり `hand_height_margin > 0` のみ

#### デモ

- 親 cue は **単一モータ固定**（`parent_arm_wave` / `parent_hand_open_close` / `parent_head_nod`）
- 幼児は応答待ち〜 completed で、対応モータを `amplitude_scale ∈ {1.0, 0.5, 0.25}` で再生（腕は掲げあり）
- annotation に `peak_channel` / `peak_score` / `response_matched` / `time_to_engagement`（仮τ）を出す
- 既存のランダム `parent_cue` scenario は残す（格子の主対象ではない。応答は掲げ+wave）

#### 定量 eval

- `scripts/experiment/eval_dyadic_response_match.py` で上記 9 格子を回し CSV 出力
- 列: channel, intensity, seed, engagement_success, time_to_engagement, peak_channel, peak_score, peak_* , response_matched, final_state, frames
- ゲートは「スクリプトが走り数表が出る」こと。τ の最終値は目視後

#### 触り先（実装時）

| ファイル | 変更 |
|----------|------|
| `cueworld/protocols/dyadic.py` | `compute_channel_salience`、engagement 再定義、`MATCH_QUANTITY_THRESHOLDS`、peak match、`context["parent_channels"]`、`debug_info` 注釈、timeout=30 |
| `cueworld/parent_tools.py` | `amplitude_scale`（既定 1.0。親の本番キューは既定のまま） |
| `cueworld/demo/demo_dyadic.py` | 親始動単一チャネル scenario、`--intensity`、annotation |
| `scripts/experiment/eval_dyadic_response_match.py` | 9 格子 CSV（新設） |
| `cueworld/tests/test_dyadic_protocol.py` 等 | match／parent_channels／ピーク／engagement 縮退回帰の単体 |

#### 編集ガイド（正本）

| 目的 | 触る場所 |
|------|----------|
| **強度しきい値 τ（目視後）** | **`cueworld/protocols/dyadic.py` の `MATCH_QUANTITY_THRESHOLDS`** |
| デモ実装・強度格子 | `cueworld/demo/demo_dyadic.py` |
| 振幅スケール API | `cueworld/parent_tools.py`（`amplitude_scale`） |
| 親 cue 固定 | `dyadic.py` の `start(context)` → `parent_channels` |
| match／ピーク | `DyadicProtocol` 内（praise・状態分岐は触らない） |
| CSV 再計測 | `scripts/experiment/eval_dyadic_response_match.py` |

**実行イメージ（コンテナ `d994f05625a7`）**

```bash
# 例: 親 arm_wave、幼児強度 0.5
python3 cueworld/demo/demo_dyadic.py --scenario parent_arm_wave --intensity 0.5 --views infant,third

# 9 格子 CSV
python3 scripts/experiment/eval_dyadic_response_match.py
```

動画: `results/3d_test/demo_dyadic_parent_<channel>_i<100|050|025>[_third].mp4`。
CSV: `results/3d_test/eval_dyadic_response_match.csv`。

#### 受け入れ基準（ゲート）— 実装後

- engagement 縮退が解消し、静止相当のシグナルでは `waiting_infant_response` を抜けないこと
- 強度 1.0 の正例で `peak_channel` が親 cue と一致し、仮 τ で `response_matched=1` になりうること
- 強度 0.25 では `peak_score` が下がり、matched あるいは engagement が落ちうることが CSV と動画で追えること
- 腕チャネルで未掲げ相当なら salience 経由で match しにくいこと（D1 ゲート）
- 既存 `parent_cue` が再較正後のタイムアウトで `completed` に到達すること
- コンテナ `d994f05625a7`、既存 dyadic 関連テスト再緑

#### 将来（今回やらない）

- matched 時だけ praise／発話を変える（実行時 B）
- `completed` / `completed_matched` の状態分岐（実行時 C）
- voice を含む cue の match、τ の ideal 相対キャリブレーション


## 追記(3回目)

D2 実装・9 格子 CSV／動画の確認後の気づき。チャネル間 salience 比較の不整合を、設計側から簡略化する。**方針は末尾「方針確定（2026-08 dig）」が正。** 上の暫定・作業仮説は経緯。

#### 問題

- `get_salient_channels` は上位 salience のチャネルを返す（argmax／上位 k）が、チャネルごとにスコアの出やすさ（スケール）が違うので、数値の横並び比較がフェアでない
- つらいのは「チャネルが多いこと」自体ではなく、**クロスチャネルの順位付け**である

#### 方針（暫定）

- **`head_nod` は捨てる**（チャネルからもデモ主対象からも外す）
- 残すチャネルは **3つ**: `arm_wave` / `hand_open_close` / `voice`
- チャネル同士の比較・argmax は行わない
- おのおの salience を取り、**各チャネル独自の閾値**を超えたら有効チャネルとして扱う（独立ゲート。複数同時に有効なら並行再生でよい）
- 手関連の salience 式は当面いまのまま（`arm_wave`＝size×regularity＋D1 高さゲート、`hand_open_close`＝開閉周期性＋同ゲート、`voice`＝現行）。大改修は先送り
- **親始動**（親が先に動いて幼児が応答）: 手の動きは今と同じパターンモータ（`wave_arm` / `animate_hand_open_close`）でよい
- **幼児始動**（親が幼児を真似る）: パターンモータではなく、幼児の肩からの関節角度をそのまま模倣する案あり（掲げ IK・match 定義との両立が要るので **別バッチ**）
- 実装順の目安: 先に「3チャネル＋独立閾値（比較なし・`head_nod` 削除）」→ その後に「幼児始動の関節角コピー」
- 偶然の同行動（幼児がもともと親 cue と同じ動きをしていた）の切り分けは、親が「幼児が今やっていないチャネル」に限定する／onset・ベースライン差分など。本追記のチャネル簡略化とは直交しうる

（下の「追加の問題」以降で作業仮説を更新。上は会話時点の暫定。）

#### 追加の問題（arm_wave の peak 潰れ）

D2 格子 CSV / `probe_arm_wave_intensity_timeseries.csv` で、**掲げ（姿勢の遷移）が `arm_wave` として採点**され、振りの強度ピークが床上げされている。

- `arm_wave` salience は `hand_movement_size × hand_movement_regularity`（D1 高さゲート付き）。掲げ中もゲートが開けば点が入る
- 親始動デモの応答窓は先に intensity 非依存の腕上げ → 弱い振り（I=0.5 / 0.25）は上げ相（約 0.36）を超えず、`peak_score` が同じ値に張り付く。I=1.0 の振りだけ上げを上回る（約 0.43）
- `time_to_engagement` は腕シナリオで強度によらず 15（高さゲート開放の瞬間）。実行時 engagement はすでに「手を挙げた」で決まっている
- `hand_open_close` の開閉 peak 自体は強度に追従する（1.00 / 0.82 / 0.41）が、同じ掲げが `peak_arm_wave=0.358` として混入する

カテゴリ誤り: 掲げは状態（高さ）、振りは周期。いまの式は前者の変位を後者のスコアにしている。

#### 改変案

1. **腕を挙げてキープする状態をチャネルにする**（`arm_raise`）。salience は高さ（`hand_height_margin`）側
2. **`head_nod` と `arm_wave` はチャネルから外す**
3. **`hand_open_close` は独立チャネルではなく、`arm_raise` 中の付加**（挙げているときに開閉していたら開閉が乗る）

親モータはすでに「掲げ IK → 周期オーバーレイ」なので、チャネル側をその階層に合わせる。

#### 検討

- `arm_raise` 本体化は、D1 の高さゲートを「ゲート」から「本体」に昇格させる話で、上のピーク潰れの直し方としても筋がよい。高さは変位ではないので、挙げ動作そのものがスコアを床上げしない
- `head_nod` 削除は本追記の暫定方針と同じ
- 開閉を `arm_raise` の付加にするのも、現行ゲート（開閉は `hand_raised` 必須）と親モータ（掲げたうえで指だけ）の言い直しに近い。argmax で wave と開閉を横並びにするより、「挙げている／挙げて開閉している」の階層の方が match が素直
- **`arm_wave` を消すのは一段大きい。** I=0.5/0.25 の潰れは式と peak 窓の問題で、振りというカテゴリ自体が無効とまでは言えない。消すと D2 の振り強度格子と「振る vs 開閉」対比が消える。幼児始動で関節角コピーを入れるなら振りはチャネルなしでも真似できる。パターンモータのままなら、幼児が振っても親は「挙げて止まる」だけになる
- 「キープ」をチャネルにするなら **peak-max は合わない**。一度挙げれば `hand_height_margin` の走行最大は下がらないので、「いま挙げ続けている」ではなく「一度挙げた」になる。キープなら窓内の持続（上げている時間や直近平均）の方が意味に合う
- voice は提案に出ていない。独立のまま（本追記の暫定どおり）

#### いまの作業仮説（dig 前。確定は下の「方針確定」）

比較・argmax はしない。独立ゲート。骨格:

```
voice                 … 独立
arm_raise             … 本体（高さ＝キープ）
  └ hand_open_close   … 付加（挙げているときだけ）
  └ arm_wave          … 同じ付加にするか、チャネルとしては捨てるか（未決）
head_nod              … 捨てる
```

match は階層: 親が `arm_raise` → 幼児が挙げていれば match。親が `arm_raise + open_close` → 挙げたうえで開閉が τ 以上なら match。開閉を付加にするなら、親 cue の原子も `hand_open_close` 単体ではなく常に `arm_raise` 付き（現行親モータはすでにそう）。

先に確定してよい範囲: **`arm_raise` 本体 + 開閉は付加 + `head_nod` 削除 + 比較なし**。`arm_wave` は「同じ付加にするか、幼児始動の関節コピーまでチャネルとしては持たないか」を次の分岐にする。

#### いまの含意（dig 前）

- D2（④b）のゲート・仮 τ 目視とは別枠。**dyadic チャネル設計の見直し**
- 「手1つに潰す」（wave と開閉を同一チャネルにする）は不採用。開閉は付加として残す
- 上の「方針（暫定）」の「残すチャネル 3つ: `arm_wave` / `hand_open_close` / `voice`」は破棄。確定は次節

#### 方針

チャネル簡略化＋`arm_raise` 本体化。D2 の仮 τ 目視とは別枠。**バッチ化してよい。**

骨格:

```
voice                 … 独立（腕ゲートなし。式と τ=0.2 は当面そのまま）
arm_raise             … 本体（挙げてキープ）
  └ hand_open_close   … 付加（arm_raise ON のときだけ）
arm_wave              … チャネルから外す（検知しない）
head_nod              … チャネルからもデモ主対象からも外す
```

比較・argmax はしない。独立ゲート。

##### `arm_raise` の量

- `hand_height_margin` を **高さの中央値からヒット率に再定義**する（`infant_signal.compute_hand_height_margin`）
- 直近 `HAND_HEIGHT_WINDOW=10` step。フレームごと帯内か: `z > z_eyes - HAND_HEIGHT_TOL`（0.4 m）
- 値は（帯内 step 数）/ 10。**10 点揃うまで 0**（短い窓の `k/n` は使わない）
- 成立: **≥ 0.8（8/10）**。1〜2 フレームの落ちは許す
- この 0.8 は幼児始動のチャネル ON、親始動の engagement、B 層 match で **同じ**。この量に 0.2 や `> 0` は使わない
- 高さの連続値は強度にしない（高さ sweep はしない）

##### `hand_open_close`

- `arm_raise` OFF なら salience 0。ON のあいだの **現行周期性の peak**
- 式は今のまま（示指往復と open↔closed 交替）
- τ は当面 0.2（目視後に `MATCH_QUANTITY_THRESHOLDS` を書き換え）
- 親 cue の原子に `open_close` 単体は無い。常に `arm_raise` 付き

##### match（親始動 B 層）

- 親が出したチャネルが **すべて** それぞれの τ 以上なら `response_matched=1`
- 幼児が余分な付加をしても減点しない（掲げ cue なのに開閉しても match）
- `peak_channel` argmax はやめる。注釈はチャネルごと成立 + 上記 match
- 実行時の praise / 状態分岐は変えない（D2 と同じ。注釈のみ）

##### 親の体

- **親始動 cue の `arm_raise`**: 掲げ IK のあと、見せとして現行振幅（0.35）の `wave_arm` を出してよい。検知は raise だけ（幼児が振らなくても挙げて 0.8 なら match）。振りは大きくしない（帯を出ると 8/10 が落ちる）
- **幼児始動の親**: 掲げキープのみ。開閉が乗っていればそれだけ足す。見せ振りはしない
- duration: 掲げ到達後 `arm_raise` 90 / `hand_open_close` 60（現行）
- ランダム cue は現行の「常に 2 スロット」: 50% `voice + モータ1本`、50% モータ 2 本。モータ候補は `arm_raise` / `open_close`。`open_close` を引いたら `arm_raise` を必ず足す。raise 単独のランダムは無し（格子 override では出す）

##### 幼児始動

- τ を超えたチャネルを **全部キュー**（最大 voice + raise + open_close）。`max_channels=2` の順位付けはしない
- 開閉は raise ON のときだけキュー
- 関節角コピーは **別バッチ**（本方針の親モータはパターンのまま）

##### `ProtocolManager`

このバッチで dyadic 訴求重みを新チャネルに合わせる。

- 載せる: `hand_height_margin`（ヒット率）、`hand_open_close_periodicity`、voice 系（数値は据え置き可）
- dyadic から外す: `hand_movement_size` / `hand_movement_regularity` / `head_nod_periodicity`
- orient / proximity は残してよい

##### デモ・定量格子

- 親始動格子: **`arm_raise`（強度なし。幼児は掲げキープで振らない）** と **`open_close` × {1.0, 0.5, 0.25}**（幼児は掲げ＋開閉）
- nod / wave 格子は廃止。`parent_arm_wave` / `parent_head_nod` シナリオは落とすか `parent_arm_raise` に置き換え
- 既存ランダム `parent_cue` は残す（上記 2 スロット）
- CSV: `peak_channel` はやめ、チャネルごと成立と上記 match。`eval_dyadic_response_match.py` を更新
- 幼児始動デモ: `head_nod` / `arm_wave` シナリオは `arm_raise` / `hand_open_close` / `voice` 系に合わせる

##### 含まない（別バッチ）

- 幼児始動の関節角コピー
- 開閉 τ の目視後の書き換え
- 偶然の同行動の切り分け（親が「幼児が今やっていないチャネル」に限定する等）
- 実行時の matched 時だけ praise を変える（D2 将来 B/C）

##### 触り先（実装時）

| ファイル | 変更 |
|----------|------|
| `cueworld/infant_signal.py` | `compute_hand_height_margin` を 10 step ヒット率に。10 未満は 0 |
| `cueworld/protocols/dyadic.py` | チャネル定数、salience、独立ゲート、match、ランダム sample、`arm_raise` モータ |
| `cueworld/protocol_manager.py` | dyadic 重み |
| `cueworld/parent_tools.py` | 親始動の見せ wave は既存 `wave_arm`。新 API 必須ではない |
| `cueworld/demo/demo_dyadic.py` | シナリオ差し替え、`arm_raise` 正例はキープ（振らない） |
| `scripts/experiment/eval_dyadic_response_match.py` | 格子と列 |
| `cueworld/tests/test_dyadic_*.py` / `test_protocol_manager_dyadic.py` | 新チャネル・ヒット率・重み |
| `cueworld/protocols/dyadic.md` | 実装後にスナップショット更新 |

##### 受け入れ基準（ゲート）

- 静止・未掲げでは `arm_raise` が立たない（窓未満は 0、ヒット率 < 0.8）
- 掲げキープ 8/10 で `arm_raise` ON。幼児が振らなくても親始動 `arm_raise` は match しうる
- `arm_raise` OFF のとき `hand_open_close` は 0
- 開閉強度 1.0 / 0.5 / 0.25 で peak が分かれること（CSV）
- 親始動 cue は振ってよいが、幼児始動の親は振らない
- `head_nod` / `arm_wave` は salience・ランダム cue・格子に出ない
- manager が振り・うなずきだけでは dyadic を押し上げず、ヒット率で押し上げる
- コンテナ `d994f05625a7`、既存 dyadic 関連テスト再緑（新定義に合わせて更新）


## 追記(4回目)
- 家事プロトコルは，現在はinspectionなしでやる(inspectionありだと，objectをinspectionするのは大丈夫だけど，掴んだ後boardをinspectionしようとするとinspectionの動きが上書きされて手のひらが開かれてしまい，落下してしまうのでそれを配置できない．)

---

## §6 着手前ゲート確認（2026-08-17）

個別プロトコルのチェック完了後、`protocol_manager.py` の選択・切替（§6）に入ってよいかを環境横断で監査した。切替ロジック自体は未改修。

### 判定: **条件付き GO**

プロトコル本体の作り直しは不要。残作業は切替ポリシー（household ガード、遊び履歴、行列チューニング）と、着手時にまとめてよい掃除である。着手を止める穴（到達圏 place 未実装、シグナル欠落、戻り値不一致）は見つからなかった。

### 確認1: カタログ・登録・ドキュメント

正本（§5）: `dyadic` / `joint_attention` / `object_giving` / `observance` / `household_solo`

| 箇所 | 実態 | 分類 |
|------|------|------|
| `protocol_manager.protocols` | 5本登録。`object_play` は未登録 | 一致 |
| `response_protocol` / `request_protocol` | JA・giving・dyadic の3本（solo 除外） | 一致 |
| `weights_dict["object_play"]` | 削除済み（バッチ A） | 一致 |
| `protocols/__init__.py` | カタログ外も export | 許容（ファイル残置方針） |
| `docs/protocol.md` | giving は到達圏 place。object_play はカタログ外・残置 | 一致（バッチ A 同期） |
| 本メモ先頭の推奨実装順表 | §3・§5 完了、§6 進行中（バッチ A） | 一致（バッチ A 同期） |

カタログ外（`object_play` / `dyadic_play` / `physical_contact`）は未登録のまま。復活させない方針とコードは一致。

### 確認2: プロトコル ↔ ProtocolManager 契約

`__call__` は基底が `(action, proximity, success_flag, reward)` を返す。`parent_info` は PM が常に `{}` を渡すが、どのプロトコルも読んでいない（実害なし）。

| キー | level | request | response | solo | has_target | persistence | 失敗経路 |
|------|-------|---------|----------|------|------------|-------------|---------|
| `observance` | 0 | F | F | T | F | 1.0（固定） | 終了しない |
| `dyadic` | 0 | T | T | F | F | 5.0 | `failure_state` あり |
| `object_giving` | 1 | T | T | F | T | 5.0 | 拒否・error timeout → failure |
| `joint_attention` | 2 | T | T | F | T | 5.0 | 拒否・error timeout → failure |
| `household_solo` | 0 | F | F | T | F | 5.0 | failure を使わず completed で抜ける |

契約上の穴（着手拒否にはしない）:

- `BaseProtocol._calculate_persistence` は `None` に戻す。カタログ5本はすべて上書きしている。
- `object_giving` / `dyadic` の `state_proximities` に `failure` が無い（`.get` で 0.0）。ペナルティ判定は `state == failure_state` なので影響なし。
- PM `__init__` は `super().__init__` なし。`eval_mode` は `__getattr__` → env。`BaseEnv.eval_mode` があるので現状は動く。
- `demo_mode` は `**kwargs` に飲まれて無視される（`demo_comprehensive.py` も同様）。

バッチ A で潰したもの: `ObservanceProtocol.start` の引数名は `context`。`household_solo` は `has_target=False` で汎用 `select_target` を走らせない。

household が自動開始も自動選択もできないのは §6 の未実装であり、ブロッカーではない。

### 確認3: シグナルパイプライン

PM `signal_keys` 17本は `get_normalized_signals()`（gesture / gaze / voice / body 既定オン）ですべて出る。

- dyadic 重みは追記(3)どおり: `hand_height_margin` / `hand_open_close_periodicity` / voice 系を載せ、wave・nod は 0。`test_protocol_manager_dyadic` が固定。
- `babbling_target` は重みに入れず、response 時の dyadic context 渡し専用。
- `active_signals` 指定時は検出器出力と行列列が同時に縮む。
- `emotion` キーは未出力。JA `intent_type=="sharing"` だけが読む。PM の JA 起動は `informative` 固定なので切替には影響しない。
- `negative_emotion_surge` は常に 0（TODO）。拒否は視線・回避側で動く。
- PM の番号コメントは `signal_keys` と同じ順（バッチ A で同期）。

### 確認4: §6 前ゲートの分類

**ブロッカー（未完なら NO-GO）→ いずれも完了**

- §3 到達圏 place: `object_giving` は `place_object_at`。`eval_object_giving_place` は memo 上 OVERALL PASS。先頭表も完了に同期済み。
- カタログ5本の FSM: 個別デモ・テストあり。
- 追記 A–H: memo 上実装済。コードも対応（`reach_radius` 系、テンプレ、音素長、escalate / show_pos / shake 掲げ / dyadic 掲げ）。
- 追記(2) T0–T2 / J1–J2 / D1–D2: **memo 本文は「実装待ち」のままだがコードは入っている**（AR=1 拒否、T2 定数コメント、`hold_gaze_links`、JA `infant_initiative`、`HAND_HEIGHT_WINDOW` ヒット率、`arm_raise`）。最大のドキュメントドリフト。
- T0 仕様の `env.action_dt = 0.1` 属性は `base.py` に無い。時間は step 前提コメントで運用。切替の前提は満たす。

**ソフト（着手可。§6 最初のスライスまたは直後）**

- 連鎖スモーク（§5 項目4）: JA shake ↔ giving の専用デモは無い。履歴検出は §6。個別デモがそれぞれ通ることの確認に留まっている。
- household inspection: 既定はオフ（`HOUSEHOLD_INSPECTION_PARAMS` の確率 0）。デモは `--inspection` の opt-in。コードパスは残置。
- H/F 目視の頭の不安定、dyadic 開閉 τ 仮値: 演出・プロトコル内。切替の前提ではない。

**後回し（memo 明示）**

- object_sound 同時発生、プロトコルごとのスケール集計、幼児始動 dyadic の関節角コピー。

### 確認5: 自動テスト（コンテナ `d994f05625a7`）

```
python3 -m unittest discover cueworld/tests -v
Ran 110 tests in 12.449s
OK
```

- household inspection 関連も今回は全成功（以前 memo にあったフレークは再現せず）。
- PM テストは `test_protocol_manager_dyadic.py` のみ（重み・dyadic context）。**自動切替の統合テストは無い**。§6 の受け入れとして後で足す。
- 自律切替の現行ベースライン: `cueworld/demo/demo_comprehensive.py`（`demo_mode=False`）。§6 完成の証明ではない。

### 確認6: 現行 PM 骨格 vs §6 仕様（次作業の地図）

既にある: response / initiative / persistence ダミークラス / ペナルティ / `recent_targets` / EMA。

未実装（着手拒否理由にしない）:

- household 中のインタラクション非遷移（`check_for_response` は solo でも走る。persistence 5.0 で抑えているだけ）
- household 開始経路（`request_type=False` のため initiative 候補に入らない）
- observance → インタラクション高遷移の検証
- 遊び連鎖の履歴検出と継続優遇（パターン未定義。着手時に dig）
- `reset` で EMA / try_count が残る
- initiative がほぼ `gaze_target_stability` のみ → D で「視線は親の手がかり、訴えにはしない」と確定。触る・持つを足すのは D2
- `save_protocol_history` の自発性定義 → D で確定（幼児始動かつ成功で上がる。式は据え置き、コメント直し）

### 条件付き GO の必須フォロー（§6 最初のスライスに含める）

1. デッド掃除: バッチ A で実施（`object_play` 重み・未使用 import 削除、コメントと `signal_keys` 同期、docs / 先頭表合わせ）。
2. household ガード: **撤回**。家事中も `check_for_response` は走らせる（persistence＋コストで起きにくくする）。
3. inspection オフ既定と `target_object` の cooldown 汚染停止: バッチ A で実施（`has_target=False`、確率既定 0）。
4. 連鎖スモークは独立プロトコルにしない。JA→giving の振る舞い確認は人間の動画、履歴検出は本実装側。

### §6 最初スライス案（実装は次プラン）

正本は下の「推奨実装順（§6 / `protocol_manager.py`）」。家事中の割り込み判定はスキップしない（persistence＋コスト）。最初の実装単位はバッチ A（inspection オフ・家事 target・observance `start`・デッド掃除）。


### ここからやるべきになりそうなこと
- 親の情報，今までのプロトコルの情報などをプロトコルが使う必要がある場所はないのか？
- protocol_managerに家事と見守りの切替えはまだない？
- 関わりプロトコルを抜けるのは，失敗だけなの？完了はない？(要確認)
- 見守りタスクの開始時の引き数が約束と違うらしい
- 家事は開始時に、家事と関係ない物を「今回の対象」として拾ってしまう。今は家事を固定起動しているだけなので実害は小さい。あとで家事を自動開始するときに、「最近使った物は避ける」リストが汚れる原因になる。(家事の対象チェック): 優先
- infant_signalに必要なものがなにか(感情に関するものなど，拒否判定にも使えるかも)
- 家事プロトコルのinspectionをオフにする: 優先
- 物体の音が同時に鳴ったときの扱い(後回しでいい)
- 遊びごとの長さの集計
- 家事中も割り込み判定はする．(割り込みされにくいだけ．例えば，割り込みするには幼児側に何かしらのコストがかかる仕組みが必要かも．)
- 切替の頻度がちょうど良いか
  - 見守りから他の関わりプロトコルへの切替え頻度が適切か
  - 家事や他のプロトコルを行っている間は切替え頻度がどうか
- 今までのプロトコル自体の履歴情報を利用できる仕組み
- PM は成功率や試行回数を覚えています。 でも reset のときに、 その一部がちゃんと初期化されないまま残っています。→これは意図的．この環境のリセットは幼児の睡眠に該当．環境の配置や関節状態などはリセットされるがそれ以外はリセットされないはず(幼児の学習レベルとかは変わらない)．
- 親が自分から始める判断が、まだ視線に寄りすぎている．→**D で定義確定**（視線は手がかり、開始主体は親）。触る・持つを足すのは §6 の D2。
- 幼児の自発性の定義．→**D で確定**（⑤）。
- プロトコルが単体で動いても，他のプロトコルとつなげたら動かない可能性がある(例えば，オブジェクト操作によって，他のプロトコルができなくなるとかはぜんぜんありえる．)．

### 推奨実装順（§6 / `protocol_manager.py`）

上の箇条書きを、依存関係で並べ替えたもの。**定義 → 家事を切替に載せられる状態にする → 切替経路を足す → 履歴と自発性の定義 → 意図の部品 → 頻度の観測・調整 → 連鎖確認**。切替ロジックを先にいじると、定義と家事の穴の上に乗ることになる。

方針として確定したもの（本節の前提）:
- **reset で EMA / try_count を消さない**。reset は幼児の睡眠。配置・関節は戻るが、学習レベル・成功率は残す。`reset` の TODO コメントは「睡眠で消すもの／残すもの」の注記に直すだけで、実装でゼロ初期化しない。
- **家事中も割り込み判定はする**。`check_for_response` を家事中にスキップしない。persistence で起きにくくし、必要なら幼児側コストを足す（ゲート確認の「household ガード＝判定しない」は撤回）。
- 物体の音の同時発生は後回しのまま。
- **幼児始動／親始動**（0b/0c、バッチ D）: 訴え判定で切替＝幼児始動。見ているだけは訴えにしない。詳細は⑤。

```
⓪ 定義・事実確認（dig / 読み取り）
    ↓
① 契約の穴＋家事を切替に載せる準備（優先・実装）
    ↓
② 見守り ↔ 家事の開始経路（PM 本体）
    ↓
③ 割り込みコスト（家事中も判定するが起きにくくする）
    ↓
④ 履歴バッファ＋ context / parent_info の受け渡し
    ↓
⑤ 自発性の定義をコードに落とす（0b/0c、視線を訴えから外す）
    ↓
⑤b 意図の部品・手がかり・信号一覧（dig）＋意図レイヤ実装
    ↓
⑥ 切替頻度の観測 → 温度・persistence の調整
    ↓
⑦ 連鎖スモーク（単体OKでもつなぐと壊れる）
    ↓
⑧ 後回し（音の同時発生、長さの集計）
```

#### ⓪ 定義・事実確認 — 実装の前に1回で片付ける

コードを大きく変える前に、切替の意味を固定する。ここが曖昧なまま重みや頻度を触ると、あとで全部やり直しになる。

| 順 | 項目 | 種別 | 何を決める／確認するか | なぜ先か |
|----|------|------|------------------------|----------|
| 0a | 関わりの抜け方（完了 vs 失敗） | 確認 | `update` は `is_finished` なら成功・失敗どちらでも observance に戻る。割り込み中断だけ失敗記録。完了は既にある | 履歴・ペナルティ・頻度の解釈がこれに依存する |
| 0b | 幼児の自発性の定義 | **確定（D）** | 「幼児始動」＝`check_for_response` で切替したこと。自発性 EMA は幼児始動かつ成功のときだけ上がる | 0c・⑤・initiative の信号設計の正本 |
| 0c | 親始動 vs 幼児始動の境界 | **確定（D、0b と一体）** | 幼児が見ている物に親が JA / giving を始めるケースは **親始動**。視線は開始の手がかりであり、開始主体は親。見るだけは訴えにしない | 親の手がかりを視線以外に足す（§6-D2）前に必要だった |
| 0d | プロトコルが親情報・履歴を読む必要があるか | 監査 | 現状 `parent_info={}` で誰も読んでいない。読む必要がある場所があるなら④で渡す。なければ空のままでよい | ④の API を先に決めない |
| 0e | `infant_signal` に足りないもの | 棚卸し | 感情は未出力。拒否は視線・回避で動く。切替に今すぐ感情は不要。拒否強化や sharing を後で足すなら別バッチ | 感情実装を切替のゲートにしない |

**0a の現状（確認結果として書いておく）**: 完了はある。`success_flag=True` で履歴保存し、`is_finished` なら observance へ。失敗は `failure_state` でペナルティ。割り込み切替時は前回を失敗扱い。抜けが失敗だけ、ではない。

**0d / 0e の現状（確認結果として書いておく）**:
- `parent_info` は PM が常に `{}` を渡しているが、現状どのプロトコルも読んでいない。④で履歴や親情報をプロトコルに渡すかは、実際に読む先が出てから決める。
- 感情シグナルは未出力。拒否は視線・回避で動いており、切替の前提として感情を足す必要はない。感情を使うなら後バッチ。

**0b / 0c はバッチ D で確定**（下の「⑤」と「dig 確定（バッチ D）」を正とする）。バッチ A 時点では家事切替の前提ではないとして先送りしていた。

#### ① 契約の穴＋家事を切替に載せる準備 — **優先・先に実装**

家事を PM から自動開始する前に潰す。ここを飛ばすと cooldown が汚れ、inspection で物が落ちる。

| 順 | 項目 | 触り先 | 内容 |
|----|------|--------|------|
| 1a | 家事 inspection をオフ | `household_schemas.py` の既定、必要なら `household_solo.start` | `inspect_*_prob` / `hesitate_prob` を **0 が正**にする。デモは `--inspection` の opt-in。テスト上書きに頼らない。コードパスは残してよい |
| 1b | 家事の対象チェック | `HouseholdSoloProtocol` / `BaseProtocol.start` | `has_target=False`。汎用 `select_target` を走らせない。家事の対象は board / item 側。`target_object` は使わず、終了時も `recent_targets` に入らない |
| 1c | 見守りの `start` 引数名 | `observance.py` | `start(self, context)` に合わせる。動いているが契約が汚いだけ |
| 1d | デッド掃除 | `protocol_manager.py` | `object_play` の重み行・未使用 import、コメントと `signal_keys` の順序を揃える。docs / 先頭表の現状合わせは同時でよい |

1a と 1b は独立なので並行可。1b は②の前提（家事自動開始で cooldown 汚染が顕在化する）。

**バッチ A で dig により確定した実装方針**:
- **1b 家事 target**: `household_solo` は遊び対象プロトコルではない。`has_target=False` とし、`target_object` は使わない。家事の対象は `target_board_id` / `current_item_id` のまま持つ。
- **家事と cooldown の関係**: 家事で置いた物は直後の JA / giving の候補に残してよい。cooldown（`recent_targets`）は関わり側の対象専用とし、household 用の特例や家事対象の追加はしない。
- **1a inspection オフ**: `HOUSEHOLD_INSPECTION_PARAMS` の既定値は `inspect_item_prob=0`, `inspect_board_prob=0`, `hesitate_prob=0` を正とする。inspection のコードパスは残す。
- **inspection のデモ入口**: `--no-inspection` ではなく `--inspection` の opt-in にする。オン時は「item inspection を必ず出す」「board inspection を必ず出す」「hesitate は出さない」を正とする。
- **1d デッド掃除の範囲**: `protocol_manager.py` の未使用 import / `object_play` 重み行 / `signal_keys` コメント不整合を直す。`docs/protocol.md` と本 memo の先頭表も現状同期する。`object_play.py` 自体は残置のまま。
- **A のテスト受け入れ**: 家事開始後に `target_object` が空であること、inspection 既定が 0 であること、household 完了後に `recent_targets` が増えないことをテストで固定する。既存の inspection オン系テストは残す。

#### ② 見守り ↔ 家事の開始経路

現状: 家事は `request_type=False` かつ `solo=True` のため、initiative 候補に入らない。終了時は observance に戻る経路だけある。**見守り→家事の開始が無い。**

| 順 | 内容 |
|----|------|
| 2a | 家事開始は確率選択に入れず、**前回の household 完了から N step 経過したら**候補にする。仮値は **N=200**。本番ではもっと長くして頻度を落とす前提で、値は引数化する |
| 2b | **見守り中**に期限が来たら household へ切り替える。ただし同じ判定タイミングで幼児の訴えが勝ったら、**関わりを優先**し、household はその関わり終了まで待つ |
| 2c | **関わり中**に household の期限が来ても割り込まない。関わり終了時に期限を過ぎていれば、observance を1回挟まず **そのまま household へ入ってよい** |
| 2d | 初回 household の基準時刻は **episode 開始時**（起床時）とみなす。つまり初回は概ね `step >= N` で発火する |
| 2e | household の実行可否は実際のタスク候補に従う。候補が無いなら household へ入らず observance 継続。期限フラグは消さず、次の空きタイミングで再判定する |

§6 の「household → インタラクションへは遷移しない」は、**開始経路の話ではなく割り込みの話**。割り込み自体は③で「起きにくくする」であり、禁止しない（本節の方針）。

**dig 確定（2026-08, バッチ B）**:
- household の時計は **前回 household の開始時刻ではなく完了時刻**から数える。家事直後の即再開を避けるため。
- observance 中に household 期限と幼児の訴えが重なったら、**幼児の訴えを優先**する。household は「親が空いているときにやる」。
- 親が自分から JA / giving / dyadic を始めて household を先延ばしにする経路は作らない。期限到来後に空いていて、幼児の訴えも勝っていなければ household に入る。

#### ③ 家事中の割り込みコスト

判定はする。**ただし最小実装では追加コストを入れない。** 連続シグナル・urgency・household 中だけの特別倍率は後回し。

- 現段階では `check_for_response` を household 中もそのまま走らせ、**現行の重み + persistence 5.0** で観測する
- 「特別に強い訴え」を入れるなら、重み調整後に再考する。今の `strengths` のスケールは固定ではなく、先にここを作り込むと手戻りになる
- 案の置き場自体（シグナル強度から persistence を十分超えないと家事を割れない、urgency、連続シグナル等）は残すが、**B では実装しない**
- 関わり同士の割り込みは現行の persistence + ペナルティのままでまず観測し、⑥で触る

#### ④ 履歴を使えるようにする

「今までのプロトコル自体の履歴」は、遊び連鎖の検出・継続優遇・parent_info の正本になる。

| 順 | 内容 |
|----|------|
| 4a | 終了ごとにバッファへ積む。reset（睡眠）では **残す**（⑬）。1件の欄は下の dig 確定を正とする |
| 4b | 0d で「プロトコルが読む」と分かったものだけ `context` / `parent_info` に載せる。読まないなら PM 内部とログに留める |
| 4c | JA↔giving のシーケンス検出と選択優遇は、パターンを **着手時に dig** してから。バッファだけ先に作ってよい |

**B の前提メモ**:
- household は幼児の学習課題ではないので、**成功率・自発性・level の EMA を更新しない**。コミュニケーション用の試行回数にも混ぜない。

遊びの長さの集計（⑨）は 4a のフィールドがあれば後から足せる。切替の本体ではないので⑧へ。

**dig 確定（2026-08, バッチ C）**:
- C はノートを書くだけ。プロトコルはまだ読まない。続きの優遇（4c）と `parent_info` 渡しは後
- 1件: `name` / `success` / `infant_initiative`（経路の旗。D で定義確定。訴え判定で始まったら True）/ 終わった時点の `target_id` / 長さ / 開始ステップ / 終了ステップ / 種類 / 終わり方
- 種類: JA は `cue_type`（指差し／振る）。dyadic は使ったチャネル。giving は空
- 書くのは関わりだけ（JA / giving / dyadic）。observance は書かない。household は今書かない。必要なら後から同じ形式で足せる
- 最後まで終わったものも、途中の割り込みも書く。終わり方は成功・失敗・割り込み。EMA は今どおり割り込み＝失敗
- 睡眠時に途中だった関わりは書かない。ノート自体は睡眠をまたいで残す
- 直近 32 件（仮）。info へ出す専用旗を作り、既定はオン。オン時は直近 8 件。カメラ用 `debug_info` とは別。学習で重ければオフ
- 対象が途中で変わったら、終わった時点の物を書く。dyadic の対象は空

#### ⑤ 自発性をコードに落とす

0b / 0c の定義が正。実装はその写し。意図の部品・手がかりの中身は ⑤b（§6-D2。§5 の二項 D2 とは別）。

**dig 確定（2026-08, バッチ D）**:

定義（0b / 0c）:
- **幼児始動** ＝ `check_for_response`（訴え判定）でプロトコルが切り替わったこと
- **親始動** ＝ `check_for_initiative`（見守り中に親が自分から始めた）こと
- 家事・見守りの開始はどちらでもない（`solo`。EMA にも履歴にも混ぜない）
- 幼児が物を見ているだけで親が JA / giving を始めたら **親始動**。視線は開始の手がかりであり、開始主体は親
- じっと見ているだけは **幼児の訴えではない**。訴えは働きかけがはっきりしたもの（指差し・リーチ・単語の発声・二項の salience など）
- 親の顔と物を行き来する視線（共同注意の呼びかけ）を訴えにする案は **D では採用しない**。`looking_at_parent_eye` は検出器にあるが PM の訴え行列には載っていない。D2 の部品一覧で扱う
- 共同注意の流れ（追従するか見せるか）も、この経路の旗（`infant_initiative`）で分岐する。開始時に信号を見て旗を付け直さない
- 履歴ノートの `infant_initiative` も同じ旗（終了するインスタンスのもの。C どおり）
- `response_type` / `request_type` は「その経路の **候補に入れるか**」。`infant_initiative` は「**実際に** 訴え経路で始まったか」。別物
- `save_protocol_history` の TODO（`function_type` で自発的かつ社会的かを再定義）は **D では使わない**。正本は経路

訴えと親のきっかけを混ぜない:
- 同じ動きを両方の判定に載せると、当たり方で旗が嘘になる（リーチなのに親始動、など）
- そのため訴え側と親側で使う信号の種類を分ける
- 判定の順は現行どおり訴えが先。だからこそ、訴えになるべき動きを親のきっかけに残すと、訴えが外れたときだけ親始動になる

線形和の穴（なぜ意図を先に合成するか）:
- 今の訴えは部品を独立に足す。揃っていない動きでも、どれか一つが強いと訴えになる
- 見守り persistence は 1.0。JA の視線 0.5・単語 0.5、giving の腕伸ばし 0.5 は、単体でも切替のくじに入りうる
- 組み合わせが必要なことはコードも部分的に知っているが、**開始判定では使っていない**
  - JA の `_pointing_is_reliable`（指差し AND 安定 AND 対象）は対象選び
  - 二項の `compute_channel_salience`（バブリング×音量×単語らしくない、掲げ、掲げ時の手開閉）は起動後のチャネル選び
- 部品は拒否・対象選び・プロトコル内でも使うので、解釈（意図）は `infant_signal` に混ぜない。各プロトコルが開始スコアを自分で出す案も D では不採用（開始の訴えは親＝PM の仕事。二項 salience の開始への流用は D2）

自発性の数字（`infant_initiative_rate`）:
- **幼児始動かつ成功** のときだけ上がる。親始動・失敗・割り込みでは 0 に近づく
- 使う先: `wait_score = wait_base_score + s_spon`。高いほど見守りやすい（うまく自分から始められるほど親は待つ）
- コメントの「割合」は誤り。成功率寄り
- 式は今の EMA のまま。直すのはコメントと docs。家事は混ぜない（C どおり）
- 試行が `base_try_count` 未満のときは待つ量には使わない（現行どおり）
- 割り込み時、PM の `current_infant_initiative` は新プロトコル用に True になるが、`success_flag=False` なので **この数字は増えない**。履歴バッファは C でインスタンスの旗を使う。EMA の更新式は D では触らない
- 不採用: 始動の割合（失敗でも上がる）、幼児始動の試行だけ見て成功率（親始動では変えない）、履歴ノートからその場計算

アーキテクチャ（中身の式は §6-D2）:
- `infant_signal.py` は部品（原子信号）のまま
- `protocol_manager` が行列の **前** で「意図」を合成する
- 訴えの行列は意図だけを見る。部品を直接足さない（腕伸ばし単体が訴えになる穴を塞ぐ）
- 向き・近さは意図そのものにしない。必要なら各意図を作るときに掛ける（D2）
- 最初の意図は 4 つ: **指差し / リーチ要求 / 単語の発声 / 二項**（二項は既存 salience を流用する前提）
- 親が自分から始める手がかりは **視線・触っている・持っている**（注意）。指差し・リーチ・単語は使わない
- 単語以外の声は仮置き: バブリングと声量は二項の訴え。親のきっかけにはしない。泣き・不快は感情バッチ。中身は D2
- 不採用（D）: 意図を `infant_signal` が出す、仮の式で意図の枠だけ D で作る

現行コード（D / D2 の参照）:
- 訴え `weights_dict`（D で視線だけ外す。他は D2 まで残る）:
  - JA: `point_intensity` 0.5, `hand_target_stability` 0.10, `arm_ext` 0.1, `gaze_target_stability` 0.5→**0**, `voice_target_stability` 0.5（残す）, `orientation_to_parent` 0.10
  - giving: `open_intensity` 0.15, `hand_target_stability` 0.10, `holding` -1.0, `arm_ext` 0.5, `gaze_target_stability` 0.15→**0**, `voice_target_stability` 0.05, `proximity_intensity` 0.05, `orientation_to_parent` 0.05
  - dyadic: `voice_volume` 0.4, `babbling_ratio` 0.5, `voice_target_stability` -0.4, `hand_height_margin` 0.40, `hand_open_close_periodicity` 0.15, `orientation_to_parent` 0.10, `proximity_intensity` 0.10（視線は元から無い）
- 親始動: `has_target` なら `gaze_target_stability` を掛ける。dyadic は `has_target=False` なので視線なし（ZPD のみ）。コードに「触っているなども」の TODO
- 「触っている」は `get_hand_status` の内部にあるが、独立した信号キーとしては出ていない。`holding` はある
- 検出器にあって PM 訴えに無い例: `looking_at_parent_eye`, `voice_volume_sustained`。pitch は履歴のみで未使用

バッチ D の実装範囲（仮の式で意図レイヤは作らない）:
- 本 memo への定義・D2 で決めることの記入
- 訴えの `weights_dict` から `gaze_target_stability` を外す（JA 0.5、giving 0.15 を載せない）。**単語の発声は残す**
- D の間は部品の足し算が残る。**腕伸ばしだけが giving の幼児始動になりうる穴は D2 まで残る**（視線の穴だけ先に塞ぐ）
- `infant_initiative_rate` のコメント／`docs/protocol_manager.md` を上の意味に合わせる
- 触る信号の追加、持つことの親側への配線、意図 4 つの枠は **§6-D2**
- D のテスト: 訴え行列で JA / giving の視線重みが 0 であること。腕伸ばし単体の穴は D2 のテスト

#### ⑤b 意図の部品・手がかり・信号一覧 — **バッチ §6-D2（D の直後、E の前）**

名前: この D2 は **§6 の切替バッチ**。§5 の二項バッチ D2（④b、幼児が親を真似る）とは **別**。紛らわしいときは「§6-D2」と書く。

定義（⑤）のあと、頻度（⑥）の前。入力が変わるので、今の足し算のまま E をやるとやり直しになる。決めたらそのバッチで意図レイヤまで実装する（仮の式をテストに固定しない）。**方針は下記「dig 確定（2026-08, バッチ §6-D2）」が正。** 上の箇条書きは着手時の問い。

**D2 で決めること（着手時に dig）**:

1. **4 つの意図の部品と掛け方**（ここが本体）
   - **指差し**: 既存 `_pointing_is_reliable`（指差し AND 手の向き安定 AND 対象がある）を流用するか。連続値にするか、0/1 か
   - **リーチ要求**: 腕伸ばし・手を開く・手の向きの安定をどう掛けるか。`holding` で抑制するか
   - **単語の発声**: `voice_target_stability` だけか。対象への結び（視線は訴えにしない、との整合）をどうするか
   - **二項**: `DyadicProtocol.compute_channel_salience` を開始判定に使うか。チャネルを max / 和のどれにするか。起動後の τ ゲートと開始の式を揃えるか
   - 参考（起動後の現行）: 音声 salience = `babbling_ratio * voice_volume * (1 - voice_target_stability)`。手開閉は掲げが閾値以上のときだけ非ゼロ
2. **意図 → プロトコルの重み**（4 意図 × JA / giving / dyadic）
   - 指差しは共同注意だけか、物渡しにも薄く乗るか
   - 単語は共同注意だけか
   - リーチは物渡しだけか
3. **親の手がかりの合成**（視線・持っている・触っている）
   - 掛け算か、max か
   - 物が要るプロトコル（JA / giving）だけか。二項の親始動に `looking_at_parent_eye` などを使うか
4. **触る信号**: 今は独立キーが無い（手の対象計算の内部にだけある）。`infant_signal` に足すか、持つ（`holding`）だけで足りるか
5. **単語以外の声**（⑤では仮置き: バブリングは二項の訴え、親のきっかけにはしない）
   - バブリングは二項の訴えのままか
   - 単語でもバブリングでもない声（音量・持続 `voice_volume_sustained`・高さ。高さは履歴のみで未使用）を訴えにするか、親のきっかけにするか、今は無視するか
   - 親を見ながらの声を二項の訴えに入れるか
   - D で不採用だった「非単語の声も親のきっかけ」は、ここに限って再検討してよい
6. **部品一覧の不足・余り**
   - PM の訴え行列に無いが検出器にはあるもの: `looking_at_parent_eye` など
   - 意図の材料に使わない部品を、開始判定から外す（プロトコル内の対象選び・拒否には残してよい）
   - 向き・近さは意図そのものにしない（⑤）。各意図を作るときに掛けるか、掛けないか
7. **実装**: PM が行列前で意図を作る。訴え行列は意図 4 つのみ。`infant_signal` は原子のまま。親手がかりも部品の直積ではなく合成してから `check_for_initiative` に渡す
8. **テストで固定したい穴**（式が決まったら）
   - 視線だけ → 幼児始動の JA にならない
   - 腕伸ばしだけ → 幼児始動の giving にならない
   - 指差し（決めた部品が揃ったとき）→ 訴え側
   - 見る・触る・持つ → 親始動の手がかりになり、旗は False

**dig 確定（2026-08, バッチ §6-D2）**:

方針として、D の「4 意図」はラベルにすぎない。掛け算で作るセットは 4 個に縛らない。同じジェスチャーの中は掛け算（揃ったら 0/1）。手と声は掛け算しない（別セットを同じ遊びへ足す）。`infant_signal.py` は原子のまま。PM が行列の前でセットを合成する。訴え行列はセットだけを見る。部品の直和は開始判定から外す。完成形の重みは温度より十分大きくする（目安 2）。弱いセットは薄い重み。値の本調整は E の観測のあと（E 自体は重み行列を触らない）。

共通の揃え（手の対象）: `hand_target_stability >= 0.6` かつ `gesture_targets` 最大質量 `>= 0.5`。持っているときは手の対象が常にその物になるので、弱い腕・リーチでは `holding` なら 0。腕の伸び `arm_ext` は静止時 1.0 のため使わない。

訴えのセット（幼児始動）:

| セット | 0/1 | 乗る先 | 対象 |
|--------|-----|--------|------|
| **揃った指差し** | 既存 `_pointing_is_reliable` と同じ（形が pointing・安定 0.6・質量 0.5・親以外の物）。`ignored_targets` は 0/1 に入れない（対象選び用。PM 側は `recent_targets` で既にマスク） | JA のみ（完成形の重み） | 手の先 |
| **揃ったリーチ** | 開き手 AND 安定 0.6 AND 質量 0.5 AND 持っていない | giving のみ（完成形の重み） | 手の先 |
| **弱い腕** | 形を問わない。持っていない AND 安定 0.6 AND 質量 0.5。完成形が立っていても常時足す | JA と giving の両方へ薄く | 手の先 |
| **単語** | 単語らしい（`voice_target_stability`）AND 今声が出ている AND 当たった単語の concept_id が場にある。視線は積に入れない。黙ったあとに残さない | **揃った指差しが 0 のときだけ** giving（完成形なみの重み）。指差しが 1 のときはくれに足さない（JA 内のラベル／言い直しだけ） | 単語の中身（concept_id）。視線より優先 |
| **二項** | `DyadicProtocol.compute_channel_salience` と同じ式・同じ τ。どれか一つのチャネルが τ 以上なら 1。複数立っても二項は 1 本（中で全部模倣） | dyadic | 物なし |

ゲート（完成形同士の競合を消す）:
- 揃った指差しが 1 のとき、単語はくれに足さない（指差し＋名前は JA。「名前だけ」がくれ）
- 物への手（揃った指差し・揃ったリーチ・弱い腕）が **JA / giving として実際にくじに乗っている** とき、二項は 0。物を介さないときだけ二項
- 例外: 拒否・ペナルティ・`recent_targets` マスクなどで JA / giving の点数が 0 なら、手が物の方向でも二項は足す（共同注意を強制 0 にしたあと残るのが二項だけなら二項）

「くれ」の距離（幼児始動も親始動も同じ）:
- 対象が幼児の到達圏内（既存 `reach_radius`、床 XY、高さは見ない）なら giving は 0
- 持っている物もくれにしない
- 届く物へのリーチ／名前呼びは訴えにしない（自分で取れる）。指差しは近くでも JA
- 親への近さ `proximity_intensity` は使わない

対象と成功:
- 手が 1（揃った指差し or 弱い腕 or 揃ったリーチ）なら対象は手の先。単語だけのくれは単語の中身
- 親の顔を見ながらの指差しは幼児始動 JA。視線は訴えの積に入れない
- JA の成功判定は **いつも視線**（手の先に決めた物を見るまで待つ）。現行の「揃った指差し中は成功も手」は変える
- 幼児始動 JA で指差しが揃っていないとき、現行は視線／ランダムに落ちる。D2 では弱い腕でも手の先を取る（プロトコル側を直す）。giving の幼児始動も手の先（現行は視線のみ）
- 単語は対象を決めない（JA 内）。くれの単語セットだけが concept_id で対象を取る。場に無い／当たっていない単語はくれにしない（ランダムにしない）

親の手がかり（旗は False。指差し・リーチ・単語は使わない）:
- JA / giving: **見る OR 触る OR 持つ**。見るは `gaze_target_stability >= 0.6` で 0/1。持つは既存 `holding`。触るは独立キーを足す（今は `get_hand_status` 内部。開き手で `detect_grasp_object`）
- dyadic: **`looking_at_parent_eye`**。親の目を見ているとき親が二項を始めやすい。見るだけでは幼児始動にしない
- giving の親始動も到達圏ゲートを通す（届く物・持っている物はくれ候補にしない）

単語以外の声:
- バブリングの定義は現行どおり（今声があるとき、直近 20 音素で最頻の有効音素が `babbling_target`、割合が `babbling_ratio`。無音なら 0）。二項の声チャネルに既に入っている
- 単語でもバブリングでもない声（持続 `voice_volume_sustained`・高さ）は開始に使わない。泣き声を拒否信号に使う案は後バッチ（感情／拒否）。D2 では載せない

開始判定から外す部品（検出器と拒否・プロトコル内には残す）:
- `orientation_to_parent` / `proximity_intensity` / `fist_intensity` / `head_nod_periodicity` / `hand_movement_size` / `hand_movement_regularity` / `arm_ext`
- 向き・近さは意図にも各セットの積にも掛けない

不採用:
- 完成形を連続の積のままにする（中間値でくれに漏れる）
- 弱いセットを完成形よりゆるいしきい値／連続にする（向きがあいまいでも訴え）
- 視線を指差しの積に入れる（親を見ながらの指差しが落ちる）
- 単語だけで JA の幼児始動（名前呼びはくれ。JA は指しが本体）
- 指差し＋単語をくれにくじで割る
- 見るだけを幼児始動にする（D どおり）

プロトコル側の追随（開始セットと対象がずれないようにする）:
- `JointAttentionProtocol.select_target`: 幼児始動かつ（揃った指差し or 弱い腕）なら `gesture_targets`
- `get_infant_target`（成功）: いつも `gaze_targets`（決めた対象を見ているか）
- `ObjectGivingProtocol`: 幼児始動なら手の先（または単語の concept_id）。到達圏外のみ

バッチ §6-D2 の実装範囲:
- 上のセット合成＋訴え行列の載せ替え＋親手がかりの合成
- `infant_signal` に触るキーを足す。原子以外の解釈は PM
- JA / giving の対象選びと JA 成功判定の視線化
- 仮の式をテストに固定しない。下の穴だけ固定する

**実装済み（2026-08）**: `cueworld/intent.py`、`protocol_manager` の意図行列・ゲート、`touching` キー、`object_in_agent_reach`、JA/giving の `intent_target_id` 追随、`get_infant_target` の視線化。テストは `test_protocol_manager_intent.py` ほか。重みの本調整は E の観測のあと（E では触らない）。

テストで固定したい穴（追加分含む）:
- 視線だけ → 幼児始動の JA にならない
- 腕伸ばしだけ（向きが揃っていない／`arm_ext` のみ）→ 幼児始動の giving にならない
- 揃った指差し → 幼児始動 JA。旗は True
- 見る・触る・持つ → 親始動。旗は False
- 届く物へのリーチ → giving にならない
- 揃った指差し＋単語 → giving にならない
- 拒否のあと JA / giving が 0 で二項だけ残る → 二項は止まない

#### ⑥ 切替頻度の観測と調整

②③⑤と **D2（⑤b）** が入ってから。今の `demo_comprehensive` は骨格のベースラインに過ぎない。入力（意図）が変わる前にノブを合わせない。

見るもの:
- 見守り → 関わりの頻度（高すぎ／低すぎ）
- 家事中の割り込みが起きすぎないか（③のコスト）
- 関わり中の persistence が強すぎて幼児の訴えを無視していないか

触るノブ: `tau_response` / `tau_initiative` / `wait_base_score` / `cue_bonus` / `persistence_*` / 家事割り込みコスト。重み行列の本チューニングはこの観測のあと。

**dig 確定（2026-08, バッチ E）**:

E の本体は「短い観測で数字 → 温度・待ち・家事間隔を動かす → comprehensive で見た目」。重み行列は触らない（D2 どおり観測のあと）。家事への追加コストは入れない（維持 5.0 がコスト）。連続シグナルカウントは入れない。

観測の単位:
- 短い観測は物理なし。`test_protocol_manager_intent.py` と同じ層（信号辞書を直接入れる）
- `check_for_response` を1回だけ呼ばない。典型の滞在 step 分 `update` を回す（判定は `check_interval=10` ごと。時計は物理と同一）
- 典型 T を主、上限 T（リトライなし1周）を併記。家事の典型は未計測なので giving 相当 **175** で仮置き
- 親始動は起床直後（`try_count=0`、候補は二項だけ）と、各レベル 30 試行済み（JA / giving も候補）の両方
- 判定の時計（1 step、`check_interval`）は物理を載せても変わらない。変わるのはプロトコル長さ（判定回数）と訴えの点灯率。点灯率は短い観測では 100%（一定信号）。長めランで後から見る

短い観測の行:

| 今のプロトコル | 信号 | 見るもの |
|----------------|------|----------|
| observance | ゼロ | 手がかりなしの親始動。冷／温で判定あたり ≈0.0056（温でも同じ）。200 step で一度でも ≈10% |
| observance | 物の視線（`gaze_target_stability`） | 温: ≈0.033。冷は二項しか無いので zeros と同じ |
| observance | 親の目（`looking_at_parent_eye`） | 冷でも ≈0.033 |
| observance | 揃った指差し | すぐ幼児始動 JA |
| household | 揃った指差し | ほぼ割れない |
| household | ゼロ | 対照（割れない） |
| JA 未乗車 | 揃ったリーチ | ≈0.12 |
| JA 乗車（視線を対象へ） | 別物リーチ | ≈0 |
| JA 乗車 | 同じ物リーチ（届かない物） | ≈0.12 |
| JA（`retry_count >= 1`） | 揃ったリーチ | ≈0.12 |
| observance・期限到来・訴えなし | （空いている） | 家事間隔（デモ 200）で household に入るか |

弱い腕・くれ中・二項中の行は足さない（維持がプロトコルで違うわけではない。JA で代表）。

親始動の式:
- 一段目は待機 vs 開始（候補数・ZPD に依存しない）。手がかりが候補に1つでも立てば `cue_bonus`、なければ 0
- 二段目は `zpd + cue_bonus * cue_i`（ペナルティ減算）の softmax。掛け算パスは残さない
- 自発性 0・τ=0.2: `wait_base_score ≈ 1.04`（判定あたり ≈0.0056、約3分）、`cue_bonus ≈ 0.36`（手がかりあり ≈0.033、約30秒）
- 本番の家事間隔は `HOUSEHOLD_INTERVAL_PRODUCTION = 6000`。デモ・短い観測の既定は 200

維持（`check_for_response` のダミークラス）:
- 見守り: **1.0**
- 家事: **5.0**（完成形でもほぼ割れない。追加コストなし）
- 関わり（JA / giving / dyadic）: 未乗車／同じ物の完成形／`retry_count >= 1` → **2.7**。乗車かつ別物（または対象なし）→ **5.0**
- 乗った証拠: JA は対象への視線。くれは提示対象への見る／触る／持つ。二項は `imitating` / `acknowledging`、または親始動で反応したあと
- 経過時間の半分は使わない。proximity は使わない
- 式の所在は `BaseProtocol._calculate_persistence`。PM は `protocol.persistence` を読むだけ。
- 典型 T（成功1回、観測の長さ用）: JA 幼児始動・指差し **130**、giving **175**、dyadic 腕 **105**
- 2.7 は完成形 2.3・τ=0.2 のとき、10 step（判定1回）で約 12%。値は仮
- `retry_count >= 1` なら乗車に関係なく 2.7（RETRY / ESCALATE / 対象変更も。もともと失敗しているので別遊びへ割れやすい方がよい）
- 温度を上げて積算を効かせる案は不採用（見守り中の点数 0 がうるさくなる）。重みを上げて積算する案も不採用（家事も同じ確率で割れる）
- 連続 K 回立ってから割る案は不採用（長いほどサイコロで上がれば足りる。短い完成形にも数%は許す）

テストで固定する穴（数値 2.7 は仮なので形だけ）:
- リトライ中 → 維持 2.7
- 未乗車 → 維持 2.7
- 乗車かつ別物 → 維持 5.0
- 乗車かつ同じ物の完成形 → 維持 2.7
- 家事中の完成形 → ほぼ割れない（5.0）
- 視線だけ → 幼児始動にならない（D2 のまま）
- 二段くじ: 温でも zeros の開始 p が冷とほぼ同じ、gaze で約6倍、候補数で一段目が変わらない

E の作業順:
1. 上の維持式を **BaseProtocol** に入れる（`typical_steps` を JA / giving / dyadic が持つ。見守り 1.0・家事 5.0 は据え置き）。**実装済み（2026-08）**
2. 短い観測（信号注入、上の行、典型＋上限、冷／温）。**実装済み**（`cueworld/switch_eval.py`、結果は `results/3d_test/switch_frequency.json`）
3. 加点・二段くじ・維持を乗った証拠へ・観測行の付け替え。**実装済み（2026-08）**
   - 手がかりなし: 判定あたり P(parent)≈0.005（温でも同じ）
   - 手がかりあり: ≈0.032（冷の物視線は zeros と同じ。親の目は冷でも 0.032）
   - JA 未乗車／同じ物／リトライ: P≈0.119。乗車かつ別物: P≈0
   - 家事中の完成形: 割れない。期限到来: household
4. comprehensive で見た目。**実装済み（2026-08）**。問題なさそう。

comprehensive の幼児:
- 見守り中: 何もしない区間と、完成形を出す区間を混ぜる（家事の時計と、手がかりなしの親始動が見えるようにする）
- 関わり中: **忠実に付き合う / 動かず失敗 / 別の完成形を出し続ける** の3本（乗車／未乗車／同じ物の割れ方を動画でも見る）
- 今の `demo_comprehensive`（t>50 から合図を出し続ける）は使わない。拒否パターンは E に含めない（感情バッチ）

不採用:
- 短い観測を `demo_comprehensive` 1本の混合ランだけにする
- 短い観測で幼児の身体まで動かす（「数値が悪い」と「ジェスチャーが揃わない」が混ざる）
- 判定1回だけの intent テストを頻度の観測にする
- 家事も関わりも維持 5.0 のまま（関わり中の完成形を無視し続ける）
- 関わりの維持だけ下げて proximity 連動（後半ロックが状態粒度で二項に効かない）
- 全プロトコルを proximity 連動（見守りの proximity は最初から 1.0 で訴えが通らない）
- E で重み行列を本調整する

観測のあとで決めること（今は仮）:
- 3分／30秒／12%／0% への微調整（短い観測のあと）
- 重み行列
- 家事の典型 T の実測（仮 175）

##### 調整について
- 手がかりなしでどのくらいコミュニケーションプロトコルへ移行するべきか
  - 200stepでどのくらい移行するべき？
- 物を見ているときどのくらいのっていいのか

#### ⑦ 連鎖スモーク（単体では動くがつなぐと壊れる）

切替が実際に起きてから。独立プロトコルは作らない。**方針は下記「dig 確定（2026-08, バッチ F）」が正。**

- 欲しい姿: 同じおもちゃで見て振る → くれ。別の物へ行くときの手放しは保険
- 物体状態（把持・床位置・board 上）が次のプロトコルを殺さないこと
- 除外は「使ったら外す」ではない。（遊び, 物）の失敗／連続成功上限
- 合格: 振る→くれの強制デモ（人が動画）＋手放し・門・リストのテスト。comprehensive は目視

#### ⑧ 後回し

| 項目 | 理由 |
|------|------|
| 物体の音が同時に鳴ったとき | memo 明示。切替と独立。`base.py` |
| 遊びごとの長さの集計 | 4a のあとで指標として足せる。選択ロジックの前提ではない |
| 感情シグナルの本実装 | 0e。拒否は現行の視線・回避で足りる間は触らない |
| 泣き声を拒否信号に使う | §6-D2: 開始には載せない。単語でもバブリングでもない声は無視。拒否強化は感情バッチ |

#### バッチの切り方（実装単位）

| バッチ | 含む | 一緒にする理由 |
|--------|------|----------------|
| **A 準備** | ⓪の確認メモ化 ＋ 1a inspection オフ ＋ 1b 家事 target ＋ 1c observance `start` ＋ 1d デッド掃除 | 小さい・局所。家事を PM に載せるゲート |
| **B ソロ切替** | ②見守り↔家事 ＋ ③割り込みコストの最小実装 | ソロの出入りが一つの話 |
| **C 履歴** | ④バッファ（長さ・時刻・種類・終わり方含む。dig 済） | B の切替が起きないと履歴の意味が薄い |
| **D 自発性** | ⑤定義（0b/0c）のメモ化 ＋ 訴えから視線を外す ＋ 自発性 EMA のコメント直し。意図レイヤは作らない | 0b/0c 必須。部品の式は開けておく |
| **D2 意図部品（§6）** | ⑤b の dig（**確定**）＋ 意図レイヤ実装 | **実装済み**（`intent.py`、PM 行列、JA/giving 対象追随、テスト） |
| **E 頻度** | ⑥観測デモ／ログ → ノブ調整。**作業順 1–4 実装済み**（加点・二段くじ、維持は乗った証拠、短い観測、comprehensive 幼児 3 本）。重みは含めない | B–D2 のあと |
| **H 終わり方** | JA 成功に途切れない約3秒の対象視線、発話は必須にしない、終わり方を履歴に分ける。二項は待ちだけ5秒に合わせる。`infant_is_engaged` に改名済み。**方針確定・実装済み** | E のあと。1回の長さが変わると F の動画が古くなる |
| **F 連鎖** | ⑦ JA↔giving。開始時の手放し、親が持っている物のくれ門、除外リストの作り直し、board 配置。**方針確定（dig 済）・未実装** | H のあと（切替は実在。長さが先） |
| **G 後回し** | ⑧。全体の詳細 docs は F のあと | 触らない |

**次の実装単位はバッチ F**（方針は下の dig 確定。未実装）。重み行列は触らない。継続優遇（4c）は着手時に dig。拒否の本実装は触らない。

**dig 確定（2026-08, バッチ H）**:

単位: 1 step ≒ 0.1 秒。論文の「同じ物を一緒に見た」は約3秒。親が乗った注意の30〜50秒はおもちゃへの注意の区切りで、プロトコル1回にはしない（続きは 4c）。

成功と終了は別。成功は印。合図を下ろす印ではない。

成功（JA）:
- 芯は、幼児がその対象を **途切れない約3秒**（30 step）見ること。親始動 informative でも同じ。単語は門にしない
- 親はそのあいだ、物を見るか指差す／振る。顔の確認は成功したあと（頭だけ幼児へ）
- 手の合図（指差し／振る）は成功しても、終了かやり直しまで続ける。振り切りは成功条件にしない
- 目を外すと3秒カウントは 0。親の顔を見ているあいだは3秒は進まない（離脱タイマも進まない）

待ちと失敗:
- まだ一度も対象を見ていない → 合図開始から **5秒** 見なければ失敗側
- 一度でも対象を見た → **5秒** のあいだ対象も親の顔も見なければその回は終わり。5秒以内に戻せば続き（また途切れない3秒を貯める）
- 近づき中はまだ合図前。3秒／5秒は合図開始から
- 失敗したら合図を **1回だけ** 強くする。見るだけ → 指差し。すでに指差し／振るなら、同じ合図のまま発話をもう一度（振るは取りに戻らない）。対象は変えない。振るへは上げない。それでもダメなら失敗
- いまの `handle_timeout` 3段（やり直し → 強くする → 対象変更）は、この経路では使わない

発話:
- 親の発話は合図と **同時**（informative はラベル、sharing はほめ言葉）。幼児始動でも同じ。いまの「合図6秒のあと」はやめる
- 成功の門にはしない。見ている最中の足場
- 幼児が喋ったら、教える／共有どちらでも **一度だけ**、正解 → `praise_with_target`／惜しい（距離 2 未満）→ `recast_with_target`／違う → `correct_with_target`。間違えても失敗にもやり直しにもしない（単語が `retry_count` を消費するのはやめる）
- 成功後、約3秒だけ幼児の声を待つ。言わなくても成功のまま終了してよい
- 幼児が喋り始めたら、単語と同じ沈黙（`offset_silence_frames=4` ≒ 0.4秒）まで終了も割り込みもしない。音が止まらずほめ／リキャストにも進まないときだけ上限 **約5秒** で切る。ほめ／リキャストが始まったらこの上限は使わない

割り込み:
- **締めの発話中に限る**。成功後の `post_shared_attention` で、親の発話キューが残っているあいだは割り込まない（既存の completed 待ちと揃える）
- 合図中（`looking` / `pointing` / `shaking`）の発話では割り込みを止めない。合図と発話が同時になったため、合図中まで止めるとバッチ E の「未乗車 1回あたり約12%」が消える
- 成功後の声待ち中で、親のキューが空、幼児もまだセグメント中でない、なら割り込まれてよい
- 幼児がセグメント中は終了も割り込みもしない

履歴（C の `kind`＝`cue_type` は残す。H で足す）:
- 親の意図（`intent_type`: informative / sharing）
- 幼児が喋ったか
- 単語の当たり（正しい／惜しい／違う）。喋っていなければ空
- 感情反応したか（H では欄だけ。`emotion` は未出力。中身は感情バッチまで空。成功条件には使わない）
- 上は教える／共有のどちらでも同じ欄に書く
- 切替は今どおり JA を informative で始める。sharing の選択は 4c まで触らない
- `end_reason` は今どおり成功／失敗／割り込み

二項:
- 待ちと離脱を **5秒**、失敗したら合図を1回やり直す。発話中は切らない
- 成功は今どおり模倣完了／反応＋ほめ。3秒の途切れない視線は JA だけ
- 腕の合図のモータ長（約9秒）は待ちの5秒とは別

いまのコードとの差（直す穴）:
- 親始動 informative は単語で成功し、視線ホールドを見ない → 3秒視線へ
- `shared_hold_steps=50` は幼児始動と simple だけ → 全経路 30 step
- looking / pointing は固定 60 step で次へ、そのあいだ幼児の視線を見ない → 合図は成功／失敗／やり直しまで続ける
- 発話は合図のあと → 合図と同時
- 確認フェーズで親は顔を幼児へ（手が差したまま）→ 顔は成功後。成功前は物へ
- 間違い単語が `retry_count` を足して待ちを伸ばす → やめる
- sharing の感情成功は `emotion` 未出力で実質動かない → 成功は視線。感情は履歴欄のみ

触れない:
- 重み行列、家事維持 5.0、家事1回の長さ、継続優遇（4c）、感情の本実装、全体の詳細 docs（H 実装のあと）、board 位置（F）
- くれの成功は把持の瞬間のまま。待ち5秒と1回やり直しは二項のあとで足せる
- `typical_steps` の本更新は実装後の実測
- `simple`（発話なし）をデモ用に残すかは実装時。PM の既定は informative のまま

不採用:
- 1回を30〜50秒にする（単位が論文の共同注意とずれる）
- 成功を足し算の3秒にする（途切れない注視でなくなる）
- 成功も待ちも全部5秒（平均約3秒の共有が成功しにくい）
- 失敗ですぐ終わる／いまの3段梯子＋対象変更
- 指差し失敗を振るへ上げる（取りに行く時間が乗る）
- 発話を成功の門に残す（親始動 informative のいま）
- 合図が終わってから話す（Deák の同時ではない）
- 成功したら合図を下ろす（足場を切る。成功は印だけ）

**dig 確定（2026-08, バッチ F）**:

単位: 独立のオブジェクト遊びプロトコルは作らない。欲しい姿は同じおもちゃで見て振る → くれ。別の物へ行くときの手放しは保険。本番の続きは、振っている最中の割り込み（同じ物リーチで維持 2.7 → 判定あたり約12%。合図中は止めない）。見守りに戻ると物を使わないので手放し、床に落ちて届くとくれは始まらない。

F の前: H（実装済み）。4c・詳細 docs・拒否の本実装は前に置かない。

くれの門:
- 親が持っている物は、床 XY が幼児の到達圏内でもくれに乗せる（持っているあいだは幼児が自分で取れない）
- 親が持っておらず床に置いて届く物は、今どおりくれにしない

開始時の手放し（他の状態へ行かず、まずこれ）:
- くれ・振る JA・家事: 今回の対象と同じ物だけ持ったまま。違う物なら手放す
- 見る／指差し JA・二項: 物を持たないので、何を持っていてもまず手放す
- 見守りは今どおり腕リセットで指が開く
- やり方: 今の手の位置で指を開いて落とす。転がっても拾い直さない。手が空いたらすぐ本編
- 約3秒待ってまだ持っていたらそのプロトコルは失敗。見守りに戻る。`held_obj` をコードで消さない。`RELEASE_TIMEOUT_STEPS = 30`。eval_release_steps（seed 42–44）の把持解除は 1 step なので、30 は失敗判定の余裕

除外リスト（`recent_targets` の「使ったら外す」はやめる）:
- 単位は **（遊びの種類, 物）**。信号の生データは隠さない
- 入れる: 失敗終了（待ち切れ・取り損ね・拒否）。割り込みは入れない。拒否は今オフのままなので、実体は待ち切れと取り損ね
- 成功は入れない。同じ物・同じ遊びは **連続で成功 3 回**（仮）まで。間に別の物が挟まったら数え直し。睡眠（reset）でゼロ
- リストに入れた**その回では戻さない**。そのあと、家事を含むどれかが**成功して終わったら全部戻す**（失敗終了と割り込みは数えない。連続成功の数えも戻す）
- 切替: くじでは物を消さない。当たった遊びの対象がリストにある → その遊びには入らない。同じ判定で、対象がリストにない他の遊びがあればそちらへ。なければ今のまま。幼児が見ている物を別の物に差し替えない

board:
- 高さは変えない。幼児は歩かせない
- いちばん近い 2 枚（北側の西寄り）を南側の、幼児から遠い端へ。残り 2 枚は北
- 南のキューブは台と被らないようずらす
- くれの届く判定は高さを見ない。台そのものは対象外のまま

合格:
- 強制デモ: 親が対象を持って振り始めてから、くじを使わずくれへ切り替える。親は持ったまま。幼児は振っているあいだ対象を見て、置かれたら取る。そのあと幼児が振るところまでは入れない。履歴の JA は割り込み（失敗）。リストには入れない
- テストで固定: 開始時の手放し、親が持っている物はくれに乗せる、リスト（失敗で入れる／割り込みは入れない／連続成功 3 回で入れる／次の成功で戻す）
- comprehensive は目視

実装範囲:
- くれ／JA／二項／家事の開始時手放し状態
- くれ門（親把持は到達圏でも可）
- PM の除外リストを（遊び, 物）へ。信号マスクはグローバルにしない
- comprehensive / 家事デモの board 配置
- 振る→くれ強制デモ
- 上の穴のテスト

触れない:
- 重み行列、家事維持 5.0、家事1回の長さ、継続優遇（4c）、感情・拒否の本実装、全体の詳細 docs（F 実装のあと）
- くれの成功は把持の瞬間のまま
- リストの 3 回・手放し 3 秒の本数字（仮。測ってから）

不採用:
- 使った物を直近 2 個まとめて隠す（同じ物の続きが死ぬ）
- JA 用／くれ用にリストを分けるだけ（成功した JA の対象が JA リストに残り、くれのあとまた見る、が止まる。信号をまとめて隠すとくれにも見えない）
- 対象がリストにあるとき別の物に差し替える（幼児は A を見ているのに親が B を始める）
- 当たった遊びがリストでも他の遊びを試さない（くれを取りこぼす）
- リストに入れたその回の終了で戻す（すぐ消える）
- 失敗終了でもリストを戻す（失敗を重ねるだけで戻る）
- 開始時の手放しをくれの place と同じ「届く床へ置く」にする（二項の始まりがくれに見える）
- 強制デモを本番の 12% くじ待ちにする（動画がブレる）
- 成功して見守りを挟まずくれへ、を本番の切替にする（4c の先取り。デモもこの経路にしない）

実装済み（2026-08）:
- くれ門: `_in_infant_reach` は親が持っている物を False（幼児は自分で取れない → くれに乗る）
- 開始時の手放し: `BaseProtocol.__call__` 前段。`keeps_held_object` / `blocks_target_on_failure` / `open_hand`。上限 30 step（実測解除 1 step）
- 除外リスト: `blocked_pairs` + `success_streak`。`recent_targets` と信号マスクは廃止。失敗で追加、割り込みは入れない、連続成功 3 回で追加、リスト非空の成功で全戻し
- board: comprehensive / 家事デモ。北の西寄り 2 枚を南の東端へ。南のキューブは西へ
- 強制デモ: `cueworld/demo/demo_ja_shake_to_giving.py`

# 最終メモ
- 除外リストは **バッチ F で作り直し済み**（使ったら外す、ではない。`blocked_pairs` / `success_streak`）
- JA の終わり方と長さは **バッチ H で方針確定・実装済み**（上の dig）。成功は途切れない約3秒の対象視線。合図と発話は同時。`post_shared_attention` は成功後の顔確認・声待ち。
- 他のプロトコル: くれは把持の瞬間。二項は模倣完了／反応＋ほめ（H で待ちだけ5秒に合わせる）。幼児の継続秒数は JA 以外見ない。
- infant_is_on_board → `infant_is_engaged` に改名済み（環境の board と混同しない）
- 全体の実装が終わって，comprehensiveをチェックして問題がなさそうなら全体の詳細なドキュメントを作る → comprehensive は問題なさそう。詳細 docs は **F の実装のあと**。
- board の位置は **バッチ F で実装済み**（高さ据え置き。近い 2 枚を南側の遠い端、残りは北。幼児は歩かせない）
- 連鎖（JA↔giving）は **バッチ F で方針確定・実装済み**（上の dig）
- 単語は成功条件ではない。反応は一度だけ。retry_count は増やさない。というのはいいのか？例えば，recastした後など．
- 幼児のプロトコル拒否が結構悪い動きをする．現状OFFにしてる．拒否の反応が起きやすすぎる．もし使うにしても，もう少し判定を厳しくする必要がある．目をそらす+泣き声必須など．


