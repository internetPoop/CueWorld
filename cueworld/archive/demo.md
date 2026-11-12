# コミュニケーション環境 デモ実装仕様書 (demo.md)

本ドキュメントは、親と幼児のコミュニケーションシミュレーション環境（`base.py` 基盤）の能力とウリを視覚的にアピールするためのデモシナリオおよびその実装仕様をまとめたものです。

---

## 1. デモの目的とウリ（アピールポイント）

この環境の最大のウリは、**「スクリプトで決められた親の動作」ではなく、「幼児のマルチモーダルなシグナル（視線・ジェスチャー・音声）を環境が解釈し、親が自律的に意思決定（プロトコルの動的切り替え）を行って物理的インタラクションを実行する」** という一連のパイプラインが完成している点です。

これを最も分かりやすく見せるために、**「幼児の要求シグナルによる『手渡し（Object Giving）』の創発デモ」** を作成します。

### 見どころ
1. **マルチモーダルシグナルの統合**: 幼児が「対象物を見る」「手を伸ばす/指差す」「声を出す」という複数のモダリティを組み合わせた行動をとる。
2. **動的なプロトコル割り込み**: 親が「見守り（Observance）」状態から、幼児のシグナル強度（`infant_signal.py`）を検知し、自律的に「手渡しプロトコル（ObjectGivingProtocol）」に切り替える（`protocol_manager.py`）。
3. **物理インタラクション**: 親が対象物（例：赤いブロック）を認識し、IKを用いて手を伸ばして把持し、幼児の元へ運んで手渡す一連の物理的動作（`parent_tools.py`）。

---

## 2. デモのシナリオとタイムライン

親のアクションは `ProtocolManager` が完全に自律制御するため、デモスクリプト側では **幼児のアクションのみを時間（ステップ数 $t$）に応じてハードコード** してシナリオを進行させます。

*   **フェーズ 1: 見守りと待機 ($t = 0 \sim 50$)**
    *   **幼児**: 何もしない（アクションベクトルは全て0）、またはランダムに周囲を見渡す。
    *   **親 (期待される動作)**: `ObservanceProtocol`。幼児を見守り、待機する。
*   **フェーズ 2: 強い要求シグナルの発信 ($t = 51 \sim 150$)**
    *   **幼児**:
        *   **視線**: 特定のオブジェクト（例: `RedCube`）の方向を向き続ける（`demo.py` の `_get_agent_look_at_object_action` を流用）。
        *   **ジェスチャー**: 対象物に向かって腕を伸ばす（肩・肘の関節を制御）、または手を広げる（Reaching）。
        *   **音声**: 断続的に `voice` をオンにする。
    *   **親 (期待される動作)**: `infant_signal` の要求強度が閾値を超え、`protocol_manager` が割り込みを判定。`ObjectGivingProtocol`（または `JointAttentionProtocol`）を開始する。
*   **フェーズ 3: 親の応答と手渡しの実行 ($t = 151 \sim 400$)**
    *   **幼児**: 自分の要求が通ったため、対象物または近づいてくる親を見る。
    *   **親 (期待される動作)**: 対象物に近づき、IKで手を伸ばして掴み（`grasp`）、幼児の元へ運んで手渡す（`hand_over`）。完了後、再び見守りに戻る。

---

## 3. 実装のステップ（AIへの指示書）

AIに以下の手順を指示することで、デモスクリプト（例: `scenario_demo.py`）を実装できます。

### Step 1: 環境とラッパーの初期化設定
`demo.py` を参考にしつつ、以下の順番で環境をラップして初期化する。
1.  `BaseEnv` を作成（オブジェクトとして `RedCube` 等を含める）。
2.  `env = InfantSignalDetector(env)` でラップ。
3.  `env = ProtocolManager(env)` でラップ（`demo_mode=False`。※もしシグナル強度の閾値調整が難しければ、デモ用の確実な遷移を保証するために `demo_mode=True` のロジックを修正して使用してもよい）。

### Step 2: 行動ループの構築
`BaseEnv` の `step` は `(infant_action, parent_action)` のタプルを要求する点に注意する。
1.  **幼児のアクション**: 上記のシナリオタイムライン（$t$ の値）に応じて、幼児のアクションベクトルを IF 文で生成する。
2.  **親のアクション**: 前のステップの `info` に格納されている `info["parent_action"]` を使用する（初回は全て0のベクトル）。`ProtocolManager.step()` は実行後に「次の親のアクション」を `info["parent_action"]` に格納する仕様であるため、このループ構造を利用する。
3.  `obs, reward, done, trunc, info = env.step((infant_action, parent_action))` を実行。

### Step 3: 注釈（アノテーション）の描画
`demo.py` にある `annotated_view_from_info` 関数を流用・改修し、親が今何を考えているか（内部状態）を動画にオーバーレイ表示する。
*   表示すべき情報:
    *   現在のプロトコル: `env.current_protocol_name` (ProtocolManager のプロパティ)
    *   現在のターゲット: 対象オブジェクトのIDや単語
    *   幼児のシグナル強度: `infant_signals` の一部（視線安定性や要求の強さ）

### Step 4: 動画の保存
シミュレーション終了後、収集したフレーム群を `images_to_video` を用いてMP4として `results/` ディレクトリに保存する。

---

## 4. その他のデモシナリオ案 (実装バリエーション)

この環境の多様なアフォーダンスやプロトコルをアピールするための追加シナリオ案です。上記の実装ステップの「幼児のアクションスクリプト」と「対象オブジェクト」を変更するだけで作成可能です。

### 案A: お片付けと自発的行動 (Object Play & Parent Initiative)
*   **概要**: 幼児が何もしていない時、親が自発的におもちゃを箱に片付けるデモ。
*   **見どころ**: `check_for_initiative` による親主導の自律的タスク開始と、`put_in_object`（コンテナへの収納）の物理演算・アフォーダンス（`can_contain`）のアピール。
*   **幼児のスクリプト**: 終始ランダムに周囲を見渡すか、おとなしくしているだけ。
*   **親の動作**: 待機状態から自発的に `ObjectPlayProtocol` を開始。床の `Lego` や `Ball` を拾い、`ContainerOpen` の上まで運んで落とす。

### 案B: 音声模倣のインタラクション (Babbling Mimicry)
*   **概要**: 幼児が特定の喃語（バブリング）を発し続け、それを親が真似して返すデモ。
*   **見どころ**: 単なる物理移動だけでなく、`LanguageManager` と `voice` 空間を用いた音素のやり取り、および `DyadicProtocol` による音声的コミュニケーションの成立。
*   **幼児のスクリプト**: `voice`, `volume`, `v_1`, `v_2` などの音声アクションを特定の値で連続的に出力し続ける（例: 「あー」という音）。
*   **親の動作**: `infant_signal` が `babbling_target` を抽出し、親がそれに反応して同じ音素系列を返すように発声する。

### 案C: 親からの共同注意と指差し (Parent-Initiated Joint Attention)
*   **概要**: 親が特定のおもちゃ（ミニカーなど）を見つけ、幼児に「見て」と注意を促すデモ。
*   **見どころ**: テンプレートを用いた発話生成（「あれはクルマだね」など）と、対象物を正確に指し示す IK（指差し `point_object`）の連携。
*   **幼児のスクリプト**: 最初は無関係な方向を見ているが、親が指差しと発話をした後（フェーズが切り替わった後）に、親が指している対象物へ視線を向ける。
*   **親の動作**: 自発的に `JointAttentionProtocol` を開始し、対象物に近づいて指差しポーズを取りながら、音声キューを発する。

### 案D: 摂食の手渡し (Feeding)
*   **概要**: 幼児が食べ物（リンゴやバナナ）を欲しがり、親がそれを手渡しではなく「口元へ」運んであげるデモ。
*   **見どころ**: 同じ「物を渡す」動作でも、対象物が食品（`is_food`）である場合やプロトコルの設定（`feeding=True`）によって、IKの目標座標が「幼児の手」から「幼児の顎（`jaw_link`）」へと動的に変化する柔軟性。
*   **幼児のスクリプト**: `Apple` を注視し、手を伸ばして要求シグナルを出す。
*   **親の動作**: `ObjectGivingProtocol (feeding=True)` が発動し、リンゴを掴んで幼児の顔の前（口元）まで運ぶ。

### 案E: 家事への集中と割り込み拒否 (Household Solo & Persistence)
*   **概要**: 親が洗濯機（`WashingMachine`）や冷蔵庫（`Refrigerator`）を操作している最中に幼児が弱く要求するが、親が家事を優先するデモ。
*   **見どころ**: 常に幼児に反応するわけではなく、現在のプロトコルの「持続強さ（Persistence）」によって、弱い割り込み要求を確率的に拒否（Reject）する人間らしい（自律的な）意思決定の様子。また、ドアの開閉（`open_type="rotate"`）などの大型家電とのインタラクション。
*   **幼児のスクリプト**: 親が家事を始めている最中に、少しだけ親の方を見て声を出す（弱いシグナル）。
*   **親の動作**: `HouseholdSoloProtocol` を実行し、洗濯機の蓋を開ける。幼児のシグナルが入っても、要求強度が持続強さを上回らないため無視して家事を続ける。

### 案F: 二項モータ模倣 (Dyadic Motor Imitation)
*   **概要**: 幼児のモータ（掲げキープ・手開閉・音声+掲げ）を親が模倣する、または親が固定／ランダム cue でお手本を見せ幼児が応答するデモ。
*   **見どころ**: `infant_signal.py` のヒット率・開閉周期性と `DyadicProtocol.compute_channel_salience()` / `get_salient_channels()` による独立ゲート、親側の掲げ／開閉模倣、応答 match 注釈（チャネルごと成立 / `response_matched`）。
*   **スクリプト**: `cueworld/demo/demo_dyadic.py`（三人称カメラ付き）。旧 `demo_imitation_motor.py` も同キー `dyadic` で動作。
*   **幼児のスクリプト**: 他デモと同様 `ParentTools` を用いて幼児エージェントにモータを適用する（`gaze_agent` で親を見つつ、掲げキープ、または掲げ + `animate_hand_open_close`）。親始動の開閉格子では `--intensity` が `amplitude_scale` になる。`arm_raise` 正例は振らない。
*   **親の動作**: `DyadicProtocol` が `waiting_infant` で salience を検出し `imitating` へ遷移、または親始動時は `performing_parent_cue` → `waiting_infant_response`。親始動の `arm_raise` は見せ振り可。幼児始動の親は振らない。

**強度しきい値 τ（目視後の編集場所）**: `cueworld/protocols/dyadic.py` の `MATCH_QUANTITY_THRESHOLDS`（`arm_raise=0.8`、voice / open_close は当面 0.2）。

**実行例（コンテナ内 `/cublex`）**

```bash
# 幼児始動: 掲げキープの模倣（infant + third）
python3 cueworld/demo/demo_dyadic.py --scenario arm_raise --views infant,third

# 幼児始動: 掲げ + 手開閉
python3 cueworld/demo/demo_dyadic.py --scenario hand_open_close

# 幼児始動: バブリング + 掲げ
python3 cueworld/demo/demo_dyadic.py --scenario voice_arm_raise

# 親始動: ランダム 2 スロットのお手本
python3 cueworld/demo/demo_dyadic.py --scenario parent_cue

# 親始動格子: 掲げキープ（強度なし）
python3 cueworld/demo/demo_dyadic.py --scenario parent_arm_raise --views infant,third

# 親始動格子: 掲げ + 開閉 × 強度
python3 cueworld/demo/demo_dyadic.py --scenario parent_hand_open_close --intensity 0.5 --views infant,third
python3 cueworld/demo/demo_dyadic.py --scenario parent_hand_open_close --intensity 0.25

# 親視点も保存
python3 cueworld/demo/demo_dyadic.py --scenario arm_raise --views infant,parent,third

# 格子 CSV（動画なし）
python3 scripts/experiment/eval_dyadic_response_match.py
python3 scripts/experiment/eval_dyadic_response_match.py --seeds 42,43,44
```

動画は `results/3d_test/demo_dyadic_<scenario>.mp4`（および `_parent` / `_third`）。
親始動格子は `demo_dyadic_parent_<channel>_i<100|050|025>.mp4`（例: `demo_dyadic_parent_hand_open_close_i050_third.mp4`）。
CSV は `results/3d_test/eval_dyadic_response_match.csv`。
