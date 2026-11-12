# 次にやること

2026-11-04 時点。正本は `cueworld/docs/`、作業メモの経緯は `cueworld/todo.md`。ここは**まだ残っている修正・実装**だけを書く。

`docs/` は「実装されている振る舞い」の正本なので、このリストはそこへ入れない。

## いま終わっていること

`base.py` のエピソード契約（幼児発声の reset、`is_terminal`、trackable のロード後 1 回化）、touching の正本、sleepiness / 関節角 / 触覚観測、hunger 廃止と選好報酬、fatigue の口、円錐の 8° リング、手形 intensity の向き、単語棄却の公開（不採用は距離無限大・`words` 空、共同注意の「すごいね」／「ちがうよ」）、JA 成功の瞬間視線（`gaze_targets_now`。平均は選択と `infant_is_engaged` に残す）、共同注意セッション内の言葉の往復（最大 5、維持 −1.0）、死キー削除（握り数値・手の動き・うなずき周期性・身体距離一式・腕の伸び・声の持続割合。拒否の逃避も外した。首そむけと声の窓は残す）。これらはコードと `env.md` / `infant_signal.md` / `protocol/joint_attention.md` が一致している。ここから入らない。

## 着手順

契約が嘘のまま新シグナルを足さない。落下速度と `place_margin` は最後。

1. 新シグナルは掲げ続け（2.2）だけ。声の感情と表情は感情モードまで着手しない（§6）
2. 拒否の再設計（`rejection=True` で測ってから。既定はオフ）
3. 親始動が訴えより先に立つかの検証
4. ログと観測宣言の掃除
5. チューニング　　　　　

---

## 2. 要実装（契約は todo / infant_signal.md にある。コードは無い）
### 2.2 持って掲げ続け（5a）

検出器が「その step で holding かつ目線帯」だったかのヒット率を、`hand_height_margin` と同じ 10 step で 1 キー出す。高さの窓は増やさない。このキーの上に連続カウントを積まない。

二項の `arm_raise` は持っていない掲げのまま。これは持っている掲げ（見せる）。

### 2.4 声の感情（5d） / 2.5 表情（5e）

着手しない。感情モードがオフのあいだは、声の感情も表情もキーにしない。方針と、オンにするときに初めて触る範囲は §6 の感情モード。

### 2.6 名前

`point_intensity` / `open_intensity` は値が強度になったあとでも名前はそのまま。改名は 2.2 のあと。2.4 / 2.5 は待たない。intent は `> 0` しか見ない。


---

## 3. 要検証（直す前に測る）

### 3.1 拒否は既定オフ。経路は残す

`ProtocolManager(rejection=False)` が既定。JA / dyadic / giving の `detect_rejection` は、`rejection=True` のときだけ効く。視線を有効対象から外したうえでの首振りは、そのとき失敗終了する。逃避（親から離れる速さ）は消した。情動サージは常に 0 なので発火しない。見守りと家事は呼ばない。

閾値を変える前に、`rejection=True` で通常のリーチや首振りが何 step に 1 回落ちるかを見る。再設計の対象は視線拒否の 2 段判定と、首そむけの記録（短い行動ベクトルだと左右と上下を足し違える）。`J1_GAZE_REJECTION_HOLDING_FALLBACK` は `False` のまま残している（把持中の誤拒否が再発したときの逃げ道）。

### 3.2 視線の手がかりで親始動が先に立つか

`todo.md` の疑問。判定の順序自体は訴えが先である。

`ProtocolManager.update` は `check_interval`（既定 10 step）ごとに、

1. `check_for_response`（指差し・リーチ・単語・二項・弱い腕）
2. 見守り中だけ家事の時計
3. それでも見守りなら `check_for_initiative`

同じ判定tickでは、訴えば親始動より先に遊びになれる。親始動は見守り中かつ訴えが maintain のときだけ。

ずれうるのは中身である。親始動の手がかり `parent_object_cue` は「視線の安定 ≥ 0.6、または触る、または持つ」。`parent_dyadic_cue` は「親の目を見ている割合 ≥ 0.6」。どちらも指差しやリーチより先に立つ。手がかりがあると開始側のスコアが 0.36 になり、待ち 1.04 に対しておよそ 30 秒に 1 回始まる（手がかりなしはおよそ 3 分）。

つまり「視線が出た瞬間に response が死ぬ」のではない。見守り中に、揃った訴えができる前に親が遊びを始めうる。頻度はくじなので、短いロールアウトで「initiative ばかり」かどうかを数える。数値が想定と違えば、手がかりから視線安定を外すか、開始スコアを下げる。コードの優先順位は変えない。

### 3.3 プロトコルインスタンスの使い回し

`self.protocols` の 5 インスタンスを切り替えのたびに `start(context)` で再利用している。選ぶ前に `level` / `request_type` / `has_target` を読むので、選択後の新規生成にはしない。

リスクは `start` で戻していない属性が前回値を引くこと。直す対象は「生成タイミング」ではなく、`start` のリセット漏れ。新しい状態をプロトコルに足したときは `start` を確認する。

---

## 4. 掃除（挙動は変えない）

- 振る舞いの正本が喃語の旧定義のまま。`infant_signal.md`（`babbling_target` / `babbling_ratio`）、`protocol/dyadic.md`、`protocol/automata/dyadic.md`、`protocol_manager.md` の `target_phoneme`。実装と契約は `babbling_bout.md`。合わせる。
- 標準出力が毎判定・毎把持で出る。`check_for_response` / `check_for_initiative`、`grasp_object` の `graspable_obj`、家事の `select_next_item`。長い学習ではログが支配する。フラグの後ろへ移す。
- `observation_space` の 2 要素目は長さ 0 の `Box`、`reset` / `step` が返す 2 要素目は `None`（親の観測は未実装）。宣言を `None` に合わせるか、返す側を空 Box に合わせる。
- `success_state` の既定は `"completed"` だが、タイムアウトも `"completed"` に落とすプロトコルがある。評価は `success_flag` だけでなく `history_extra()` を見る、で足りる。名前の統一は急がない。
- `urgency` は全プロトコル 0。緊急度の分岐は無い。
- カタログ外の `PhysicalContactProtocol` / `ObjectPlayProtocol` / `DyadicPlayProtocol`、容器の contain / open、`hand_over`、`easy_voice=False` は現行の 5 プロトコルから呼ばれない。消すなら語彙のハミング距離（`language.md` §9）に注意。未使用語を消すと他単語の距離条件が変わる。

---

## 5. 最後に触るチューニング

バグ修正と混ぜない。

- 落下: `fall_per_step` 既定 0.01。実験口は 0.05 / 0.10 / 0.22。着地後の視線、giving、3 step 以内を見てから既定を決める。
- `place_margin = 0.1`。`place_dist = reach_radius - place_margin` は giving と JA の立ち位置・置き位置の正本。変えるとプロトコルの幾何が全部ずれる。

---

## 6. 今はやらない

意図として残すか、観測の形が固まってから。

### 感情モード（`enable_emotion`。既定オフ。ゲートは実装済み）

`BaseEnv(enable_emotion=False)` が既定。フラグと、オフのときの次元・拒否は入っている。声の感情の分類や表情のキー本体は、オンにしたときが初めての実装時点である。

オフのあいだ（実装済み）:

- 行動から `lip` と発声 `pitch` を除外（15 次元）。発声観測は `volume, v_1, v_2, v_3`（4 次元、`phoneme_index = 1`）。`joint_angles` から `lip` を除外（12 次元）。fatigue から `lip` を除外（9 次元）。
- 表情の更新と出力をしない。pitch 履歴も積まない。
- 共同注意の `intent_type="sharing"` は `start` で `ValueError`。使うのは `informative` と、発話しない `simple`。二項・受け渡し・単語成功の「すごい」は残す。
- 拒否（視線・首そむけ）と唇リンクの `face_links` 判定は対象外。そのまま。

オンにするときに初めて触るもの（未実装）:

- 声の感情。volume と pitch からの分類、`emotion_prototypes`、拒否の `negative_emotion_surge` への接続（最初はキーを出して観測するだけ）。
- 表情。`get_expression_signals`、唇からの smile / distress。6 分類ダミーと同じラベルは声側へ移す。
- `sharing` の笑顔の枝（いまは空。`joint_attention.md` §7）。

いまコードに残っているものは、オンにするまでの置き場である。

- `emotion_prototypes` は書いてあるが未接続。音素認識に pitch は使わない。
- `get_smile_intensity` / `get_distress_intensity` / `get_surprise_intensity` は `left_lip_joint` の符号違い。呼び出しもキーも無い。`get_face_expression` は常に `(0, 0.0)`。`get_expression_signals` は未定義。
- Protocol Manager の共同注意起動は `informative` 固定。`emotion_reacted` は常に `None`。

コードの方針として、今は触らない。

- 物理エンジンを `step` で進めない。衝突なし。片腕。把持を簡単にしてある。テーブルなし。
- 親の発声は上書き、幼児は積算。データセット再生のための非対称。揃えない。
- `parent_position` は親が有声のときだけ入る。
- 有声閾値 0.25。0 にはしない。
- `household_items_by_category` から把持物を外さない。候補側の `_is_eligible_household_item` が外している。
- 親胴体が幼児に近づきすぎない制限。touching のあとの新機能。
- agent 辞書をクラスにする全面書き換え。
- 途中再開。保存候補は step_count、sleepiness、発声、held_obj、物体 pose、関節。実装は後。
- プロトコルの `reward_function` を `reward[0]` に足すこと。いまは `info["protocol_reward"]` のみ。学習信号は選好報酬（把持 × `preference`、その step に親の目を見た × `parent_preference`）。切り分け用の密な報酬は別スイッチで、既定オフ（§10）。

---

## 7. 幼児の単語認識（位相と、無音を待たない確定）

実装済み。正本は `language.md` §2 / §4–7 と `infant_signal.md` §履歴長 / §6 / §9。確認は `cueworld/tests/test_voice_recognition.py`、`cueworld/tests/test_word_queue.py`、`docs/voice_recognition_report.md`。

やらないもの（1.5 倍に伸ばした連続発話、別音素 2 フレームの早い誤確定、次点への取り替え）は `memo.md` とレポートに残してある。

---

## 8. 喃語と二項のセッション（5c）

実装済み。契約は `babbling_bout.md`。確認は `cueworld/tests/test_babbling_bout.py` と `cueworld/tests/test_dyadic_protocol.py`。

喃語は、沈黙で区間が閉じたとき残った 3〜6 音で、どの単語にも採用されなかった列。声の salience は列の保持中だけ 1。二項は 1 音 6 step で 1 回返し、最大 5 往復。親が先に出す声は `p0 p1 p0 p1` を 1 回。Protocol Manager への書き込みはセッション終了の 1 回で、成功は `session_success`。共同注意と受け渡しも同じ書き方。往復ごとの維持 −1.0 は二項と共同注意。受け渡しには写さない。

振る舞いの正本をこの実装に合わせる作業は §4 に残してある。

---

## 9. 共同注意のセッション

実装済み。契約は `joint_attention_session.md`。正本は `protocol/joint_attention.md`。確認は `cueworld/tests/test_ja_session.py`。

成功フラグは最初の視線ホールドのまま、そのあと言葉の往復を最大 5 回。維持は往復ごとに −1.0。


## 10. 切り分けテスト

見るだけ共同注意の口は実装済み。合意は `ja_looking_isolation.md`。確認は `cueworld/tests/test_ja_looking_isolation.py`。既定は本番のまま（密な報酬オフ、合図は強める、成功後も言葉の待ち、配置は毎回ずらす、観測は全キー）。

受け渡しの口も実装済み。合意は `giving_isolation.md`。確認は `cueworld/tests/test_giving_isolation.py`。既定は本番のまま（密な報酬オフ、掴み後も称賛、掲げと取り換えあり、親は発話する）。

二項の掲げの口も実装済み。合意は `dyadic_raise_isolation.md`。確認は `cueworld/tests/test_dyadic_raise_isolation.py`。既定は本番のまま（密な報酬オフ、成功後も持続の真似と喃語待ち、配置は毎回ずらす、観測は全キー）。

音声の口も実装済み。合意は `voice_isolation.md`。確認は `cueworld/tests/test_voice_isolation.py`。既定は本番のまま（密な報酬オフ、下駄オフ）。固定プロトコル `voice_echo` で、親が見本の 1 語を繰り返し、幼児の認識コストを密な報酬にする。親の返事は出さない。

入っているスイッチ:

- `ProtocolManager(dense_reward=True)`。学習報酬は、プロトコルの `dense_reward_function` だけ。選好は足さない。オフのとき密な項は 0。古い `reward_function` は `info["protocol_reward"]` のまま。
  - 共同注意: 合図開始から視線成功までの角度差（180° を 1）と、成功の瞬間の +2。
  - 受け渡し: `waiting_infant` から掴み成功までの指標 `e = d × (1 + (1 - cos θ))` の減少分（1 m を 1）と、掴みの瞬間の +2。
  - 二項の掲げ: 見本開始からヒット率 0.8 までの不足 `h`（帯の下端 − 手の高さ、0.4 m を 1）の減少分と、成功の瞬間の +2。
  - 音声: 判定 step の見本語コストから `1 − (コスト − c*)`。c* は下駄なし 0、下駄あり −1。完全一致は +1。
- 共同注意の `start` context。`lock_cue` は時間切れでも合図を変えない。`close_on_success` は視線成功でセッションを成功のまま閉じ、エピソードも終わる。
- 受け渡しの `start` context。`close_on_grasp` は掴み成功でセッションを成功のまま閉じ、掲げと取り換えをせず待ちを 2 回で失敗にする。このテストでは `parent_utterance=False` も併用する。
- 二項の `start` context。`close_on_raise` はヒット率 0.8 でセッションを成功のまま閉じ、見本終了でも未達なら失敗で閉じる。このテストでは `parent_channels=["arm_raise"]` も併用する。
- 音声の `start` context。`use_geta` は下駄の有無。`target_concept` で見本語を固定できる（無指定なら物の 20 語を辞書順に巡回）。`max_rounds` は見本の回数（既定 5）。
- `BaseEnv(layout_freeze=)`。`none` / `full`（位置と対象を固定）/ `rotate_target`（位置は固定、対象だけ順に変える）。
- `BaseEnv(infant_obs_keys=)`。指定したキーだけを観測と `observation_space` に残す。行動の絞り込みは既存の `infant_action_names`。
- オラクル観測。合意は `oracle_isolation.md`。確認は `cueworld/tests/test_oracle_isolation.py`。`infant_obs_keys` に `oracle_objects` / `oracle_parent` / `oracle_cues` があるときだけ、画像の代わりに物・親・合図の数値を渡す（本番既定には出さない）。

まだ残っているもの:

- この口を使った学習そのもの。最初にどの配置モードで走らせるかは未決（共同注意も受け渡しも二項の掲げも）。音声は下駄あり／なしのどちらを先に走らせるかが未決。学習器に幼児の声だけを見せるラッパーは未実装（`voice_isolation.md` の「学習器向けのラッパー」）。
- 指差しから始める共同注意、掲げて見せる共同注意の、密な報酬の中身。
- 親の返事を観測に載せる間接的フィードバックの音声テスト。
- オラクル観測を使った学習そのもの。合意と口は `oracle_isolation.md`（実装済み。確認は `test_oracle_isolation.py`）。どの切り分け試験から画像を置き換えるかは未決。
- その他。この環境に説得力を持たせるために必要な機能があるかは要検討。
