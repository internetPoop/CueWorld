# base.py
## 観察
- 環境内でdoneになったらそこでis_terminal=Trueにする
- sleepinessも観察に必要
- 欲求やfatigueを入れたらそれも必要かも
- 自分の関節角度も必要では？
- 自分の掴んでいる，触っているものの触覚情報もほしい(掴んでいると触っているのフラグになるものもほしい)
## その他
- docを読んで，注意事項に注目．その注意事項が意図したものかどうかをチェックする．(これをすべてのドキュメントに対して行う)
- hungerいるかな？いらない気がする．hungerではなく，欲求ベクトルみたいなものを用意するのがいい気がする．(複数次元を持ち，各オブジェクト種類に対応する．選好や欲求になる．)
- fatigueの実装をするべきかも．
- 幼児エージェントのtorsoに親エージェントのtorsoが近づきすぎないように制限する機構がほしいかも
- 親と幼児の音声の定義を別にしたのやめたほうが良かった可能性ある？？幼児の感じにそろえるべき？
- この環境にもし，途中再開機能をつけるなら，何を定期的に保存しておくべきか
- volumeの有しきい値を0.25でなくて0にしても良いのでは？
- touchingの経路がbase.pyとinfant_signal.pyで食い違っている．揃えたい．多分そのものを掴める範囲に手があることをtouchとするべきだと思う．
- graspの判定の際に角度を無視できる距離があるが，これはもう少し大きくてもいいのでは？(grasp_margin=0程度の距離ならありにしてもいいような)
- 簡易物理の落下の速さが遅すぎでは？1stepで0.22とかでも良い気がする．(1.0mの高さを落下する物体の速さの平均を0.1秒換算したもの)
- 毎ステップget_all_trackable_targetsを読んでいるが，その必要性はない．load_urdfの後で1回呼ばれるだけでいい．
- household_items_by_categoryでは，幼児の掴んでいるオブジェクトを除外していない．これを除外するべきか考える．
- place_margin=0.1はでかい気がする．というかplace_marginって必要かな？置くのはreach_radius基準でいいのでは？
- 幼児の発声がreset時にresetされるようにしたい
- 欲求ベクトルを追加したら，それに合わせて報酬を定義する．プロトコル内でも，報酬を定義して使えるようにする．(親を見ることも報酬にするか？親関連の基本的な報酬もbase.pyに書く)
- agentを辞書でなくクラスにするべき？？

## 着手順（base.pyをどれから直すか）

方針: **エピソード契約のバグ → touchingの正本 → 観測の拡充 → 設計判断 → チューニング**。欲求・疲労・報酬・クラス化から入らない。それらは観測とresetの形が決まってから。infant_signal.py の touching / holding は base 側の公開APIに依存するので、baseが先。

### 0. コードを触る前に仕分け(完了✅️)
env.md §13 の注意事項を「バグとして直す / 意図として残す / 後回し」に分ける。下の項目の多くはこれと一致している。ここを飛ばすと、親音声の上書き vs 幼児の積算のような意図した非対称まで潰す。

直す寄り（§13が明示）: 幼児発声がresetで戻らない、`is_terminal`未接続、観測空間の宣言と返り値のずれ、hungerが死んでいる。
残す寄り: 物理エンジンを進めない、衝突判定なし、片腕、把持を簡単にしてある。親→幼児しかhandoverしないのも、現行はplace方式なので急がない。
要判断: volume閾値0.25、`parent_position`が発声時だけ、音声の親子非対称。

### 1. 最初に直す（小さい・依存が少ない・学習の前提）(完了✅️)
1. **infant_voiceをresetで0に戻す**（`infant_voice_window`も空に）。§13のバグ。エピソード独立性が壊れている。1箇所。
2. **is_terminalをdoneに接続する**。`_get_obs`が常にFalse。resetでは`is_first=True, is_terminal=False`、stepでは`is_terminal=done`。`is_terminal=done`となるようにしたい．
3. **get_all_trackable_targetsをload後1回だけにする**。body idはロード順で不変（env.md §11）。独立で低リスク。ウォーミングアップ向き。

### 2. その次（infant_signal.pyの前提なのでbaseが先）(完了✅️)
4. **touchingの正本をbase.pyに置く**。`detect_grasp_object` に一本化し、`graspable_obj` / `graspable_link` を agent に公開。infant_signal はそれを読む。`is_graspable=False` は検出・距離計算から除外。`grasp_contact_distance` で角度免除。

### 3. 観測の拡充（2の後）(完了✅️)
5. **sleepinessを観測に入れる**。is_terminalとセット。終了の前兆。
6. **自分の関節角度**（自己受容）。行動が変位なので、状態として必要。
7. **掴み/触れのフラグと触覚**。4に依存。holding / touching の正本は base の `held_obj` / `graspable_obj`（シグナル側は既に接続済み）。観測は `tactile` + `holding`（`touching` キーなし）。

hunger / fatigue / 欲求を足すなら、この観測の形が固まってからキーを増やす。

### 4. 設計判断が先、実装は後
- hunger廃止・選好報酬・fatigue（完了✅️）: hunger 削除。欲求ベクトルは作らず、物体クラス `preference` + `parent_preference` で選好報酬（Detector）。fatigue は幼児のみ、`enable_fatigue` 既定 False。詳細は env.md §6.1 / infant_signal.md §10。
- 親/幼児の音声定義を揃えるか。§5の非対称（親は上書き、幼児は積算）はデータセット再生用の意図の可能性が大きい。安易に揃えない。
- agent辞書→クラス。infant_signalのholdingバグの温床だが、全面書き換えになる。holdingはinfant_signal側の`agent.get("held_obj")`修正で先に直せる。クラス化は触覚公開のあとでも遅くない。
- 親胴体が幼児に近づきすぎない制限。新機能。touchingより後。
- 途中再開で何を保存するか。リストアップだけなら今でも可（step_count, sleepiness, 発声, held_obj, 物体pose, 関節）。実装は後。
- household_items_by_categoryから把持物を除外するか。`household_task_candidates`は既に`_is_eligible_household_item`で除外している。辞書自体は「その色のactive item」の索引なので、除外しない方が正しい可能性が高い。reset時点では誰も持っていない。

### 5. チューニング（挙動が安定してから。バグ修正と混ぜない）
- volume: 状態・観測とも `volume` は `[0, 1]`。有声は全層 `>= voice_volume_threshold`（既定 0.25）。閾値 0 案は不採用（完了✅️）
- graspの角度無視距離（`1e-4`→`grasp_contact_distance`）。4で実装済み。
- 落下: 配置 z = `bbox_half[2]`、`fall_per_step` 実験口あり（既定 0.01）。デモで 0.05/0.10/0.22 を試してから既定を決める（着地後視線・giving・3 step 以内）
- place_margin=0.1。`place_dist`はgiving/JAの立ち位置・置き位置の正本。変えるとプロトコル幾何が全部ずれる。最後。

### この順にしない理由
- 欲求ベクトルや報酬から始める → 観測・reset・touchが動く標的になっていない。
- agentクラス化から始める → 全面書き換えに上のバグ修正が埋もれる。
- place_margin / 落下速度から始める → プロトコルの幾何がずれ、バグ修正と切り分けできない。

# infant_signal.py
- holdingのgetattrバグは§2で修正済み
- 8°リングが関与しないのは問題
- point_intensity / open_intensity とaccuracyで逆になっているので揃える．(base.pyでaccuracyではなくintensityを取得するようにする)．指先の接触は使用しない．(完了✅️)
- touchingの判定をこっちでやっている？base.pyのほうをこの定義にして，base.pyのものを使うようにする．(base.pyが先)(完了✅️)
- 表情シグナルを作るかどうか → **作る**（着手順 5e）。唇は `lip` 行動で動くので検出器だけ先でも常時 0 にはならない
- 各シグナルがどう使われているのか？全部使われている？無駄があるのでは？ → infant_signal.md 各表の **使用** 列が正本（着手順 2 は完了）。死キーは 4 で消す
- body modalityいらないかも → **いらない。削除**（着手順 4）。拒否の `avoidance_intensity` は残す
- point_intensity と open_intensityという名前を変えるか，値を変えるか
- infant_signal.pyの中ですでにそのシグナルの持続率の判定になっているところがあり，プロトコルの中でも持続率を計測するようなところがある．これだと持続率の持続率を計算しているよくわからないことになっていないか．（完了✅️。正本は着手順 3。プロトコルの `persistence`（割り込まれにくさ）とは別）
- arm_extを使っていないが，使っても良いのでは？ → **残す**（出力のみ）。intent / 弱いジェスチャには今は入れない（静止時 1.0 で誤発火した）。将来の読み手用にキーと計算は残す
- fistはいらない → 削除。open を -1〜1 にして fist を兼ねる案は採らない（極性を混ぜると intent の `> 0` が壊れる）
- hand_movement_size / hand_movement_regularity / head_nod_periodicity はいらない → 削除（二項 salience から既に外してある）
- 物を持って掲げ続けているというシグナルがほしい → 着手順 5a。`holding` × 高さ。新しい高さ窓は作らない
- babblingの定義は今のままでいいのか（異なる母音の並びが繰り返されるけど単語には判定されない、のほうがいいのでは） → 着手順 5c。契約を書いてから
- 音声の感情の定義。声の大きさやピッチで判定したい → 着手順 5d。表情とは別チャネル。拒否にはまだ繋がない

## 着手順（infant_signal.pyをどれから直すか）

方針: **契約バグ（完了）→ 使用箇所は infant_signal.md の使用列（完了）→ 持続の責務（完了）→ 死キー削除（fist / 動き・うなずき / body。`arm_ext` は残す）→ 新シグナルは契約を書いてから（掲げ続け → 単語距離バグ → babbling → 声の感情 → 表情）→ 名前の整理**。新キーは持続契約の前には入らない。対象ベクトルと intensity の向きが嘘のままだと、その上の「弱い腕」「持続」の議論が全部ずれる。touching / holding は base 側が先で、完了済み。ここからは検出器本体。

正本は infant_signal.md（キー契約）と §9 の注意事項。intent.md は「どのキーを 0/1 に束ねるか」で、検出器の値の意味は変えない。

### 0. コードを触る前に仕分け

infant_signal.md §9 を「バグとして直す / 意図として残す / 後回し」に分ける。下の項目の多くはこれと一致する。ここを飛ばすと、履歴平均の安定性まで「瞬間値にしろ」と潰す。

直す寄り（§9が明示）:
- 8° リングの確度が常に 0（中心+3° の 5 本しか効いていない）（完了✅️）
- `point_intensity` / `open_intensity` が理想姿勢からの**距離**（小さいほど良い）なのに強度として出している。reset 直後の開き手は 0 になり、`intent.reach_is_complete` の `> 0` が成立しない（完了✅️）
- `fist_intensity` は姿勢名 `"fist"` を見ているが `_get_hand_shape` は `"closed"` を返すので常に 0（完了✅️：`"closed"` に合わせた）
- `voice_best_distance` は単語棄却時に 0.0（「距離が小さい」を素通り）。intent の単語ゲートと食い違う

残す寄り:
- 円錐レイの骨格（中心+リング）。8° を効かせるのは式の修正であって、レイ本数や円錐そのものを捨てない
- 安定性（`*_stability`）が履歴平均のエントロピーであること。瞬間の向きではない
- `touching` が指先接触ではなく `graspable_obj`（完了済み）
- `update()` が `env.step()` の直後なので、シグナルはその step の行動を反映する

判断済み（実装は 4・5。持続の契約は 3 で完了）:
- 表情を出す → 出す（5e）
- body モダリティ → 削除（4）。拒否の `avoidance_intensity` は残す
- `arm_ext` → **残す**（出力のみ・未使用のまま）。intent / 弱いジェスチャには今は入れない
- `fist_intensity` / `hand_movement_size` / `hand_movement_regularity` / `head_nod_periodicity` → 削除（4）
- 掲げ続けシグナルを足す（5a）
- babbling を「異なる音素の繰り返し、かつ単語ではない」方向で再定義する（5c。契約は書いてから）
- 音声の感情は volume / pitch で求める（5d）。表情とは分ける。拒否にはまだ繋がない

まだ決めてからコード（6）:
- intensity の名前を変えるか（値は直済み。改名は 6）

### 1. 最初に直す（契約バグ。intent / 対象ID の前提）

対象ベクトルと手形の数値が嘘だと、`hand_target_is_clear` / `gaze_targets` / 弱い腕の議論が全部空振りする。ここは小さいが学習・プロトコルの前提。

1. **8° リングが確度 0 で捨てられるのを直す。**（完了✅️）`cone_ray_accuracy` + `CONE_MIN_ACCURACY=0.25`。中心 1.0 / 3° 0.625 / 8° 0.25。`HAND_TARGET_MASS_MIN` 0.5→0.2。定数は `infant_signal.py`（`cone_*`）と `intent.py` で差し替え可能。

2. **`point_intensity` / `open_intensity` の向きを強度にする。**（完了✅️）検出器は `_get_hand_shape` の第 3 戻りをそのまま `pose_power` にしている。base の把持は第 3 戻りを捨てている（`hand_shape, grip_power, _`）ので、**base 側で距離ではなく intensity（近いほど大きい、[0,1]）を返す**のが筋。todo の「指先の接触は使わない」は touching 移行済みで追加作業なし。
   - これで reset 直後の開き手でも `open_intensity > 0` になり、`reach_is_complete` が素通りしなくなる。
   - **名前はまだ変えない。** 値が強度になってから、まだ名前が嘘なら 6 で改名する。値と名前を同時にいじると intent の `> 0` とテストの期待値が切り分けできない。
   - `fist_intensity` は `"closed"` に合わせた（完了✅️）。キー自体は 4 で消す。新規の握り検出は足さない。

### 2. 使用箇所（完了✅️。正本は infant_signal.md）

infant_signal.md 各表の **使用** 列がキーごとの読み手（intent / プロトコル / PM）。テスト・デモ・`ProtocolManager.signal_keys` は含めない、と §1 に書いてある。todo 側に表を書き直さない。4 の削除は、この列が **未使用** のキーだけ。

4 で消す（使用列が未使用）: `fist_intensity`, `hand_movement_size` / `hand_movement_regularity` / `head_nod_periodicity`, body 一式。
残す（未使用でも）: `arm_ext`（将来用）。`voice_volume_sustained`（5d が窓を使うかも）。表情メソッド（5e で作り直す）。`negative_emotion_surge`（拒否にはまだ繋がない）。拒否の `avoidance_intensity`（body ではない）。

### 3. 設計判断が先、実装は後（持続の責務）（完了✅️）

対象は **シグナルが有効とみなされるまでの時間の畳み込み**。プロトコルの `persistence`（割り込まれにくさ、2.7 / 5.0）は別物で、この項目の外。

**契約（infant_signal.md §1「時間の契約」が正本。コードは後）:**

- 十分な長さの履歴は検出器が持つ。式はすべて検出器に置く。窓長と「その畳み込みを使うか」は、読み手によって変わるものだけ読み手が決める（ヘルパに `n` を渡す）。
- シグナル自身の性質（`*_stability`、周期性、喃語比）は公開キーにしてよい。ヒット率・連続カウントなど「有効とみなす長さ」は、読み手のスケールが揃えられるなら揃えて公開キーにする。揃えられないものだけヘルパ。
- 窓付き公開キーに対し、読み手は**閾値かそのまま**だけ。再平均・step 間差分・そのキーの連続カウントは禁止。変化が欲しいときは瞬間側を見る。
- step 内のセンサ融合（円錐レイ、`infant_voice_window` の平均 → その step の volume）は「瞬間」であって、step 間の窓ではない。

**キーごとの読み分け（実装時）:**

- `gaze_targets`: 30 step 平均のまま（選択、`infant_is_engaged`、PM 見込み対象）。瞬間は `gaze_targets_now`（実装済み）。JA の `get_infant_target` / `AttentionHoldTracker` は瞬間。平均の argmax を 30 連続しない。
- `looking_at_parent_eye`: 30 step ヒット率のまま（intent `parent_dyadic_cue`）。瞬間キーは出さない。選好の `I_look` は内部のその step ヒットのまま。JA 報酬の割合差分はやめる。
- `gesture_targets`: 平均のみ。瞬間キーは足さない（連続カウントする読み手がいない）。
- `hand_height_margin`: 10 step ヒット率のまま（二項 `arm_raise`）。
- 5a（持って掲げ続け）: 検出器が「holding かつ目線帯」のヒット率を **同じ 10 step の一つの窓**で出す。高さ窓は増やさない。このキーの上に連続も積まない。`arm_raise` は高さだけの `hand_height_margin`。
- 5c の babbling: 述語を変えても窓は一つ。比の上に持続を重ねない。5d の感情も同じ（観測用の一つの畳み込み。拒否の微分にはまだ繋がない）。

**5a / 5c はこの契約に従う。** コードを動かす前に infant_signal.md のキー表へ落とす。

### 4. 死んでいるキーの削除（infant_signal.md 使用列が未使用のもの。新シグナルより先）

誰も読まないキーを先に落とす。削除と追加を同じ diff にしない。検出器の出力・計算・履歴、`_is_modality_active`、PM `signal_keys`、infant_signal.md、テストをセットで。

消す:
- `fist_intensity`（未使用。open の極性で兼ねない）
- `hand_movement_size` / `hand_movement_regularity`
- `head_nod_periodicity`（二項 MOTOR から既に外れている）
- body モダリティ一式: `distance_to_parent`, `proximity_intensity`, `orientation_to_parent`, `approach_intensity`, `body_targets`。`get_body_signals` 未定義のフォールバックも消える

残す:
- `arm_ext` と `get_arm_extension` / `arm_extensions`。使用列は「出力のみ・未使用」のまま。intent には入れない（静止時 1.0）。キーは残す
- `avoidance_intensity` と、それを支える `parent_positions` 履歴（拒否側。body ではない）
- `head_turn_away_intensity`（行動履歴。拒否側）
- `voice_volume_sustained`（5d が窓を使うかも）
- 表情のダミーメソッド（5e で置き換える）

### 5. 新シグナルは契約を infant_signal.md に書いてから（3 の持続契約のあと）

この中の順: **小さい・既存キーの積 → 単語のバグ → babbling（単語との境界）→ 声の感情（新キー）→ 表情（別モダリティ）**。読み手（intent / どのプロトコル / ログのみ）を契約に書いてからキーを足す。

#### 5a. 物を持って掲げ続けている

検出器が「その step で holding かつ目線帯」だったかのヒット率を、`hand_height_margin` と同じ 10 step で出す。瞬間×割合の積ではない。新しい高さ窓は作らない。

二項の `arm_raise` は**持っていない掲げ**。これは**持っている掲げ**（見せる／示す）。giving の `holding` とは高さが違う。

決まったこと（3 の契約）:
- 検出器が「holding かつ目線帯」のヒット率を、`hand_height_margin` と同じ 10 step で出す（積のキー。高さ窓は増やさない）
- 「続けている」をそのヒット率の上に重ねない。二項 `arm_raise` は高さだけの `hand_height_margin` のまま

#### 5b. `voice_best_distance` の棄却時 0.0

単語意図のバグ。検出器が `None` のとき距離 0 を返さない（大きな値か、intent 側で `best_word is None` を見る）。intent とセット。

5c の前提。babbling を「単語ではない」で切るなら、棄却が嘘のままだと境界が壊れる。持続の議論より小さく、1 のあとでもよかったが、babbling 再定義に巻き込む。

#### 5c. babbling の再定義

現行: 有声中の音素履歴の**最頻出 1 音素**とその割合。これは「同じ音を伸ばしている」寄りで、canonical babbling（異なる音の繰り返し）の**逆**に近い。

案: **異なる音素の並びが繰り返されるが、単語にはならない。**

契約で決める（コードの前に）:
- 音素は母音ではなく `p0`..`p7`。「異なる母音」は「単位に 2 種以上の音素」と読み替える
- 繰り返しの定義（周期、n-gram の再出）
- 単語との排他: `words` が採択されている区間は babbling にしないか
- `babbling_target` が 1 音素のままだと二項の模倣音素（`target_phoneme`）が壊れる。列にするか、繰り返し単位の代表にするか
- dyadic の voice salience（`babbling_ratio × volume × (1 - stability)`）の意味が変わる。テストとセット

5b のあと。繰り返し窓を足すなら 3 の契約に従う。

#### 5d. 音声の感情（volume / pitch）

履歴はすでにある。`emotion_prototypes`（Calm / Excitement / Distress / Whine / Refusal / Sleepy）も volume・pitch 特徴で書いてあるが未接続。language.md どおり、音素認識には pitch を使わない。感情だけが pitch を読む。

表情（唇）とは**別チャネル**。`get_face_expression` の 6 分類ダミーと声のプロトタイプが同じラベルなのを、実装時にほどく（6 分類は声側へ）。

決めること:
- 6 分類を残すか、正負＋強度にするか
- 瞬間か窓か（3 に従う）
- `negative_emotion_surge` に微分で繋ぐか。拒否判定は今オフで、入れると過剰発火するので、**この段階では拒否に繋がない**。キーを出して観測するだけ

#### 5e. 表情シグナル

作る。唇は `lip` 行動で `left_lip_joint` / `right_lip_joint` が動く。

ただし既存の `get_smile_intensity` / `get_distress_intensity` / `get_surprise_intensity` は **全部 `left_lip_joint` の符号が違うだけ**。1 DoF なのに 3 表情のフリをしている。そのままキー化するな。

契約で:
- 唇 1 軸で何が区別できるか（例: smile vs distress は符号。surprise は別関節が無いなら出さない）
- キー名と値域
- 誰が読むか（intent / 報酬 / ログのみ）。読み手が拒否だけなら 5d と同じくまだ繋がない

`get_expression_signals` を定義して、ダミーの `hasattr` 分岐を消す。履歴の 6 分類は 5d の声側へ移す。

### 6. 名前・掃除（挙動が安定してから。バグ修正と混ぜない）

- `point_intensity` / `open_intensity` の改名。値が強度になったあとでも「intensity」でよいか。intent は `> 0` しか見ないので、改名は検索置換とテストのキー名だけ。プロトコルの saliency には載っていない。
- PM の `signal_keys` の死キー落としは 4 でやる。5 で足したキーをログに載せるかもここ。
- `voice_volume_sustained` を消すなら 5d のあと（声の感情が窓を使わなかったとき）。
- `hasattr(self, "get_expression_signals")` は 5e で消える。`get_body_signals` フォールバックは 4 で消える。

### この順にしない理由

- 表情や声の感情から始める → 持続契約が無く、拒否に繋ぐと過剰発火。顔と声の 6 分類が混線したままキーが出る。
- babbling 再定義から始める → `voice_best_distance` が棄却を素通りするので「単語ではない」が嘘。dyadic の模倣音素も同時に壊れる。
- 掲げ続けから始める → 3 の契約どおり、同じ 10 step で連言ヒット率を出す。高さ窓を重ねない。
- 死キー削除で `avoidance_intensity` まで消す → infant_signal.md の使用列を見る。拒否は残す。
- `arm_ext` を消す／intent に足すところから始める → **どちらもしない**。キーは残す。intent に入れると静止時 1.0 で誤発火する。
- fist を open の負で兼ねる → 極性を混ぜると intent の `> 0` が壊れる。
- intensity の改名と値の反転を同時にやる → 値は済み。改名は 6。
- 持続の責務を先にコードで分ける → `looking_at_parent_eye >= 0.6` の意味が変わり、親始動の二項が静かに死ぬ／常時発火する。契約は 3 で書いた。実装はキー表へ落としてから。
- touching を検出器で再実装する → base の `graspable_obj` が正本（完了済み）。戻さない。

# protocol_manager.py
- プロトコルのインスタンスが使い回されるけど，この理由は？プロトコルは選ばれてからインスタンス化ではだめ？プロトコルを選ぶ前にプロトコルクラスの何かを使うとかだと確かにそう．
- initiativeのチェックでは，視線が主に使われる．一方，responseでは指さしなどが使われる．ここで，一つ疑問がある．視線のほうが先にでやすいので，先にinitiativeばかり起きてresponseは起きないのでは？要チェック．




