# 言語 (`language/`)

`cueworld/language/` は、親子が交わす「発話」を音素の並びとして定義し、生成側（親のテンプレート発話）と認識側（幼児発話の単語推定）の両方を担う。
本ドキュメントは**音素・語彙・テンプレート・認識の規則の正本**である。§§1–9 は実装済みの規則である。音素距離で DP コストを決める案は [§10](#10-将来的な変更案) に残す。

音声が行動・観測としてどう流れるかは [env.md §5](env.md#5-行動空間) と [§6](env.md#6-観測空間)。
認識結果がどのシグナルになるかは [infant_signal.md](infant_signal.md)。

| ファイル | 役割 |
|----------|------|
| `phonemes.py` | 音素 → 発声ベクトル座標 |
| `vocabulary.py` | 概念 ID → 音素列 / 日本語表記 |
| `templates.py` | プロトコル別の発話テンプレート |
| `manager.py` | `LanguageManager`。上記 3 つを引くファサード |
| `voice_recognizer.py` | `VoiceRecognizer`。音素推定と単語推定 |

---

## 1. 音素

音素は **`p0` 〜 `p7` の 8 種**。発声ベクトルの識別軸 `(v_1, v_2, v_3)` 上の点として定義される（`build_phoneme_mapping`）。

座標は添字 `i` の 3 ビット表記から自己記述的に決まる。ビット `b0, b1, b2` がそれぞれ `v_1, v_2, v_3` に対応し、`1` なら `+1`、`0` なら `-1`。

| 音素 | `(v_1, v_2, v_3)` | 音素 | `(v_1, v_2, v_3)` |
|------|-------------------|------|-------------------|
| `p0` | `(-1, -1, -1)` | `p4` | `(-1, -1, +1)` |
| `p1` | `(+1, -1, -1)` | `p5` | `(+1, -1, +1)` |
| `p2` | `(-1, +1, -1)` | `p6` | `(-1, +1, +1)` |
| `p3` | `(+1, +1, -1)` | `p7` | `(+1, +1, +1)` |

- 立方体の 8 頂点。**任意の 2 音素の距離は最小 2.0**（1 ビット違い）で、等方的に散らばる。特定のペアだけ混同しやすい、ということが起きない。
- `PHONEME_RADIUS = 0.8`。プロトタイプの許容半径で、認識側で距離を割り引くのに使う（[§5](#5-音素の推定)）。
- `v_1..v_3` に調音的な意味はない。**単に離散的な識別軸**として設計してある。
- `volume`（index 0）と `pitch`（index 1）は音素座標に含まれない。`volume` は有声／無声の判定にだけ使い、`pitch` は韻律用で認識には使われない。

`build_phoneme_mapping()` は `{"p0": np.array([...]), ...}` を返し、`BaseEnv.__init__` が `self.phoneme_mapping` として保持する。
`build_phoneme_prototypes()` は `{"p0": {"center": ..., "radius": PHONEME_RADIUS}, ...}` を返し、認識側が使う。

---

## 2. 語彙

`CONCEPT_TO_PHONEMES`（`vocabulary.py`）が `概念 ID → 音素列` の唯一の定義。**単語の中身を知りたいときはこの表を直接読むこと**（本書では再掲しない）。

規則:

- **全単語が固定長 5 音素**。長さの違いを手がかりにできないようにしてある。
- **同じ音素は隣り合わない。**
- **先頭の音素集合と末尾の音素集合は交わらない。** 先頭は `p0`–`p3`、末尾は `p4`–`p7`。
- **ハミング距離は 4 を優先し、入りきらない組だけ 3 を許す。** 距離 4 の組は、置換だけで 2 音が相手へ寄ると同点になる。距離 3 の組は、2 音がちょうどその相手へ寄ると別の単語として出ることがある。
- 残す概念はオブジェクト 20 と praise、recast、correct、douzo、declare、prompt。mother、kiss、oishiine、onomatopoeia_shake、onomatopoeia_contain、soliloquy は辞書に無い。
- `RESERVED_PHONEME_CODES` に予備の符号が 10 個ある。概念 ID は付けず、認識の候補にも入れない。あとから概念を足すときは、その符号へ名前を付ける。
- 概念 ID は物体側の `concept_id`（[object.md](object.md)）と同じ名前空間。`red_cube`, `blue_board` などがそのままキーになる。物体以外の機能語（`praise`, `declare`, `prompt` など）も同じ表に並ぶ。

`CONCEPT_TO_HUMAN_WORD` はログとデバッグ表示のための日本語表記で、認識にも生成にも使われない。

`LanguageManager` の入口:

| メソッド | 返り値 |
|----------|--------|
| `get_phonemes(concept_id)` | 音素リスト。未登録なら `[]` |
| `get_human_word(concept_id)` | 日本語表記。未登録なら concept_id をそのまま |
| `get_concept_for_object(obj)` | 物体インスタンスの `concept_id` |
| `get_word_to_phonemes_dict()` | `CONCEPT_TO_PHONEMES` のコピー。認識器の辞書になる |

**認識器が返す「単語」は概念 ID である**（日本語表記ではない）。`get_word_to_phonemes_dict` の docstring は「日本語文字列 → 音素」と書いてあるが、実際のキーは概念 ID。

---

## 3. テンプレート

`UTTERANCE_TEMPLATES`（`templates.py`）は `プロトコル名 → テンプレートキー → 概念 ID のリスト`。

```python
"joint_attention": {
    "declare_object": ["declare", "TARGET"],
    ...
}
```

規則:

- 要素は概念 ID。**`"TARGET"` だけが特別**で、呼び出し時に渡された `concept_id` に差し替えられる。
- **1 テンプレートは最大 2 語**。機能語が先、対象語が後。文法構造は扱わない。
- プロトコル名で引いて無ければ `"common"` にフォールバックする。それでも無ければ警告を出して空リストを返す（発話しない）。
- プロトコル名はクラス名から自動導出される。`XxxYyyProtocol` → `xxx_yyy`（`JointAttentionProtocol` → `joint_attention`）。テンプレートのキーはこの名前に合わせる必要がある。

### 音素列への展開（`get_template_phonemes`）

各概念 ID を語彙の 5 音素へ展開し、`(音素, 継続フレーム数)` のリストにする。

| 定数 | 値 | 意味 |
|------|-----|------|
| `PHONEME_DURATION_FRAMES` | 6 | 1 音素を保持する step 数（生成側のフレーム。[§4](#時間の単位)） |
| `INTER_WORD_GAP_FRAMES` | 1 | 単語間に挿入する無声（`"none"`）の step 数 |

- 語頭には gap を入れない。**単語と単語の間だけ**。
- 2 語のテンプレートは `5×6 + 1 + 5×6 = 61` step ≒ 6.1 秒。1 語なら 30 step ≒ 3 秒。ここでのフレームは step である。
- `TARGET` があるのに `concept_id` が `None`、または未登録の概念 ID があった場合は、警告を出してそのスロットを飛ばす（発話全体は中断しない）。

### 発話の送出

`BaseProtocol` 側の 2 メソッドが担う（[protocol/README.md](protocol/README.md)）。

- `queue_utterance(template_key, concept_id)`: 展開結果を **1 フレーム 1 音素**に潰して `utterance_queue`（`deque`）に積む。`parent_utterance=False` なら何もしない。
- `_emit_utterance(action, volume)`: キューの先頭を 1 つ取り出し、`apply_phoneme_to_action` で行動ベクトルの `v_1..v_3` と `volume` に書く。`"none"` または未知ラベルなら `volume = 0`（無声）。キューが空なら行動を変えない。

したがって**発話は 1 step に 1 音素フレームで進み、プロトコルの状態遷移とは独立に消化される**。プロトコルが状態を変えてもキューは残る。生成側のフレームは step であり、認識のティックではない（[§4](#時間の単位)）。

---

## 4. 親が幼児の声を認識する流れ

親専用の耳はない。`InfantSignalDetector.update` が幼児の `infant_voice_window` を読み、親はそこから出るシグナルだけを見る（[infant_signal.md §6](infant_signal.md#6-音声シグナル)）。生成側のテンプレート（[§3](#3-テンプレート)）とは別経路である。

### 時間の単位

| 単位 | 何か | 長さ |
|------|------|------|
| step | `BaseEnv.step` の 1 回。`step_count` はこの回数 | 約 0.1 秒 |
| フレーム | その step の発声ベクトル 1 個。毎 step、`infant_voice_window` に追加される | 1 step |
| ティック | 音素を 1 ラベルにする認識の 1 回。`step_count % voice_interval == 0` の step だけ起きる | 既定 3 step（`voice_interval = 3`） |

窓 `infant_voice_window` の長さは `voice_window_size = 3`。直近 3 フレーム（直近 3 step）が入っている。

生成側の `PHONEME_DURATION_FRAMES = 6` は、親の発話で 1 音素を 6 フレーム、つまり 6 step 保持する、という意味である。

認識側の `offset_silence_frames = 4` は、名前に frame とあるが数えているのはティックである。無音 4 ティックは 12 step ≒ 1.2 秒。単語の判定はティック数ではなく、畳んだ音素キューの長さで行う。

### 流れ

```
infant_voice_window（直近 3 フレーム）
  ↓ 1 ティックにつき、各フレームを get_phoneme
  ├─ コスト最小の 1 ラベル → phoneme_history（喃語。単語判定には使わない）
  └─ 残す閾値を通ったラベルを時間順に畳む → 長さ 6 のキュー
       ↓ 話しているあいだは 6 個のときだけ判定
       ↓ 無声 4 ティックで 3〜6 個なら判定。1〜2 個は何も出さない
       └─ 概念 ID（words）と、対象があるときの返事（voice_reply）
```

対象は、音素を頂点に置いた発話。6 step でも 9 step でも、同じ音は 1 個に畳まれる。移行中の座標はここでは扱わない。

有声かどうかは窓の平均 volume で決める。閾値は 0.25。喃語用の 1 ラベルは、窓の 3 フレームのうち音素コストが一番小さいフレームである。同点は窓の末尾に近いほう。

単語キューへ入れるのは、そのティックの窓にある、残す閾値（`phoneme_keep_threshold = 0.39`）を通ったフレームだけである。隣り合う同じ音素は 1 個に畳む。`unknown` と無声は入れない。

- 話しているあいだは、キューが 6 個のときだけ判定する。採用できる単語があれば、その単語に使った音と、その前の音を捨て、うしろだけを残す。単語がなければ切らない。次の音で一番古い音が流れ出て、また 6 個のときだけ判定する。
- 無声が 4 ティック続いたら区間を閉じる。3 個以上なら判定する。1 個か 2 個なら単語も不採用も出さない。
- 採用したときは、下駄後の生コストが `words` に残る。小さいほど近い。

`voice_segment_seq` は、単語を公開したときと、黙って 3 音以上を判定したときに増える。

### 例: 1 語を頂点に置いて言う

音素を 3 step、6 step、9 step のどれで置いても、同じ音は 1 個に畳まれる。5 音そろっただけでは判定しない。黙って 4 ティック置くと、生コスト 0 でその語が出る。1 音か 2 音で黙ると、単語も不採用も出さない。

無音なしに次の語へ続けると、次の語の先頭が 6 音目になる。先頭と末尾の音素集合は交わらないので、6 音目は先の語の続きではない。そこで先の語を切り、6 音目を次の語の列に残す。

数値は [voice_recognition_report.md](voice_recognition_report.md) の、音素頂点へ直接置いた発話の結果である。

---

## 5. 音素の推定

`InfantSignalDetector` は `voice_interval` ごとに `infant_voice_window` の各フレームを `get_phoneme` する。スコアが無い無声フレームは候補にしない。残ったフレームのうち音素コスト（スコア辞書の最小）が一番小さいものをそのティックの 1 ラベルにする。同点は窓の末尾に近いほう。有声判定の volume は窓の平均のまま。

`get_phoneme(avg_voice, volume_index=0, phoneme_index=2)`。1 フレームを判定する。

1. `volume < voice_volume_threshold`（既定 0.25、`BaseEnv.voice_volume_threshold`）なら `None` を返す。無声。有声は `volume >= threshold`。
2. `avg_voice[2:]` を特徴ベクトルとし、各プロトタイプ中心とのユークリッド距離を取る。
3. `use_radius=True`（既定）なら `dist = max(0, dist - 0.25 × radius)`。半径内をゼロに落とさず滑らかに割り引く。
4. コストに正規化: `score = min(dist / phoneme_cost_scale, 1.0)`（`phoneme_cost_scale = 2.0` は隣接音素間の距離）。全音素分を辞書で返す。
5. 最良と 2 番目を比べ、**最良コスト ≤ `phoneme_match_threshold`（0.4）かつ差 ≥ `phoneme_ambiguity_margin`（0.06）**ならそのラベル。満たさなければ `"unknown"`。
6. 単語キューへ入れるのは、ラベルがあり、最良コスト ≤ `phoneme_keep_threshold`（0.39）のフレームだけ。頂点に置いた音（スコア 0）は通り、隣り合う音素の中点（スコア 0.4）は落ちる。

`None`（無声）と `"unknown"`（有声だが同定できない）は区別される。単語キューにはどちらも入れない。

---

## 6. 発話区間の切り出し

`update_segment(volume, kept_labels=...)`。

- `volume >= voice_volume_threshold` を有声とする。残す閾値を通ったラベルを、隣り合う同じ音は 1 個に畳んでキューへ入れる。キューの最大は 6。
- 話しているあいだは、キューが 6 個になったときだけ判定する。採用できれば、単語に使った最後の音までを捨て、うしろを残す。採用できなければ切らず、次の音で最古が流れ出る。
- 無声のティックが `offset_silence_frames`（4 ティック）続いたら区間終了。3 個以上なら判定する。1 個か 2 個なら列を捨てて何も出さない。
- `reset()` で区間状態と返事の記憶をクリアする。

`onset_hangover`（2）は属性として定義されているが、どこからも参照されていない。

---

## 7. 単語の推定

畳んだ観測列 `X` と辞書の各単語 `Y`（長さ 5）を、ラベルの一致だけで DP する。フレームの音素スコアは置換に使わない。予備の 10 符号は候補に入れない。

### コスト

| 操作 | コスト |
|------|--------|
| 単語の前にはみ出した音を 1 つ捨てる | 0.01 |
| 列が 6 音のときの 6 音目だけを捨てる | 0 |
| 6 音目より前の、単語の途中にある余計な音を捨てる | 1.0 |
| 足りない音を足す | 1.0 |
| 違う音素への置換 | 1.0（一致なら 0） |

黙って 3 音、4 音、5 音で判定するときは、コスト 0 で捨てる末尾はない。`cost_elongate`、`cost_del_blank`、`cost_free_edge`、長さでの正規化は使わない。

### 下駄と採用

プロトコルが対象の概念 ID を認識器へ渡しているとき、対象の 3 音が間を空けずに列へ入っていれば、その対象の生コストを 1.0 引く。間に別の音が入る 3 音には付けない。引いた結果が音だけの第 1 位を越えてもよい。

単語の採用は、下駄を引いたあとの生コストが 3.0 未満のときだけ行う。いちばん安い語を採用する。同点は概念 ID の辞書順で一つに決める。2 位との差 0.08 と系列品質 0.18 は使わない。音素を残す閾値で、曖昧なフレームは列に入らない。

### 返事

返事の種類は判定のときに決める。対象が無いときは返事を出さず、音だけの採用だけ行う。

| 返事 | 条件 | 言うとき | テンプレート |
|------|------|----------|--------------|
| すごい | 下駄のあとで対象が 1 位、かつ下駄の前の生コストが 0.01 以下 | その場 | `praise_with_target` |
| おしい | 下駄のあとで安い順の 3 語以内、その生コストが 3.0 未満、かつ下駄の前の生コストが 0.01 より大きい | 黙ったとき | `recast_with_target` |
| ちがうよ | 上のどちらでもない | 黙ったとき | `correct_with_target` |

3.0 未満が 3 語に満たないときは、そこで止める。同点で 3 語を超えるときは、安い順に 3 語で切り、その中の 3.0 未満だけを残す。まだ言っていない「ちがうよ」が「おしい」になったときは「おしい」を一度言う。「おしい」のあと「ちがうよ」は言わない。どちらのあとでも「すごい」は言う。同じまとまりで「すごい」を出したら、閉じたときの「おしい」「ちがうよ」は出さない。区間が閉じたら「もう言った」を戻し、次のまとまりでも同じ種類を言える。

`last_word_recognition_metrics` に生コスト、下駄後、長さ、返事の種類が残る。不採用のスコア表は `words` に出さない。

---

## 8. 現行カタログが使うテンプレート

| プロトコル | テンプレートキー | 中身 |
|-----------|------------------|------|
| `joint_attention` | `declare_object` | `["declare", "TARGET"]` |
| `joint_attention` | `praise_with_target` | `["praise", "TARGET"]` |
| `joint_attention` | `recast_with_target` | `["recast", "TARGET"]` |
| `joint_attention` | `correct_with_target` | `["correct", "TARGET"]` |
| `object_giving` | `present` | `["douzo", "TARGET"]` |
| `dyadic` | `prompt` | `["prompt"]`（common） |
| 共通 | `praise` | `["praise"]`（JA の sharing 意図と dyadic が使う） |

`HouseholdSoloProtocol` と `ObservanceProtocol` は発話しない。

---

## 9. 注意事項

**辞書から外した語彙と、まだ呼ばないテンプレートがある。**
mother、kiss、oishiine、onomatopoeia_shake、onomatopoeia_contain、soliloquy は `CONCEPT_TO_PHONEMES` に無い。`recast` は「おしい」の `recast_with_target` で使う。裸のテンプレート `recast` / `correct` / `target_only` と、`object_play` / `physical_contact` / `dyadic_play` 用のエントリは、現行カタログからは呼ばれない。

**`pitch` は認識に使われない。**
発声ベクトルの index 1 は生成側で書き込めるが、認識側は `avg_voice[2:]` しか見ない。

**`PhysicalContactProtocol` は `queue_utterance` を独自に上書きしている。**
`BaseProtocol` のテンプレート展開経路を通らない。現行カタログには含まれない。

**音素の継続 step 数は、畳み込みが吸収する。**
生成側は `PHONEME_DURATION_FRAMES = 6` step。認識側は隣り合う同じ音を 1 個に畳むので、3 step でも 9 step でも同じ 1 音素になる。

---

## 10. 将来的な変更案

すぐには実装しない。作業メモは [voice_recognition_plan.md](voice_recognition_plan.md)。

「おしい」「ちがうよ」を黙る前に言うまでの時間は、まだ決めていない。今回は無声 4 ティックで言う。

いずれ、幼児が出した音が各音素へどれだけ近いかで、DP の操作コストを決める。削除のコストは、いちばん近い音素への距離で決める。近いほど削除は高い。置換のコストは、置き換え先の音素への距離で決める。いちばん近い音素でも距離が残るので、完全一致にはほぼならない。挿入のコストは一定のままにする。

このコストに合わせて、採用の 3.0 未満、「すごい」の 0.01、「おしい」の 3.0 未満を作り直す。この場合、Unknown のような音素を作らなくてもよいかもしれない。
