# いくつかの単語を幼児エージェントに話させて，親エージェントが認識するテスト

親が読んでいるのは `InfantSignalDetector`。発声は音素頂点へ直接置き、`infant_voice_window` と `update()` に流した。
生コストと下駄後は、その発話で最後に公開された単語の値。出なかったときは最後の公開が無いので空欄。

## 単語を素早く言う

音素あたり 3 step（採取 1 回）。畳んで 5 音になり、黙ったときに判定する。

| 話した語 | 位相 | 出た語 | 正誤 | 生コスト | 下駄後 | 返事 | 理由 |
|---|---:|---|---|---:|---:|---|---|
| red_cube | 0 | red_cube | 正 | 0.000 | 0.000 | — |  |
| red_sphere | 0 | red_sphere | 正 | 0.000 | 0.000 | — |  |
| red_cylinder | 0 | red_cylinder | 正 | 0.000 | 0.000 | — |  |
| red_cone | 0 | red_cone | 正 | 0.000 | 0.000 | — |  |
| blue_cube | 0 | blue_cube | 正 | 0.000 | 0.000 | — |  |
| blue_sphere | 0 | blue_sphere | 正 | 0.000 | 0.000 | — |  |
| blue_cylinder | 0 | blue_cylinder | 正 | 0.000 | 0.000 | — |  |
| blue_cone | 0 | blue_cone | 正 | 0.000 | 0.000 | — |  |
| green_cube | 0 | green_cube | 正 | 0.000 | 0.000 | — |  |
| green_sphere | 0 | green_sphere | 正 | 0.000 | 0.000 | — |  |
| green_cylinder | 0 | green_cylinder | 正 | 0.000 | 0.000 | — |  |
| green_cone | 0 | green_cone | 正 | 0.000 | 0.000 | — |  |
| yellow_cube | 0 | yellow_cube | 正 | 0.000 | 0.000 | — |  |
| yellow_sphere | 0 | yellow_sphere | 正 | 0.000 | 0.000 | — |  |
| yellow_cylinder | 0 | yellow_cylinder | 正 | 0.000 | 0.000 | — |  |
| yellow_cone | 0 | yellow_cone | 正 | 0.000 | 0.000 | — |  |
| red_board | 0 | red_board | 正 | 0.000 | 0.000 | — |  |
| blue_board | 0 | blue_board | 正 | 0.000 | 0.000 | — |  |
| green_board | 0 | green_board | 正 | 0.000 | 0.000 | — |  |
| yellow_board | 0 | yellow_board | 正 | 0.000 | 0.000 | — |  |
| praise | 0 | praise | 正 | 0.000 | 0.000 | — |  |
| recast | 0 | recast | 正 | 0.000 | 0.000 | — |  |
| correct | 0 | correct | 正 | 0.000 | 0.000 | — |  |
| douzo | 0 | douzo | 正 | 0.000 | 0.000 | — |  |
| declare | 0 | declare | 正 | 0.000 | 0.000 | — |  |
| prompt | 0 | prompt | 正 | 0.000 | 0.000 | — |  |

### 認識できなかったパターン

なし。

## 単語を少し伸ばして言う

音素あたり 6 step（採取 2 回）。開始 step を 0、1、2 の 3 位相。

| 話した語 | 位相 | 出た語 | 正誤 | 生コスト | 下駄後 | 返事 | 理由 |
|---|---:|---|---|---:|---:|---|---|
| red_cube | 0 | red_cube | 正 | 0.000 | 0.000 | — |  |
| red_sphere | 0 | red_sphere | 正 | 0.000 | 0.000 | — |  |
| red_cylinder | 0 | red_cylinder | 正 | 0.000 | 0.000 | — |  |
| red_cone | 0 | red_cone | 正 | 0.000 | 0.000 | — |  |
| blue_cube | 0 | blue_cube | 正 | 0.000 | 0.000 | — |  |
| blue_sphere | 0 | blue_sphere | 正 | 0.000 | 0.000 | — |  |
| blue_cylinder | 0 | blue_cylinder | 正 | 0.000 | 0.000 | — |  |
| blue_cone | 0 | blue_cone | 正 | 0.000 | 0.000 | — |  |
| green_cube | 0 | green_cube | 正 | 0.000 | 0.000 | — |  |
| green_sphere | 0 | green_sphere | 正 | 0.000 | 0.000 | — |  |
| green_cylinder | 0 | green_cylinder | 正 | 0.000 | 0.000 | — |  |
| green_cone | 0 | green_cone | 正 | 0.000 | 0.000 | — |  |
| yellow_cube | 0 | yellow_cube | 正 | 0.000 | 0.000 | — |  |
| yellow_sphere | 0 | yellow_sphere | 正 | 0.000 | 0.000 | — |  |
| yellow_cylinder | 0 | yellow_cylinder | 正 | 0.000 | 0.000 | — |  |
| yellow_cone | 0 | yellow_cone | 正 | 0.000 | 0.000 | — |  |
| red_board | 0 | red_board | 正 | 0.000 | 0.000 | — |  |
| blue_board | 0 | blue_board | 正 | 0.000 | 0.000 | — |  |
| green_board | 0 | green_board | 正 | 0.000 | 0.000 | — |  |
| yellow_board | 0 | yellow_board | 正 | 0.000 | 0.000 | — |  |
| praise | 0 | praise | 正 | 0.000 | 0.000 | — |  |
| recast | 0 | recast | 正 | 0.000 | 0.000 | — |  |
| correct | 0 | correct | 正 | 0.000 | 0.000 | — |  |
| douzo | 0 | douzo | 正 | 0.000 | 0.000 | — |  |
| declare | 0 | declare | 正 | 0.000 | 0.000 | — |  |
| prompt | 0 | prompt | 正 | 0.000 | 0.000 | — |  |
| red_cube | 1 | red_cube | 正 | 0.000 | 0.000 | — |  |
| red_sphere | 1 | red_sphere | 正 | 0.000 | 0.000 | — |  |
| red_cylinder | 1 | red_cylinder | 正 | 0.000 | 0.000 | — |  |
| red_cone | 1 | red_cone | 正 | 0.000 | 0.000 | — |  |
| blue_cube | 1 | blue_cube | 正 | 0.000 | 0.000 | — |  |
| blue_sphere | 1 | blue_sphere | 正 | 0.000 | 0.000 | — |  |
| blue_cylinder | 1 | blue_cylinder | 正 | 0.000 | 0.000 | — |  |
| blue_cone | 1 | blue_cone | 正 | 0.000 | 0.000 | — |  |
| green_cube | 1 | green_cube | 正 | 0.000 | 0.000 | — |  |
| green_sphere | 1 | green_sphere | 正 | 0.000 | 0.000 | — |  |
| green_cylinder | 1 | green_cylinder | 正 | 0.000 | 0.000 | — |  |
| green_cone | 1 | green_cone | 正 | 0.000 | 0.000 | — |  |
| yellow_cube | 1 | yellow_cube | 正 | 0.000 | 0.000 | — |  |
| yellow_sphere | 1 | yellow_sphere | 正 | 0.000 | 0.000 | — |  |
| yellow_cylinder | 1 | yellow_cylinder | 正 | 0.000 | 0.000 | — |  |
| yellow_cone | 1 | yellow_cone | 正 | 0.000 | 0.000 | — |  |
| red_board | 1 | red_board | 正 | 0.000 | 0.000 | — |  |
| blue_board | 1 | blue_board | 正 | 0.000 | 0.000 | — |  |
| green_board | 1 | green_board | 正 | 0.000 | 0.000 | — |  |
| yellow_board | 1 | yellow_board | 正 | 0.000 | 0.000 | — |  |
| praise | 1 | praise | 正 | 0.000 | 0.000 | — |  |
| recast | 1 | recast | 正 | 0.000 | 0.000 | — |  |
| correct | 1 | correct | 正 | 0.000 | 0.000 | — |  |
| douzo | 1 | douzo | 正 | 0.000 | 0.000 | — |  |
| declare | 1 | declare | 正 | 0.000 | 0.000 | — |  |
| prompt | 1 | prompt | 正 | 0.000 | 0.000 | — |  |
| red_cube | 2 | red_cube | 正 | 0.000 | 0.000 | — |  |
| red_sphere | 2 | red_sphere | 正 | 0.000 | 0.000 | — |  |
| red_cylinder | 2 | red_cylinder | 正 | 0.000 | 0.000 | — |  |
| red_cone | 2 | red_cone | 正 | 0.000 | 0.000 | — |  |
| blue_cube | 2 | blue_cube | 正 | 0.000 | 0.000 | — |  |
| blue_sphere | 2 | blue_sphere | 正 | 0.000 | 0.000 | — |  |
| blue_cylinder | 2 | blue_cylinder | 正 | 0.000 | 0.000 | — |  |
| blue_cone | 2 | blue_cone | 正 | 0.000 | 0.000 | — |  |
| green_cube | 2 | green_cube | 正 | 0.000 | 0.000 | — |  |
| green_sphere | 2 | green_sphere | 正 | 0.000 | 0.000 | — |  |
| green_cylinder | 2 | green_cylinder | 正 | 0.000 | 0.000 | — |  |
| green_cone | 2 | green_cone | 正 | 0.000 | 0.000 | — |  |
| yellow_cube | 2 | yellow_cube | 正 | 0.000 | 0.000 | — |  |
| yellow_sphere | 2 | yellow_sphere | 正 | 0.000 | 0.000 | — |  |
| yellow_cylinder | 2 | yellow_cylinder | 正 | 0.000 | 0.000 | — |  |
| yellow_cone | 2 | yellow_cone | 正 | 0.000 | 0.000 | — |  |
| red_board | 2 | red_board | 正 | 0.000 | 0.000 | — |  |
| blue_board | 2 | blue_board | 正 | 0.000 | 0.000 | — |  |
| green_board | 2 | green_board | 正 | 0.000 | 0.000 | — |  |
| yellow_board | 2 | yellow_board | 正 | 0.000 | 0.000 | — |  |
| praise | 2 | praise | 正 | 0.000 | 0.000 | — |  |
| recast | 2 | recast | 正 | 0.000 | 0.000 | — |  |
| correct | 2 | correct | 正 | 0.000 | 0.000 | — |  |
| douzo | 2 | douzo | 正 | 0.000 | 0.000 | — |  |
| declare | 2 | declare | 正 | 0.000 | 0.000 | — |  |
| prompt | 2 | prompt | 正 | 0.000 | 0.000 | — |  |

### 認識できなかったパターン

なし。

## 単語を伸ばして言う

音素あたり 9 step（採取 3 回）。畳み込みで 5 音になり、黙ったときに判定する。

| 話した語 | 位相 | 出た語 | 正誤 | 生コスト | 下駄後 | 返事 | 理由 |
|---|---:|---|---|---:|---:|---|---|
| red_cube | 0 | red_cube | 正 | 0.000 | 0.000 | — |  |
| red_sphere | 0 | red_sphere | 正 | 0.000 | 0.000 | — |  |
| red_cylinder | 0 | red_cylinder | 正 | 0.000 | 0.000 | — |  |
| red_cone | 0 | red_cone | 正 | 0.000 | 0.000 | — |  |
| blue_cube | 0 | blue_cube | 正 | 0.000 | 0.000 | — |  |
| blue_sphere | 0 | blue_sphere | 正 | 0.000 | 0.000 | — |  |
| blue_cylinder | 0 | blue_cylinder | 正 | 0.000 | 0.000 | — |  |
| blue_cone | 0 | blue_cone | 正 | 0.000 | 0.000 | — |  |
| green_cube | 0 | green_cube | 正 | 0.000 | 0.000 | — |  |
| green_sphere | 0 | green_sphere | 正 | 0.000 | 0.000 | — |  |
| green_cylinder | 0 | green_cylinder | 正 | 0.000 | 0.000 | — |  |
| green_cone | 0 | green_cone | 正 | 0.000 | 0.000 | — |  |
| yellow_cube | 0 | yellow_cube | 正 | 0.000 | 0.000 | — |  |
| yellow_sphere | 0 | yellow_sphere | 正 | 0.000 | 0.000 | — |  |
| yellow_cylinder | 0 | yellow_cylinder | 正 | 0.000 | 0.000 | — |  |
| yellow_cone | 0 | yellow_cone | 正 | 0.000 | 0.000 | — |  |
| red_board | 0 | red_board | 正 | 0.000 | 0.000 | — |  |
| blue_board | 0 | blue_board | 正 | 0.000 | 0.000 | — |  |
| green_board | 0 | green_board | 正 | 0.000 | 0.000 | — |  |
| yellow_board | 0 | yellow_board | 正 | 0.000 | 0.000 | — |  |
| praise | 0 | praise | 正 | 0.000 | 0.000 | — |  |
| recast | 0 | recast | 正 | 0.000 | 0.000 | — |  |
| correct | 0 | correct | 正 | 0.000 | 0.000 | — |  |
| douzo | 0 | douzo | 正 | 0.000 | 0.000 | — |  |
| declare | 0 | declare | 正 | 0.000 | 0.000 | — |  |
| prompt | 0 | prompt | 正 | 0.000 | 0.000 | — |  |

### 認識できなかったパターン

なし。

## 単語の中に関係ない音素を一つだけ混ぜて言う

6 step の語の 2 音素目のあとに、その位置と違う最初の音素を 6 step 入れる。キューが 6 個になったときに判定する。

| 話した語 | 位相 | 出た語 | 正誤 | 生コスト | 下駄後 | 返事 | 理由 |
|---|---:|---|---|---:|---:|---|---|
| red_cube | 0 | red_cube | 正 | 1.000 | 1.000 | — |  |
| red_sphere | 0 | red_sphere | 正 | 1.000 | 1.000 | — |  |
| red_cylinder | 0 | red_cylinder | 正 | 1.000 | 1.000 | — |  |
| red_cone | 0 | red_cone | 正 | 1.000 | 1.000 | — |  |
| blue_cube | 0 | blue_cube | 正 | 1.000 | 1.000 | — |  |
| blue_sphere | 0 | blue_sphere | 正 | 1.000 | 1.000 | — |  |
| blue_cylinder | 0 | blue_cylinder | 正 | 0.000 | 0.000 | — |  |
| blue_cone | 0 | blue_cone | 正 | 1.000 | 1.000 | — |  |
| green_cube | 0 | green_cube | 正 | 0.000 | 0.000 | — |  |
| green_sphere | 0 | green_sphere | 正 | 1.000 | 1.000 | — |  |
| green_cylinder | 0 | green_cylinder | 正 | 1.000 | 1.000 | — |  |
| green_cone | 0 | green_cone | 正 | 1.000 | 1.000 | — |  |
| yellow_cube | 0 | yellow_cube | 正 | 1.000 | 1.000 | — |  |
| yellow_sphere | 0 | yellow_sphere | 正 | 1.000 | 1.000 | — |  |
| yellow_cylinder | 0 | yellow_cylinder | 正 | 1.000 | 1.000 | — |  |
| yellow_cone | 0 | yellow_cone | 正 | 1.000 | 1.000 | — |  |
| red_board | 0 | red_board | 正 | 0.000 | 0.000 | — |  |
| blue_board | 0 | blue_board | 正 | 1.000 | 1.000 | — |  |
| green_board | 0 | green_board | 正 | 1.000 | 1.000 | — |  |
| yellow_board | 0 | yellow_board | 正 | 1.000 | 1.000 | — |  |
| praise | 0 | praise | 正 | 1.000 | 1.000 | — |  |
| recast | 0 | recast | 正 | 0.000 | 0.000 | — |  |
| correct | 0 | correct | 正 | 0.000 | 0.000 | — |  |
| douzo | 0 | douzo | 正 | 1.000 | 1.000 | — |  |
| declare | 0 | declare | 正 | 1.000 | 1.000 | — |  |
| prompt | 0 | prompt | 正 | 1.000 | 1.000 | — |  |

### 認識できなかったパターン

なし。

## 単語の中に沈黙を混ぜて言う（無音 2 ティック）

6 step の語の 2 音素目のあと、区間が閉じない無音 2 ティックを挟む。

| 話した語 | 位相 | 出た語 | 正誤 | 生コスト | 下駄後 | 返事 | 理由 |
|---|---:|---|---|---:|---:|---|---|
| red_cube | 0 | red_cube | 正 | 0.000 | 0.000 | — |  |
| red_sphere | 0 | red_sphere | 正 | 0.000 | 0.000 | — |  |
| red_cylinder | 0 | red_cylinder | 正 | 0.000 | 0.000 | — |  |
| red_cone | 0 | red_cone | 正 | 0.000 | 0.000 | — |  |
| blue_cube | 0 | blue_cube | 正 | 0.000 | 0.000 | — |  |
| blue_sphere | 0 | blue_sphere | 正 | 0.000 | 0.000 | — |  |
| blue_cylinder | 0 | blue_cylinder | 正 | 0.000 | 0.000 | — |  |
| blue_cone | 0 | blue_cone | 正 | 0.000 | 0.000 | — |  |
| green_cube | 0 | green_cube | 正 | 0.000 | 0.000 | — |  |
| green_sphere | 0 | green_sphere | 正 | 0.000 | 0.000 | — |  |
| green_cylinder | 0 | green_cylinder | 正 | 0.000 | 0.000 | — |  |
| green_cone | 0 | green_cone | 正 | 0.000 | 0.000 | — |  |
| yellow_cube | 0 | yellow_cube | 正 | 0.000 | 0.000 | — |  |
| yellow_sphere | 0 | yellow_sphere | 正 | 0.000 | 0.000 | — |  |
| yellow_cylinder | 0 | yellow_cylinder | 正 | 0.000 | 0.000 | — |  |
| yellow_cone | 0 | yellow_cone | 正 | 0.000 | 0.000 | — |  |
| red_board | 0 | red_board | 正 | 0.000 | 0.000 | — |  |
| blue_board | 0 | blue_board | 正 | 0.000 | 0.000 | — |  |
| green_board | 0 | green_board | 正 | 0.000 | 0.000 | — |  |
| yellow_board | 0 | yellow_board | 正 | 0.000 | 0.000 | — |  |
| praise | 0 | praise | 正 | 0.000 | 0.000 | — |  |
| recast | 0 | recast | 正 | 0.000 | 0.000 | — |  |
| correct | 0 | correct | 正 | 0.000 | 0.000 | — |  |
| douzo | 0 | douzo | 正 | 0.000 | 0.000 | — |  |
| declare | 0 | declare | 正 | 0.000 | 0.000 | — |  |
| prompt | 0 | prompt | 正 | 0.000 | 0.000 | — |  |

### 認識できなかったパターン

なし。

## 単語の中に沈黙を混ぜて言う（無音 4 ティック）

6 step の語の 2 音素目のあと、区間が閉じる無音 4 ティックを挟む。割れたあとの 3 音は別の語になり得る。

| 話した語 | 位相 | 出た語 | 正誤 | 生コスト | 下駄後 | 返事 | 理由 |
|---|---:|---|---|---:|---:|---|---|
| red_cube | 0 | red_cube | 正 | 2.000 | 2.000 | — |  |
| red_sphere | 0 | blue_sphere | 誤 | 2.000 | 2.000 | — | 誤認識 |
| red_cylinder | 0 | red_cylinder | 正 | 2.000 | 2.000 | — |  |
| red_cone | 0 | red_cone | 正 | 2.000 | 2.000 | — |  |
| blue_cube | 0 | blue_cube | 正 | 2.000 | 2.000 | — |  |
| blue_sphere | 0 | blue_sphere | 正 | 2.000 | 2.000 | — |  |
| blue_cylinder | 0 | blue_cylinder | 正 | 2.000 | 2.000 | — |  |
| blue_cone | 0 | blue_cone | 正 | 2.000 | 2.000 | — |  |
| green_cube | 0 | declare | 誤 | 2.000 | 2.000 | — | 誤認識 |
| green_sphere | 0 | green_sphere | 正 | 2.000 | 2.000 | — |  |
| green_cylinder | 0 | blue_cylinder | 誤 | 2.000 | 2.000 | — | 誤認識 |
| green_cone | 0 | green_cone | 正 | 2.000 | 2.000 | — |  |
| yellow_cube | 0 | yellow_cube | 正 | 2.000 | 2.000 | — |  |
| yellow_sphere | 0 | yellow_sphere | 正 | 2.000 | 2.000 | — |  |
| yellow_cylinder | 0 | yellow_cylinder | 正 | 2.000 | 2.000 | — |  |
| yellow_cone | 0 | blue_cube | 誤 | 2.000 | 2.000 | — | 誤認識 |
| red_board | 0 | red_board | 正 | 2.000 | 2.000 | — |  |
| blue_board | 0 | blue_board | 正 | 2.000 | 2.000 | — |  |
| green_board | 0 | green_board | 正 | 2.000 | 2.000 | — |  |
| yellow_board | 0 | yellow_board | 正 | 2.000 | 2.000 | — |  |
| praise | 0 | praise | 正 | 2.000 | 2.000 | — |  |
| recast | 0 | recast | 正 | 2.000 | 2.000 | — |  |
| correct | 0 | correct | 正 | 2.000 | 2.000 | — |  |
| douzo | 0 | douzo | 正 | 2.000 | 2.000 | — |  |
| declare | 0 | blue_cone | 誤 | 2.000 | 2.000 | — | 誤認識 |
| prompt | 0 | prompt | 正 | 2.000 | 2.000 | — |  |

### 認識できなかったパターン

| 話した語 | 位相 | 出た語 | 正誤 | 生コスト | 下駄後 | 返事 | 理由 |
|---|---:|---|---|---:|---:|---|---|
| red_sphere | 0 | blue_sphere | 誤 | 2.000 | 2.000 | — | 誤認識 |
| green_cube | 0 | declare | 誤 | 2.000 | 2.000 | — | 誤認識 |
| green_cylinder | 0 | blue_cylinder | 誤 | 2.000 | 2.000 | — | 誤認識 |
| yellow_cone | 0 | blue_cube | 誤 | 2.000 | 2.000 | — | 誤認識 |
| declare | 0 | blue_cone | 誤 | 2.000 | 2.000 | — | 誤認識 |

## 単語の一部のみを短く言う

3 音素目だけ 3 step。他は 6 step。

| 話した語 | 位相 | 出た語 | 正誤 | 生コスト | 下駄後 | 返事 | 理由 |
|---|---:|---|---|---:|---:|---|---|
| red_cube | 0 | red_cube | 正 | 0.000 | 0.000 | — |  |
| red_sphere | 0 | red_sphere | 正 | 0.000 | 0.000 | — |  |
| red_cylinder | 0 | red_cylinder | 正 | 0.000 | 0.000 | — |  |
| red_cone | 0 | red_cone | 正 | 0.000 | 0.000 | — |  |
| blue_cube | 0 | blue_cube | 正 | 0.000 | 0.000 | — |  |
| blue_sphere | 0 | blue_sphere | 正 | 0.000 | 0.000 | — |  |
| blue_cylinder | 0 | blue_cylinder | 正 | 0.000 | 0.000 | — |  |
| blue_cone | 0 | blue_cone | 正 | 0.000 | 0.000 | — |  |
| green_cube | 0 | green_cube | 正 | 0.000 | 0.000 | — |  |
| green_sphere | 0 | green_sphere | 正 | 0.000 | 0.000 | — |  |
| green_cylinder | 0 | green_cylinder | 正 | 0.000 | 0.000 | — |  |
| green_cone | 0 | green_cone | 正 | 0.000 | 0.000 | — |  |
| yellow_cube | 0 | yellow_cube | 正 | 0.000 | 0.000 | — |  |
| yellow_sphere | 0 | yellow_sphere | 正 | 0.000 | 0.000 | — |  |
| yellow_cylinder | 0 | yellow_cylinder | 正 | 0.000 | 0.000 | — |  |
| yellow_cone | 0 | yellow_cone | 正 | 0.000 | 0.000 | — |  |
| red_board | 0 | red_board | 正 | 0.000 | 0.000 | — |  |
| blue_board | 0 | blue_board | 正 | 0.000 | 0.000 | — |  |
| green_board | 0 | green_board | 正 | 0.000 | 0.000 | — |  |
| yellow_board | 0 | yellow_board | 正 | 0.000 | 0.000 | — |  |
| praise | 0 | praise | 正 | 0.000 | 0.000 | — |  |
| recast | 0 | recast | 正 | 0.000 | 0.000 | — |  |
| correct | 0 | correct | 正 | 0.000 | 0.000 | — |  |
| douzo | 0 | douzo | 正 | 0.000 | 0.000 | — |  |
| declare | 0 | declare | 正 | 0.000 | 0.000 | — |  |
| prompt | 0 | prompt | 正 | 0.000 | 0.000 | — |  |

### 認識できなかったパターン

なし。

## 単語の一部のみを長く言う

3 音素目だけ 9 step。他は 6 step。

| 話した語 | 位相 | 出た語 | 正誤 | 生コスト | 下駄後 | 返事 | 理由 |
|---|---:|---|---|---:|---:|---|---|
| red_cube | 0 | red_cube | 正 | 0.000 | 0.000 | — |  |
| red_sphere | 0 | red_sphere | 正 | 0.000 | 0.000 | — |  |
| red_cylinder | 0 | red_cylinder | 正 | 0.000 | 0.000 | — |  |
| red_cone | 0 | red_cone | 正 | 0.000 | 0.000 | — |  |
| blue_cube | 0 | blue_cube | 正 | 0.000 | 0.000 | — |  |
| blue_sphere | 0 | blue_sphere | 正 | 0.000 | 0.000 | — |  |
| blue_cylinder | 0 | blue_cylinder | 正 | 0.000 | 0.000 | — |  |
| blue_cone | 0 | blue_cone | 正 | 0.000 | 0.000 | — |  |
| green_cube | 0 | green_cube | 正 | 0.000 | 0.000 | — |  |
| green_sphere | 0 | green_sphere | 正 | 0.000 | 0.000 | — |  |
| green_cylinder | 0 | green_cylinder | 正 | 0.000 | 0.000 | — |  |
| green_cone | 0 | green_cone | 正 | 0.000 | 0.000 | — |  |
| yellow_cube | 0 | yellow_cube | 正 | 0.000 | 0.000 | — |  |
| yellow_sphere | 0 | yellow_sphere | 正 | 0.000 | 0.000 | — |  |
| yellow_cylinder | 0 | yellow_cylinder | 正 | 0.000 | 0.000 | — |  |
| yellow_cone | 0 | yellow_cone | 正 | 0.000 | 0.000 | — |  |
| red_board | 0 | red_board | 正 | 0.000 | 0.000 | — |  |
| blue_board | 0 | blue_board | 正 | 0.000 | 0.000 | — |  |
| green_board | 0 | green_board | 正 | 0.000 | 0.000 | — |  |
| yellow_board | 0 | yellow_board | 正 | 0.000 | 0.000 | — |  |
| praise | 0 | praise | 正 | 0.000 | 0.000 | — |  |
| recast | 0 | recast | 正 | 0.000 | 0.000 | — |  |
| correct | 0 | correct | 正 | 0.000 | 0.000 | — |  |
| douzo | 0 | douzo | 正 | 0.000 | 0.000 | — |  |
| declare | 0 | declare | 正 | 0.000 | 0.000 | — |  |
| prompt | 0 | prompt | 正 | 0.000 | 0.000 | — |  |

### 認識できなかったパターン

なし。

## 単語の一部のみ欠損

3 音素目を抜く。残り 4 音素は 6 step。黙ったときに判定する。

| 話した語 | 位相 | 出た語 | 正誤 | 生コスト | 下駄後 | 返事 | 理由 |
|---|---:|---|---|---:|---:|---|---|
| red_cube | 0 | red_cube | 正 | 1.000 | 1.000 | — |  |
| red_sphere | 0 | red_sphere | 正 | 1.000 | 1.000 | — |  |
| red_cylinder | 0 | red_cylinder | 正 | 1.000 | 1.000 | — |  |
| red_cone | 0 | red_cone | 正 | 1.000 | 1.000 | — |  |
| blue_cube | 0 | blue_cube | 正 | 1.000 | 1.000 | — |  |
| blue_sphere | 0 | blue_sphere | 正 | 1.000 | 1.000 | — |  |
| blue_cylinder | 0 | blue_cylinder | 正 | 1.000 | 1.000 | — |  |
| blue_cone | 0 | blue_cone | 正 | 1.000 | 1.000 | — |  |
| green_cube | 0 | green_cube | 正 | 1.000 | 1.000 | — |  |
| green_sphere | 0 | green_sphere | 正 | 1.000 | 1.000 | — |  |
| green_cylinder | 0 | green_cylinder | 正 | 1.000 | 1.000 | — |  |
| green_cone | 0 | green_cone | 正 | 1.000 | 1.000 | — |  |
| yellow_cube | 0 | yellow_cube | 正 | 2.000 | 2.000 | — |  |
| yellow_sphere | 0 | yellow_sphere | 正 | 1.000 | 1.000 | — |  |
| yellow_cylinder | 0 | yellow_cylinder | 正 | 1.000 | 1.000 | — |  |
| yellow_cone | 0 | yellow_cone | 正 | 1.000 | 1.000 | — |  |
| red_board | 0 | red_board | 正 | 2.000 | 2.000 | — |  |
| blue_board | 0 | blue_board | 正 | 1.000 | 1.000 | — |  |
| green_board | 0 | green_board | 正 | 1.000 | 1.000 | — |  |
| yellow_board | 0 | yellow_board | 正 | 1.000 | 1.000 | — |  |
| praise | 0 | praise | 正 | 1.000 | 1.000 | — |  |
| recast | 0 | recast | 正 | 1.000 | 1.000 | — |  |
| correct | 0 | correct | 正 | 2.000 | 2.000 | — |  |
| douzo | 0 | douzo | 正 | 1.000 | 1.000 | — |  |
| declare | 0 | declare | 正 | 1.000 | 1.000 | — |  |
| prompt | 0 | prompt | 正 | 1.000 | 1.000 | — |  |

### 認識できなかったパターン

なし。

## 異なる単語を連続して話す（6 step）

無音なし。全 676 組。6 音目で先の語を切り、残りから次の語を取る。

成功 676 / 676。 生コストは最小 0.000、最大 0.000。 認識できなかった組はない。

## 異なる単語を連続して話す（9 step）

無音なし。全 676 組。畳み込みで 6 step と同じ長さになる。

成功 676 / 676。 生コストは最小 0.000、最大 0.000。 認識できなかった組はない。

## 対象を red_cube に固定した返事

完全一致は「すごい」、置換 1 つと連続 3 音は「おしい」、無関係は「ちがうよ」、1 音と 2 音は返事なし。

| 話した語 | 位相 | 出た語 | 正誤 | 生コスト | 下駄後 | 返事 | 理由 |
|---|---:|---|---|---:|---:|---|---|
| 完全一致 | 0 | red_cube | 正 | 0.000 | -1.000 | sugoi |  |
| 置換1 | 0 | red_cube | 正 | 1.000 | 1.000 | oshii |  |
| 連続3音 | 0 | red_cube | 正 | 2.000 | 1.000 | oshii |  |
| 無関係 | 0 | prompt | 正 | 0.000 | 0.000 | chigauyo |  |
| 1音 | 0 | — | 正 | — | — | — |  |
| 2音 | 0 | — | 正 | — | — | — |  |

### 認識できなかったパターン

なし。
