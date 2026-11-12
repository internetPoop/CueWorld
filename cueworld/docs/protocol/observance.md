# 見守り (`protocols/observance.py`)

親が何もせず幼児を見ているだけの基本状態。他に選ぶものがないときに PM が入れる待機プロトコル。

共通契約は [README.md](README.md)。

| 属性 | 値 |
|------|-----|
| `solo` | `True` |
| `has_target` | `False`（基底の既定） |
| `uses_pre_release` | `False` |
| `persistence` | 常に 1.0 |
| `state_proximities` | `{"observing": 1.0}` |

---

## 1. 振る舞い

状態は `observing` ひとつだけで、遷移も終了もしない。毎 step、

- `gaze_agent` で顔を幼児へ向ける
- `reset_arm_pose` で腕を初期姿勢へ戻す

`is_finished` は立たないので、**PM が別のプロトコルへ切り替えるまで続く**。

`state_proximities` が最初から 1.0 なのは、「見守りは常に受理状態にある」＝ 途中で終わっても失敗にならない、という意味。

---

## 2. `uses_pre_release = False`

開始時に物を持っていても、[README.md §4](README.md#4-開始時の手放し) の `pre_release` 状態を通らない。`reset_arm_pose` が指を開くので、放っておけば自然に落ちるため。

---

## 3. `persistence = 1.0`

`_calculate_persistence` をオーバーライドして固定値 1.0 を返す。カタログ中で最も低く、**どんな割り込みにも譲る**。幼児が何か信号を出せば、PM は迷わず他のプロトコルへ切り替えられる。

`HouseholdSoloProtocol` の 5.0（[household_solo.md §5](household_solo.md#5-pm-との関係)）と対になっていて、「親が手を空けて待っている状態」と「親が家事に没頭している状態」を作り分ける。

---

## 4. 注意事項

**`_calculate_persistence` に調整余地の TODO が残っている。**
1.0 という値は暫定。

**幼児の位置に関わらず接近しない。**
遠くにいても顔を向けるだけで、近づく動作は入らない。
