# cueworld ドキュメント

`cueworld` の仕様の正本。**実装されている振る舞いだけを書く**方針で、1 つの事実は 1 か所にだけ置く。重複しそうな箇所はリンクで済ませる。

外部向け（環境の全体像を 1 本で説明するもの）は [overview.md](overview.md)。

読み始めるなら [env.md](env.md) から。スタック全体の構成表がある。

## 下から上へ

| ドキュメント | 対象 |
|-------------|------|
| [env.md](env.md) | `BaseEnv`。座標系・行動空間・観測空間・接触と把持・`step()` の処理順 |
| [object.md](object.md) | 物体の識別属性・寸法・アフォーダンス |
| [position.md](position.md) | 初期配置の方式とパラメータ |
| [language.md](language.md) | 音素・語彙・発話テンプレート・音声認識 |
| [parent_tools.md](parent_tools.md) | 親の行動生成ツール（視線・移動・腕・掲げる・振る） |
| [infant_signal.md](infant_signal.md) | 幼児シグナルの原子キーとその定義 |
| [intent.md](intent.md) | 原子シグナルから合成する訴えの 0/1 セット |
| [protocol/README.md](protocol/README.md) | プロトコル共通契約。個別 5 本への入口 |
| [protocol/automata/](protocol/automata/README.md) | 各プロトコルと切り替え判定の状態遷移図 |
| [protocol_manager.md](protocol_manager.md) | 親の意思決定。切り替え・親始動・学習の記録 |

各ドキュメントの末尾には「注意事項」の節があり、**実装上の不具合・未使用の経路・暫定値**をまとめている。仕様として当てにする前にそこを確認すること。
