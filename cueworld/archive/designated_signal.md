# 幼児のシグナル指定モード（Designated Signal Mode）の実装設計

## 1. 目的
テスト検証や分析のために、環境内で観測・評価される幼児のシグナルを限定・指定するモードを実装します。
例えば、「視線（`gaze_target_stability`）のみ」や「音声（`voice_target_stability`）のみ」を親が評価した場合に、どのようにプロトコル遷移が発生するかをテストできるようにします。
また、環境全体でシグナルの指定方法を単一の仕組みに統一し、設定の煩雑さや不整合を防ぎます。

## 2. 客観的評価と現状の課題
現状では、`InfantSignalDetector` 側で大まかなモダリティ単位（`active_modalities`）の制限を行い、`ProtocolManager` 側で個別のシグナル単位（`active_signals`）のマスクを行うという二重管理の構造になっています。
この構造には以下の課題があります。
- **設定の分散**: どのシグナルが有効になっているかを把握するために、複数のラッパーの設定を確認する必要がある。
- **重み行列（`weights_dict`）の破綻**: `InfantSignalDetector` で制限された結果、返却されるシグナルのキーが減ると、`ProtocolManager` が固定のインデックス順で持つ `weights_dict` の列数（次元）と一致しなくなり、行列計算がエラーを起こす。

これらの課題を解決するため、**「`InfantSignalDetector` をシグナル管理の Single Source of Truth（信頼できる唯一の情報源）とする」** アプローチは非常に適切です。`InfantSignalDetector` で具体的なシグナルキーのリストを一元管理し、`ProtocolManager` は環境のプロパティ（`self.env.active_signals`）を参照して動的に対応する設計へと移行します。

## 3. 実装方針

### 3.1. `InfantSignalDetector` の修正 (シグナルの一元管理)
- **`active_modalities` の廃止**: モダリティ単位の設定を廃止し、代わりに `active_signals: Optional[List[str]] = None` を `__init__` の引数として受け取ります。
- **シグナルのフィルタリング**: `get_normalized_signals()` 等でシグナルを生成・返却する際、`self.active_signals` に指定されているキーのシグナルのみを計算・出力します（`None` の場合は全シグナルを出力）。これにより不要な計算負荷も削減できます。

### 3.2. `ProtocolManager` の修正 (動的重み行列の構築)
- **引数の削除**: `__init__` 引数から `active_signals` を削除します。
- **環境からの参照**: ラップしている環境から `self.active_signals = getattr(self.env, "active_signals", None)` を参照して取得します。
- **動的な `weights_dict` の構築**: 
  リストではなく、シグナル名をキーとした辞書（`Dict[str, float]`）で各プロトコルの重みを定義します。
  取得した `self.active_signals` （または全シグナルのデフォルトリスト `self.signal_keys`）の順番に従って、動的に `weight_matrix` を構築します。
  
```python
  # 動的構築のイメージ
  effective_signals = self.active_signals if self.active_signals is not None else self.signal_keys
  
  weights_dict = {
      "joint_attention": {"point_intensity": 0.5, "gaze_target_stability": 0.5, ...},
      "object_giving": {"holding": -1.0, "arm_ext": 0.5, ...},
      # ...
  }
  
  weight_matrix = []
  for p_name in self.response_protocol:
      p_weights = weights_dict.get(p_name, {})
      # effective_signals の順に重みを抽出（未定義なら 0.0）
      row = [p_weights.get(sig_key, 0.0) for sig_key in effective_signals]
      weight_matrix.append(row)
      
  self.weights_for_strength_calc = np.array(weight_matrix)
```
これにより、`InfantSignalDetector` から出力されるシグナルの次元数がどう変化しても、常に対応する重み行列が正しく構築され、次元不一致のエラーが起きません。

## 4. 環境の初期化と利用方法
デモスクリプト等で環境を初期化する際、ラッパーの順番と引数は以下のようになります。

```python
# 1. BaseEnv の初期化
env = BaseEnv(...)

# 2. InfantSignalDetector の初期化（ここでシグナルを指定）
# 例: 視線と音声の安定性のみを有効にする
env = InfantSignalDetector(env, active_signals=["gaze_target_stability", "voice_target_stability"])

# 3. ProtocolManager の初期化（引数でのシグナル指定は不要）
env = ProtocolManager(env, demo_mode=False)

# ProtocolManager 内部では env.active_signals が自動的に参照され、
# 指定されたシグナルのみで構成される重み行列が動的に構築される。
```

## 5. InfantSignalDetector における内部処理の最適化についての考察
### 5.1. 現状の実装が「モダリティ単位での計算・更新」を残している理由 
InfantSignalDetector の現状の実装では、最終的な出力は active_signals でフィルタリングされるものの、内部の履歴更新（update）やシグナル生成（get_normalized_signals）は _is_modality_active を介して「モダリティ単位」でまとめて実行されています。 これを行っている理由は以下の通りです。
1. 計算と履歴データの強い依存関係:

例えば gesture モダリティにおける hand_positions（手の位置履歴）や hand_targets（対象物履歴）は、point_intensity、hand_movement_size、hand_target_stability といった複数のシグナル計算で共有されています。シグナルキー単位で細かく計算を分離しようとすると、「どの履歴データを更新すべきか」「ベースとなる計算（get_hand_statusなど）をいつ呼ぶべきか」の依存関係の管理が非常に複雑になります。 
2. 実装の安全性とシンプルさ:

既存のコードがモダリティ単位で機能するように作られていたため、バグのリスクを最小限に抑えつつ出力を指定のシグナルに制限するための措置です。大枠でモダリティごとの計算を走らせ、最後に不要なキーを捨てるアプローチを取っています。
<!-- ### 5.2. active_signals に完全準拠した最適化の実装計画 
もし、将来的にパフォーマンス向上（計算コストの削減）などを目的として、指定された active_signals に必要な処理だけを厳密に実行するように改修する場合、以下のような実装方針になります。 
1. 履歴更新（update）の依存関係の明示:

毎ステップの update で一律に履歴を保存するのではなく、「各シグナルがどの履歴データを要求するか」のマッピングを作成します。
self.active_signals に含まれるシグナル群が要求する履歴データ（例: hand_positions や phoneme_history）のみを毎ステップ更新するように update を修正します。 
2. 個別のシグナル計算メソッドへの分割とキャッシュ:
現在の get_gesture_signals() のように複数のシグナルを一括で計算する構造から、独立した計算メソッドへ分割します。
レイキャストや音声のDPマッチングなど、複数のシグナルで共通して重い計算を行う処理は、1ステップごとのキャッシュ（メモ化）を導入して重複計算を防ぎます。 
3. get_normalized_signals のディスパッチャ化:
active_signals に登録されたキーに対応するメソッドのみを呼び出す構造に変更します。 -->
