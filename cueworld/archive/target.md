# ターゲットオブジェクトの決定ロジックに関する評価と実装案

## 1. 客観的評価

ターゲットの決定を `ProtocolManager` から各プロトコルクラス内部へ移動させる設計は、**非常に適切（Highly Appropriate）** であり、推奨される変更です。

### 評価の理由（メリット）

1.  **責務の分離（カプセル化の向上）**
    `ProtocolManager` の本来の役割は「どのプロトコルを（いつ）起動するか」の決定と管理です。「そのプロトコルで具体的に何を使うか」はプロトコル内部の詳細な実行コンテキストであるため、プロトコル自身が管理するべきです。現状の `Manager` がターゲットを決定する実装は、責務の漏出（Leakage of Responsibility）を起こしています。
2.  **プロトコル固有のアフォーダンス・制約への対応**
    プロトコルによってターゲットにできるオブジェクトの条件は全く異なります。
    *   `JointAttentionProtocol`: 視界内にある任意の目立つオブジェクト。
    *   `ObjectGivingProtocol (feeding=True)`: `is_edible`（食べられる）属性を持つオブジェクトのみ。
    *   `HouseholdSoloProtocol`: 片付けるべき散らかったオブジェクトや、特定のコンテナ。
    これらを `ProtocolManager` で一括管理しようとすると、巨大で複雑な分岐（スパゲッティコード）を生む原因になります。
3.  **プロトコル内での柔軟な状態遷移（リトライ・対象変更）**
    ご指摘の通り、「ターゲットへの接近に失敗した」「幼児が別のものを見始めた」といった場合に、プロトコルを完全に終了して `Observance` に戻るのではなく、**プロトコルの状態を維持したままターゲットだけを再選択する** という、より人間らしく柔軟なリカバリー動作が可能になります。

---

## 2. 実装案

ターゲット選択の権限をプロトコルに移譲するための具体的な実装ステップです。

### ステップ 1: `ProtocolManager` の修正
`check_for_response` および `check_for_initiative` 内で行っているターゲット選択ロジックを削除します。代わりに、生の `infant_signals` などをそのまま `context` に詰めて渡します。

```python
# protocol_manager.py の変更イメージ
def check_for_response(self, infant_signals: Dict[str, Any]):
    # ... プロトコルの選択 ...
    
    context = {}
    context["infant_initiative"] = True
    context["infant_signals"] = infant_signals # 信号をそのまま渡す
    
    # 以前ここにあった target_object の決定処理は全て削除
    
    return True, sampled_protocol_name, context
```

### ステップ 2: `BaseProtocol` にターゲット選択インターフェースを追加
すべてのプロトコルの基底クラスに、ターゲットを選択・設定するメソッドを定義します。

```python
# protocols/base.py の変更イメージ
class BaseProtocol:
    # ...
    def start(self, context: Dict[str, Any]):
        self.is_finished = False
        self.state = "init"
        self.infant_initiative = context.get("infant_initiative", False)
        self.utterance_queue.clear()
        
        # Manager から渡された信号を保持
        self.infant_signals = context.get("infant_signals", {})
        
        # ターゲットの初期選択
        if self.has_target:
            self.select_target()

    def select_target(self):
        """
        ターゲットオブジェクトを決定する。
        プロトコル固有の条件がある場合は、サブクラスでオーバーライドする。
        """
        # デフォルトの実装（幼児の注視を優先し、なければランダムなど）
        pass
```

### ステップ 3: 各プロトコルでのオーバーライドと再選択
特定の条件が必要なプロトコルでは、`select_target` をオーバーライドして独自のロジックを書きます。
また、プロトコル実行中（`_manage_state` 内）に失敗を検知した際、再びこのメソッドを呼ぶことでリトライを実現します。

```python
# protocols/object_giving.py の変更イメージ
class ObjectGivingProtocol(BaseProtocol):
    # ...
    def select_target(self):
        # feeding=True なら is_food 属性を持つものだけを抽出して選択
        # そうでなければ、幼児の手が届く範囲のおもちゃを選択
        pass
        
    def _manage_state(self, infant_signals):
        if self.state == "approaching" and self._is_blocked():
            # 到達できなければ別のターゲットに変えてやり直す
            self.select_target()
            if self.target_object is None:
                self.state = self.failure_state # 代わりがなければ失敗
```

### まとめ
この設計にすることで、`ProtocolManager` は非常にスッキリとし、各プロトコルのロジックは自己完結するため、並行開発やデバッグが圧倒的に容易になります。