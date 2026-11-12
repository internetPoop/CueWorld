"""
各プロトコルで使用する発話のテンプレート（フォーマット）を定義するファイル
プロトコル側ではこれらのテンプレートのキーを指定し、LanguageManagerを通じて
具体的な音素と継続時間のタプルリストを取得します。

テンプレートは概念ID列（機能語 → TARGET の 2 スロット、または単語）。
各概念は vocabulary の固定長 5 符号に展開され、音素長は manager が一括付与する。
"""

PROTOCOL_TEMPLATES = {
    # 共同注意 (JointAttentionProtocol)
    "joint_attention": {
        # これ X（ラベリング）
        "declare_object": ["declare", "TARGET"],
    },
    
    # 対象物授受 (ObjectGivingProtocol)
    "object_giving": {
        # どうぞ X
        "present": ["douzo", "TARGET"],
    },

    # 二項模倣 (DyadicProtocol)
    "dyadic": {
        # 促し
        "prompt": ["prompt"],
    },

    # 音声切り分け (VoiceEchoProtocol)
    "voice_echo": {
        # 見本の対象語だけ
        "target_only": ["TARGET"],
    },
    
    # 一般的・共通のリアクション
    "common": {
        # すごい X
        "praise_with_target": ["praise", "TARGET"],
        # そうだね X
        "recast_with_target": ["recast", "TARGET"],
        # ちがうよ X
        "correct_with_target": ["correct", "TARGET"],
        
        # 単体・対象なし版
        "praise": ["praise"],
        "recast": ["recast"],
        "correct": ["correct"],
        
        # 単語のみ
        "target_only": ["TARGET"],
    }
}
