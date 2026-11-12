"""
シミュレーション環境内の単語・概念・音素を対応づける辞書ファイル

CONCEPT_TO_PHONEMES は識別用の内部コード（固定長 5。
同じ音素は隣り合わず、先頭と末尾の音素集合は交わらない。
ハミング距離は 4 を優先し、入りきらない組だけ 3）。
RESERVED_PHONEME_CODES は認識候補に入れない予備符号。
人間向け表示は CONCEPT_TO_HUMAN_WORD を使う。
生成: scripts/experiment/optimize_vocabulary.py --write
"""

CONCEPT_TO_PHONEMES = {
    # オブジェクト
    "red_cube": ["p2", "p0", "p4", "p3", "p6"],
    "red_sphere": ["p0", "p1", "p3", "p2", "p7"],
    "red_cylinder": ["p1", "p4", "p2", "p5", "p6"],
    "red_cone": ["p3", "p5", "p4", "p6", "p5"],
    "blue_cube": ["p1", "p7", "p5", "p1", "p7"],
    "blue_sphere": ["p3", "p2", "p7", "p3", "p4"],
    "blue_cylinder": ["p0", "p6", "p0", "p7", "p4"],
    "blue_cone": ["p2", "p6", "p2", "p4", "p5"],
    "green_cube": ["p3", "p7", "p0", "p2", "p6"],
    "green_sphere": ["p1", "p3", "p4", "p2", "p4"],
    "green_cylinder": ["p2", "p7", "p6", "p0", "p4"],
    "green_cone": ["p2", "p3", "p7", "p6", "p7"],
    "yellow_cube": ["p0", "p3", "p5", "p3", "p5"],
    "yellow_sphere": ["p0", "p5", "p7", "p1", "p6"],
    "yellow_cylinder": ["p3", "p0", "p6", "p4", "p7"],
    "yellow_cone": ["p1", "p2", "p1", "p7", "p5"],
    "red_board": ["p2", "p5", "p1", "p5", "p4"],
    "blue_board": ["p3", "p1", "p5", "p7", "p6"],
    "green_board": ["p3", "p4", "p3", "p0", "p5"],
    "yellow_board": ["p1", "p0", "p3", "p6", "p4"],

    # アクション・状態・プロトコル用語
    "praise": ["p0", "p2", "p4", "p5", "p7"],
    "recast": ["p1", "p5", "p0", "p3", "p7"],
    "correct": ["p2", "p1", "p0", "p1", "p5"],
    "douzo": ["p0", "p3", "p1", "p0", "p6"],
    "declare": ["p0", "p7", "p2", "p6", "p4"],
    "prompt": ["p3", "p6", "p1", "p2", "p7"],
}

# 認識候補に入れない予備符号。あとから概念を足すときに名前を付ける。
RESERVED_PHONEME_CODES = [
    ["p1", "p6", "p7", "p4", "p6"],
    ["p2", "p4", "p5", "p2", "p4"],
    ["p0", "p5", "p6", "p2", "p5"],
    ["p3", "p1", "p2", "p1", "p4"],
    ["p2", "p7", "p3", "p7", "p6"],
    ["p1", "p0", "p7", "p0", "p5"],
    ["p0", "p4", "p0", "p4", "p7"],
    ["p1", "p2", "p6", "p1", "p6"],
    ["p2", "p5", "p2", "p0", "p7"],
    ["p3", "p6", "p3", "p5", "p4"],
]

# 概念ID -> 人間にわかりやすい日本語単語（ログ出力やプロトコルのテンプレート等に使用）
CONCEPT_TO_HUMAN_WORD = {
    # オブジェクト
    "red_cube": "あかいしかく",
    "red_sphere": "あかいまる",
    "red_cylinder": "あかいえんちゅう",
    "red_cone": "あかいさんかく",
    "blue_cube": "あおいしかく",
    "blue_sphere": "あおいまる",
    "blue_cylinder": "あおいえんちゅう",
    "blue_cone": "あおいさんかく",
    "green_cube": "みどりのしかく",
    "green_sphere": "みどりのまる",
    "green_cylinder": "みどりのえんちゅう",
    "green_cone": "みどりのさんかく",
    "yellow_cube": "きいろいしかく",
    "yellow_sphere": "きいろいまる",
    "yellow_cylinder": "きいろいえんちゅう",
    "yellow_cone": "きいろいさんかく",
    "red_board": "あかいぼーど",
    "blue_board": "あおいぼーど",
    "green_board": "みどりのぼーど",
    "yellow_board": "きいろいぼーど",

    # アクション・状態・プロトコル用語
    "praise": "すごい",
    "recast": "そうだね",
    "correct": "ちがうよ",
    "douzo": "どうぞ",
    "declare": "これ",
    "prompt": "ねえ",
}
