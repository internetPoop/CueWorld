from typing import List, Dict, Optional, Any, Tuple
import warnings
from .vocabulary import CONCEPT_TO_PHONEMES, CONCEPT_TO_HUMAN_WORD
from .templates import PROTOCOL_TEMPLATES

# 1 単語 = 5 音素 = 30 フレーム（dyadic の MOTOR_CYCLE_STEPS=30 と同単位）。
# 意図的に導出せず独立定数にする（language が protocols に依存しないため）。
PHONEME_DURATION_FRAMES = 6
# 単語境界のソフト区切り。観測上ゼロベクトルになれば足りる長さで、
# VoiceRecognizer のセグメント分割（offset_silence_frames）用ではない。
INTER_WORD_GAP_FRAMES = 1


class LanguageManager:
    """
    シミュレーション環境における言語（単語、音素、テンプレート）の管理を行うクラス。
    """
    def __init__(self):
        self.concept_to_phonemes = CONCEPT_TO_PHONEMES
        self.concept_to_human_word = CONCEPT_TO_HUMAN_WORD
        self.templates = PROTOCOL_TEMPLATES
        
        # infant_signal 用の 日本語文字列 -> 音素リスト の辞書を事前計算
        self._human_word_to_phonemes = CONCEPT_TO_PHONEMES
        # for concept_id, word in self.concept_to_human_word.items():
        #     if concept_id in self.concept_to_phonemes:
        #         self._human_word_to_phonemes[word] = self.concept_to_phonemes[concept_id]

    def get_word_to_phonemes_dict(self) -> Dict[str, List[str]]:
        """
        infant_signal.py などの認識モジュール向けに、
        「日本語文字列 -> 音素のリスト」のマッピングを返す。
        """
        return self._human_word_to_phonemes

    def get_concept_for_object(self, obj: Any) -> Optional[str]:
        """
        オブジェクトインスタンスから概念ID（concept_id）を取得する。
        オブジェクト自身が concept_id を持っていればそれを優先し、
        なければクラス名から推論する。
        """
        return obj.concept_id
        

    def get_human_word(self, concept_id: str) -> str:
        """
        概念IDから日本語文字列を取得する。
        未登録の場合は概念IDをそのまま返す。
        """
        return self.concept_to_human_word.get(concept_id, concept_id)

    def get_phonemes(self, concept_id: str) -> List[str]:
        """
        概念IDから音素のリストを取得する。
        """
        return self.concept_to_phonemes.get(concept_id, [])

    def get_template_phonemes(self, protocol_name: str, template_key: str, concept_id: str = None) -> List[Tuple[str, int]]:
        """
        プロトコル名とテンプレートキーを指定し、音素と継続時間のタプルリストを生成する。

        テンプレートは概念ID列（例: ["praise", "TARGET"]）。
        "TARGET" は concept_id に差し替え、各概念を語彙の固定長音素列へ展開する。
        各音素の継続時間は PHONEME_DURATION_FRAMES、単語間には
        INTER_WORD_GAP_FRAMES フレームの "none"（無声）を挿入する。
        """
        protocol_templates = self.templates.get(protocol_name, {})
        template_list = protocol_templates.get(template_key)
        
        if not template_list:
            # 見つからなければ common から探す
            template_list = self.templates.get("common", {}).get(template_key)
            
        if not template_list:
            warnings.warn(
                f"Unknown utterance template: protocol={protocol_name!r}, key={template_key!r}",
                stacklevel=2,
            )
            return []
            
        result: List[Tuple[str, int]] = []
        words_emitted = 0
        for item in template_list:
            resolved_id = concept_id if item == "TARGET" else item
            if resolved_id is None:
                warnings.warn(
                    f"TARGET in template {template_key!r} but concept_id is None; skipping slot",
                    stacklevel=2,
                )
                continue
            phonemes = self.get_phonemes(resolved_id)
            if not phonemes:
                warnings.warn(
                    f"Unknown concept_id {resolved_id!r} in template {template_key!r}; skipping",
                    stacklevel=2,
                )
                continue
            if words_emitted > 0 and INTER_WORD_GAP_FRAMES > 0:
                result.append(("none", INTER_WORD_GAP_FRAMES))
            for p in phonemes:
                result.append((p, PHONEME_DURATION_FRAMES))
            words_emitted += 1
                
        return result
