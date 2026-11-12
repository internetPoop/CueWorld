import random
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from .base import BaseProtocol
from cueworld.household_schemas import (
    TASK_SCHEMAS,
    HOUSEHOLD_STYLE_PARAMS,
    HOUSEHOLD_INSPECTION_PARAMS,
    ITEM_TYPICALITY_BY_CLASS,
    DEFAULT_ITEM_TYPICALITY,
)


class HouseholdSoloProtocol(BaseProtocol):
    """家事・単独プロトコル（board ベース配置）"""
    def __init__(self, env: Any):
        super().__init__("HouseholdSoloProtocol", env)

        # プロトコルのメタデータ（属性）
        self.function_type = "declarative"
        self.relation_type = "solo"
        self.is_urgent = False
        self.solo = True
        self.has_target = False  # 家事対象は board / item 側。target_object は使わない
        self.parent_utterance = False  # 家事中は発話しない

        self.state_proximities = {
            "init": 0.0,
            "inspecting_item": 0.2,
            "hesitating": 0.22,
            "moving_to_item": 0.3,
            "grasping_item": 0.5,
            "inspecting_board": 0.65,
            "moving_to_board": 0.7,
            "placing_on_board": 0.85,
            "check_done": 0.9,
            "completed": 1.0,
        }

        self.state_timer = 0
        self.global_intent: Optional[str] = None
        self.household_style = "committed"
        self.inspection_params: Dict[str, Any] = dict(HOUSEHOLD_INSPECTION_PARAMS)
        self.hesitation_alt_item_id: Optional[int] = None
        self.inspection_history: List[Dict[str, Any]] = []
        self._hesitation_next_state = "moving_to_item"
        self.target_board_id: Optional[int] = None
        self.candidate_items: List[int] = []
        self.current_item_id: Optional[int] = None
        self.items_placed = 0
        self.last_item_selection: Dict[str, Any] = {}
        self.item_selection_history: List[Dict[str, Any]] = []
        self.state = "init"

    def _item_typicality_score(self, item_id: int) -> float:
        obj = self.env.all_objects[item_id]
        class_name = obj.__class__.__name__
        return ITEM_TYPICALITY_BY_CLASS.get(class_name, DEFAULT_ITEM_TYPICALITY)

    def select_next_item(
        self,
        item_ids: List[int],
        style: str,
        rng: Optional[random.Random] = None,
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        """距離ソフトマックス + 典型性バイアス + style 重みで次の item を確率的に選ぶ。"""
        if not item_ids:
            return None, {}

        rng = rng or random
        style_cfg = HOUSEHOLD_STYLE_PARAMS.get(style, HOUSEHOLD_STYLE_PARAMS["committed"])
        dist_weight = style_cfg["distance_weight"]
        typ_weight = style_cfg["typicality_weight"]
        temperature = style_cfg["temperature"]

        logits: List[float] = []
        entries: List[Dict[str, Any]] = []
        for oid in item_ids:
            dist = self.env._dist_body_to(self.env.parent, self.env._obj_pos(oid))
            dist_score = float(np.exp(-dist / temperature))
            typ_score = self._item_typicality_score(oid)
            logit = dist_weight * np.log(dist_score + 1e-9) + typ_weight * np.log(typ_score + 1e-9)
            logits.append(float(logit))
            entries.append({
                "item_id": oid,
                "name": self.env.all_objects[oid].name,
                "distance": float(dist),
                "dist_score": dist_score,
                "typicality": typ_score,
                "logit": float(logit),
            })

        logits_arr = np.array(logits, dtype=np.float64)
        logits_arr -= logits_arr.max()
        probs = np.exp(logits_arr)
        probs /= probs.sum()

        chosen_idx = rng.choices(range(len(item_ids)), weights=probs.tolist(), k=1)[0]
        chosen_id = item_ids[chosen_idx]

        reason = {
            "style": style,
            "candidates": entries,
            "probabilities": probs.tolist(),
            "chosen_id": chosen_id,
            "chosen_name": self.env.all_objects[chosen_id].name,
        }
        prob_summary = {
            entry["name"]: round(float(prob), 3)
            for entry, prob in zip(entries, probs.tolist())
        }
        print(
            f"[HouseholdSoloProtocol] select_next_item: chosen={reason['chosen_name']} "
            f"(id={chosen_id}) style={style} probs={prob_summary}"
        )
        for entry, prob in zip(entries, probs.tolist()):
            print(
                f"  - {entry['name']}: dist={entry['distance']:.2f} "
                f"dist_score={entry['dist_score']:.3f} typicality={entry['typicality']:.2f} "
                f"p={prob:.3f}"
            )
        return chosen_id, reason

    def _is_item_done(self, item_id: int) -> bool:
        if self.target_board_id is None:
            return False
        board_obj = self.env.all_objects[self.target_board_id]
        item_obj = self.env.all_objects[item_id]
        if item_id in board_obj.objects_on_board:
            return True
        if getattr(item_obj, "put_on", None) == self.target_board_id:
            return True
        return False

    def _remaining_item_candidates(self) -> List[int]:
        return [oid for oid in self.candidate_items if not self._is_item_done(oid)]

    def _record_inspection(self, kind: str, **extra: Any) -> None:
        entry = {
            "kind": kind,
            "global_intent": self.global_intent,
            "current_item_id": self.current_item_id,
            "target_board_id": self.target_board_id,
            **extra,
        }
        self.inspection_history.append(entry)
        print(f"[HouseholdSoloProtocol] enter {kind}: {extra}")

    def _start_inspecting_item(self) -> None:
        item_name = self.env.all_objects[self.current_item_id].name
        duration = int(self.inspection_params["inspect_item_duration"])
        self._record_inspection(
            "inspecting_item",
            target_id=self.current_item_id,
            target_name=item_name,
            duration_steps=duration,
        )
        self.state = "inspecting_item"
        self.state_timer = 0

    def _start_inspecting_board(self) -> None:
        board_name = self.env.all_objects[self.target_board_id].name
        duration = int(self.inspection_params["inspect_board_duration"])
        self._record_inspection(
            "inspecting_board",
            target_id=self.target_board_id,
            target_name=board_name,
            duration_steps=duration,
        )
        self.state = "inspecting_board"
        self.state_timer = 0

    def _start_hesitating(self, next_state: str = "moving_to_item") -> None:
        others = [
            oid for oid in self._remaining_item_candidates()
            if oid != self.current_item_id
        ]
        self.hesitation_alt_item_id = random.choice(others) if others else None
        duration = int(self.inspection_params["hesitate_duration"])
        alt_name = None
        if self.hesitation_alt_item_id is not None:
            alt_name = self.env.all_objects[self.hesitation_alt_item_id].name
        self._record_inspection(
            "hesitating",
            next_state=next_state,
            alt_item_id=self.hesitation_alt_item_id,
            alt_item_name=alt_name,
            duration_steps=duration,
        )
        self.state = "hesitating"
        self.state_timer = 0
        self._hesitation_next_state = next_state

    def _enter_post_selection_state(self) -> bool:
        """サブゴール選択後、確率的に inspection / hesitation を挟んでから接近へ。"""
        if self.current_item_id is None:
            self.state = "completed"
            self.is_finished = True
            return False

        if random.random() < float(self.inspection_params["hesitate_prob"]):
            self._start_hesitating(next_state="moving_to_item")
            return True
        if random.random() < float(self.inspection_params["inspect_item_prob"]):
            self._start_inspecting_item()
            return True

        self.state = "moving_to_item"
        self.state_timer = 0
        return True

    def _enter_moving_to_item(self) -> bool:
        """moving_to_item へ入るたびに foraging 的に次の item を選ぶ。"""
        remaining = self._remaining_item_candidates()
        if not remaining:
            self.current_item_id = None
            self.state = "completed"
            self.is_finished = True
            return False

        self.current_item_id, self.last_item_selection = self.select_next_item(
            remaining, self.household_style
        )
        if self.current_item_id is None:
            self.state = "completed"
            self.is_finished = True
            return False

        self.item_selection_history.append(dict(self.last_item_selection))
        return self._enter_post_selection_state()

    def _apply_task_selection(self, task: Dict[str, Any]) -> None:
        self.global_intent = task["task_type"]
        self.target_board_id = task["board_id"]
        self.candidate_items = list(task["item_ids"])
        self.items_placed = 0
        self.last_item_selection = {}
        self.item_selection_history = []
        self.inspection_history = []
        self.hesitation_alt_item_id = None

        board_obj = self.env.all_objects[self.target_board_id]
        label = TASK_SCHEMAS[self.global_intent]["label_ja"]
        print(f"[HouseholdSoloProtocol] global_intent={self.global_intent} ({label})")
        print(f"[HouseholdSoloProtocol] Selected board: {board_obj.name} (ID: {self.target_board_id})")
        print(f"[HouseholdSoloProtocol] Candidate items: {self.candidate_items}")

    def _get_task_candidates(self) -> List[Dict[str, Any]]:
        """Wrapper 越しに BaseEnv の household_task_candidates を取得する。"""
        env = self.env
        seen = set()
        while env is not None and id(env) not in seen:
            seen.add(id(env))
            if hasattr(env, "household_task_candidates"):
                return list(getattr(env, "household_task_candidates", []) or [])
            env = getattr(env, "env", None)
        return []

    def _select_task(self, context: Dict[str, Any]) -> bool:
        """TASK_SCHEMAS と BaseEnv の household_task_candidates からタスクを選ぶ。"""
        candidates = self._get_task_candidates()
        if not candidates:
            print("[HouseholdSoloProtocol] No feasible household task; skipping.")
            return False

        fixed_task = context.get("fixed_task")
        if fixed_task is not None:
            matched = [c for c in candidates if c["task_type"] == fixed_task]
            if not matched:
                print(
                    f"[HouseholdSoloProtocol] fixed_task={fixed_task!r} is not feasible; skipping."
                )
                return False
            task = matched[0]
        else:
            task = random.choice(candidates)

        self._apply_task_selection(task)
        return True

    def start(self, context: Dict[str, Any]):
        super().start(context)
        self.state = "init"
        self.state_timer = 0
        self.global_intent = None
        self.target_board_id = None
        self.candidate_items = []
        self.current_item_id = None
        self.items_placed = 0
        self.last_item_selection = {}
        self.item_selection_history = []
        self.inspection_history = []
        self.hesitation_alt_item_id = None
        # Phase 3: style は固定（Phase 6 で確率割当）
        self.household_style = context.get("household_style", "committed")
        self.inspection_params = dict(HOUSEHOLD_INSPECTION_PARAMS)
        for key in self.inspection_params:
            if key in context:
                self.inspection_params[key] = context[key]

        if self._select_task(context):
            if not self._enter_moving_to_item():
                return
            self.debug_info = {
                "global_intent": self.global_intent,
                "household_style": self.household_style,
                "target_board_id": self.target_board_id,
                "candidate_items": list(self.candidate_items),
                "current_item_id": self.current_item_id,
                "last_item_selection": self.last_item_selection,
                "inspection_history": list(self.inspection_history),
            }
        else:
            self.state = "completed"
            self.is_finished = True

    def _manage_state(self, infant_signals: Dict[str, Any]) -> None:
        self.state_timer += 1

        if self.state == "completed":
            self.is_finished = True
            return

        if self.state == "inspecting_item":
            if self.state_timer >= int(self.inspection_params["inspect_item_duration"]):
                self.state = "moving_to_item"
                self.state_timer = 0

        elif self.state == "hesitating":
            if self.state_timer >= int(self.inspection_params["hesitate_duration"]):
                self.state = self._hesitation_next_state
                self.state_timer = 0

        elif self.state == "inspecting_board":
            if self.state_timer >= int(self.inspection_params["inspect_board_duration"]):
                self.state = "moving_to_board"
                self.state_timer = 0

        elif self.state == "moving_to_item":
            if self.current_item_id is None:
                self.state = "completed"
                self.is_finished = True
                self.state_timer = 0
                return

            dist = self.env._dist_body_to(self.env.parent, self.env._obj_pos(self.current_item_id))
            if dist < 0.6:
                self.state = "grasping_item"
                self.state_timer = 0

        elif self.state == "grasping_item":
            if self.env.parent["held_obj"] == self.current_item_id:
                if hasattr(self.env, "first_hand_status"):
                    self.env.first_hand_status = {}
                if random.random() < float(self.inspection_params["inspect_board_prob"]):
                    self._start_inspecting_board()
                else:
                    self.state = "moving_to_board"
                    self.state_timer = 0
            elif self.state_timer > 100:
                # 失敗したら foraging 選択をやり直す
                self._enter_moving_to_item()

        elif self.state == "moving_to_board":
            dist = self.env._dist_body_to(self.env.parent, self.env._obj_pos(self.target_board_id))
            if dist < 0.45:
                self.state = "placing_on_board"
                self.state_timer = 0

        elif self.state == "placing_on_board":
            if self._is_item_on_board():
                self.state = "check_done"
                self.state_timer = 0
            elif self.state_timer > 200:
                # 手放し後に board 上へ乗らなかった場合は再接近
                if self.env.parent["held_obj"] is None:
                    self._enter_moving_to_item()
                else:
                    self.state = "moving_to_board"
                    self.state_timer = 0

        elif self.state == "check_done":
            # Phase 1: 1 個配置したら完了
            self.items_placed += 1
            self.state = "completed"
            self.is_finished = True
            self.state_timer = 0

    def _is_item_on_board(self) -> bool:
        if self.current_item_id is None or self.target_board_id is None:
            return False
        return self._is_item_done(self.current_item_id)

    def _gaze_inspection_action(self, action: np.ndarray, object_id: int) -> np.ndarray:
        """接近せず首だけで対象を見る。"""
        self.env.reset_arm_pose(self.env.parent, action=action)
        self.env.gaze_object(self.env.parent, object_id, yaw_link="head", action=action)
        return action

    def _hesitation_gaze_action(self, action: np.ndarray) -> np.ndarray:
        """躊躇中は選んだ item と別候補を交互に見る（接近しない）。"""
        self.env.reset_arm_pose(self.env.parent, action=action)
        if self.current_item_id is None:
            return action
        gaze_id = self.current_item_id
        if (
            self.hesitation_alt_item_id is not None
            and self.state_timer % 20 >= 10
        ):
            gaze_id = self.hesitation_alt_item_id
        self.env.gaze_object(self.env.parent, gaze_id, yaw_link="head", action=action)
        return action

    def _place_on_board_action(self, action: np.ndarray) -> np.ndarray:
        """board 手前まで接近してから手を開き、is_platform 経由で配置する。"""
        board_pos = self.env._obj_pos(self.target_board_id)
        dist = self.env._dist_body_to(self.env.parent, board_pos)
        if dist > 0.4:
            self.env.approach_object(
                self.env.parent, self.target_board_id, action, approach_dist=0.32
            )
            self.env.grip_object(self.env.parent, action)
            return action

        # board の中央上で手放し、is_platform の重なり判定に入りやすくする
        target_pos = (board_pos[0], board_pos[1], board_pos[2] + 0.18)
        self.env.hand_reach_location(self.env.parent, target_pos, action=action)
        am = self.env.action_map
        action[am["index_joint"]] = -3.14
        action[am["other_fingers"]] = -3.14
        return action

    def _determine_action(self) -> np.ndarray:
        action = np.zeros(self.action_dim, dtype=np.float32)
        self.env.reset_arm_pose(self.env.parent, action=action)
        if self.state == "init" or self.state == "completed":
            return action

        if self.state == "inspecting_item":
            if self.current_item_id is not None:
                self._gaze_inspection_action(action, self.current_item_id)

        elif self.state == "inspecting_board":
            if self.target_board_id is not None:
                self._gaze_inspection_action(action, self.target_board_id)

        elif self.state == "hesitating":
            self._hesitation_gaze_action(action)

        elif self.state == "moving_to_item":
            if self.current_item_id is not None:
                self.env.approach_object(self.env.parent, self.current_item_id, action, approach_dist=0.5)

        elif self.state == "grasping_item":
            if self.current_item_id is not None:
                self.env.grasp_object(self.env.parent, self.current_item_id, action, approach_dist=0.5)

        elif self.state == "moving_to_board":
            self.env.approach_object(self.env.parent, self.target_board_id, action, approach_dist=0.32)
            self.env.grip_object(self.env.parent, action)

        elif self.state == "placing_on_board":
            self._place_on_board_action(action)

        return action

    def keeps_held_object(self, held_id) -> bool:
        return self.current_item_id is not None and held_id == self.current_item_id

    def _calculate_persistence(self, infant_signals: Dict[str, Any]):
        self.persistence = 5.0  # 家事実行中は割り込みを強く抑制する
