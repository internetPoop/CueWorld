"""T2 用: プロトコル状態の滞在 step を記録する薄いヘルパ。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class DwellTracker:
    """状態遷移ごとに退出時の滞在 step を記録する。"""

    def __init__(self) -> None:
        self.dwells: Dict[str, List[int]] = {}
        self.extras: Dict[str, int] = {}
        self._current: Optional[str] = None
        self._steps: int = 0
        self._shake_reach_logged: bool = False
        self._gesture_reach_logged: bool = False

    def update(self, state: Optional[str]) -> None:
        if state is None:
            return
        if state != self._current:
            if self._current is not None:
                self.dwells.setdefault(self._current, []).append(self._steps)
            self._current = state
            self._steps = 1  # 進入したこの step を含める（state_timer と整合）
            if state == "shaking":
                self._shake_reach_logged = False
        else:
            self._steps += 1

    def observe_protocol(self, protocol: Any) -> None:
        """前進系の到達待ち（shake / gesture）を別キーで記録。"""
        if protocol is None:
            return
        state = getattr(protocol, "state", None)
        if (
            state == "shaking"
            and not self._shake_reach_logged
            and bool(getattr(protocol, "shake_reached", False))
        ):
            # shaking 進入からの到達待ち step（到達した瞬間の state_timer）
            self.extras["shake_reach"] = int(getattr(protocol, "state_timer", self._steps))
            self._shake_reach_logged = True
        if (
            not self._gesture_reach_logged
            and bool(getattr(protocol, "gesture_reached", False))
        ):
            if getattr(protocol, "gesture_pos", None) is None:
                self.extras["gesture_reach"] = 0
            else:
                self.extras["gesture_reach"] = int(
                    getattr(protocol, "gesture_wait_timer", 0)
                )
            self._gesture_reach_logged = True

    def finalize(self) -> None:
        if self._current is not None:
            self.dwells.setdefault(self._current, []).append(self._steps)
            self._current = None

    def last_dwell(self, state: str) -> Optional[int]:
        vals = self.dwells.get(state)
        if not vals:
            return None
        return vals[-1]

    def print_summary(self, *, seed: int, success: int, run: str) -> None:
        self.finalize()
        parts = [f"DWELL seed={seed} run={run} success={success}"]
        for state in sorted(self.dwells.keys()):
            # 再進入時は最後の滞在を代表値に（失敗系 escalate 前の本待ちなど）
            parts.append(f"{state}={self.dwells[state][-1]}")
        for key in sorted(self.extras.keys()):
            parts.append(f"{key}={self.extras[key]}")
        print(" ".join(parts))
        print("===== Dwell Summary =====")
        print(f"  seed={seed} run={run} success={success}")
        for state in sorted(self.dwells.keys()):
            print(f"  dwell[{state}] = {self.dwells[state]}")
        for key in sorted(self.extras.keys()):
            print(f"  extra[{key}] = {self.extras[key]}")
        print("=========================")
