"""見せる合図の掲げ点が目からの既定距離で、共同注意の注視が成功すること。"""
import sys
import unittest
from collections import deque
from pathlib import Path

import numpy as np
import pybullet as p

_DEMO = Path(__file__).resolve().parents[1] / "demo"
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))

from cueworld.base import BaseEnv
from cueworld.infant_signal import InfantSignalDetector
from cueworld.parent_tools import ParentTools, show_distance_from_eyes
from cueworld.protocol_manager import ProtocolManager
from demo_catalog_common import (
    HOUSEHOLD_OFF,
    SLIDE_ARENA,
    SpeechGate,
    find_object,
    ja_engage_action,
    slide_object_placements,
    step_parent,
)


class TestShowDistanceFromEyes(unittest.TestCase):
    def test_red_cube_half_extent(self):
        class _Cube:
            bbox_half_extents = [0.06, 0.06, 0.06]

        self.assertAlmostEqual(show_distance_from_eyes(_Cube()), 0.22)
        self.assertAlmostEqual(show_distance_from_eyes(None), 0.22)


class TestJaShowSuccess(unittest.TestCase):
    def test_show_cue_reaches_shared_attention(self):
        placements = slide_object_placements()
        base = BaseEnv(
            gui=False,
            num_agents=2,
            num_objects=len(placements),
            object_counts={},
            max_steps=400,
            return_views=False,
            action_repeat=1,
            arena_half_extents=SLIDE_ARENA,
            arena_rows=3,
            arena_cols=4,
            fixed_placements=placements,
            debug_info=False,
            placement_method="grid",
            use_cinematic_model=True,
        )
        env = ProtocolManager(
            InfantSignalDetector(ParentTools(base)),
            household_interval_steps=HOUSEHOLD_OFF,
            rejection=False,
        )
        try:
            _, info = env.reset(seed=42)
            red = find_object(env, "RedCube")
            env.set_fixed_protocol(
                "joint_attention",
                {
                    "fixed_target": red,
                    "cue_type": "shake",
                    "intent_type": "informative",
                    "infant_initiative": False,
                },
            )
            gate = SpeechGate()
            speech_queue = deque()
            speech_started = [False]
            saw_shaking = False
            final = None
            for _t in range(280):
                action = ja_engage_action(
                    env, red, speech_queue, speech_started, gate.ready
                )
                _, _, _, _, info, _ = step_parent(env, info, action)
                protocol = env.current_protocol
                state = protocol.state
                if state == "shaking" and not saw_shaking:
                    saw_shaking = True
                    obj = env.all_objects[red]
                    expected = show_distance_from_eyes(obj)
                    eye = np.asarray(
                        p.getLinkState(
                            env.infant["body"],
                            env.infant["eyes_link"],
                            computeForwardKinematics=True,
                            physicsClientId=env.client_id,
                        )[4],
                        dtype=float,
                    )
                    show = np.asarray(protocol.show_pos, dtype=float)
                    self.assertAlmostEqual(float(np.linalg.norm(show - eye)), expected, delta=0.03)
                    again = np.asarray(
                        env.compute_show_position(
                            env.parent,
                            env.infant,
                            show_dist=None,
                            target_object=obj,
                        ),
                        dtype=float,
                    )
                    explicit = np.asarray(
                        env.compute_show_position(
                            env.parent,
                            env.infant,
                            show_dist=expected,
                            target_object=obj,
                        ),
                        dtype=float,
                    )
                    np.testing.assert_allclose(again, explicit, atol=1e-5)
                if state in ("post_shared_attention", "completed", "failure"):
                    final = state
                    if state == "completed" or state == "failure":
                        break
            self.assertTrue(saw_shaking)
            self.assertTrue(env.current_protocol.ja_success)
            self.assertIn(final, ("post_shared_attention", "completed"))
            self.assertNotEqual(final, "failure")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
