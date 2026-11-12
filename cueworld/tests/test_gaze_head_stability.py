"""近接物体を頭で見るときの pitch 振動を抑える。"""

import math
import unittest

import numpy as np
import pybullet as p

from cueworld.base import BaseEnv
from cueworld.parent_tools import ParentTools


def _normalize_infant_action(env, infant_action: np.ndarray) -> np.ndarray:
    am = env.action_map
    for name, step_size in env.infant_joint_mappings.items():
        if name in ("volume", "pitch"):
            continue
        if name in am:
            infant_action[am[name]] = infant_action[am[name]] / (
                step_size * getattr(env, "action_repeat", 1)
            )
    return np.clip(infant_action, -1.0, 1.0)


def _make_env():
    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=1,
        object_counts={"RedCube": 1},
        max_steps=500,
        action_repeat=1,
        arena_half_extents=(1.6, 1.2),
        debug_info=False,
        placement_method="grid",
    )
    env = ParentTools(base)
    env.reset(seed=42)
    return base, env


def _place_in_front(base, oid, dist, z, lateral=0.05):
    infant_pos, _ = p.getBasePositionAndOrientation(
        base.infant["body"], physicsClientId=base.client_id
    )
    _, torso_fwd = base._agent_torso_pose(base.infant)
    fwd_xy = np.array([torso_fwd[0], torso_fwd[1]], dtype=np.float32)
    fwd_xy = fwd_xy / (np.linalg.norm(fwd_xy) + 1e-8)
    obj_pos = [
        float(infant_pos[0]) + float(fwd_xy[0]) * dist + lateral * float(-fwd_xy[1]),
        float(infant_pos[1]) + float(fwd_xy[1]) * dist + lateral * float(fwd_xy[0]),
        float(z),
    ]
    p.resetBasePositionAndOrientation(
        oid, obj_pos, [0, 0, 0, 1], physicsClientId=base.client_id
    )


class TestStabilizeGazeDeltas(unittest.TestCase):
    def setUp(self):
        self.tools = ParentTools.__new__(ParentTools)

    def test_deadzone_zeros_sub_degree_pitch(self):
        yaw, pitch = self.tools._stabilize_gaze_deltas(0.0, math.radians(0.4))
        self.assertEqual(pitch, 0.0)
        self.assertEqual(yaw, 0.0)
        _, pitch_edge = self.tools._stabilize_gaze_deltas(0.0, math.radians(1.2))
        self.assertEqual(pitch_edge, 0.0)

    def test_near_target_pitch_is_damped(self):
        raw = math.radians(10.0)
        _, pitch = self.tools._stabilize_gaze_deltas(0.0, raw)
        self.assertAlmostEqual(pitch, raw * ParentTools._GAZE_PITCH_NEAR_GAIN, places=5)

    def test_large_pitch_keeps_full_gain(self):
        raw = math.radians(40.0)
        _, pitch = self.tools._stabilize_gaze_deltas(0.0, raw)
        self.assertAlmostEqual(pitch, raw, places=5)


class TestCloseObjectHeadGaze(unittest.TestCase):
    def setUp(self):
        self.base, self.env = _make_env()

    def tearDown(self):
        self.env.close()

    def test_close_look_down_does_not_oscillate_tilt(self):
        """くれと同様、手元の高さの物を頭で見ても tilt が符号反転し続けない。"""
        oid = next(iter(self.base.all_objects.keys()))
        _place_in_front(
            self.base, oid, dist=float(self.base.place_dist), z=0.35, lateral=0.05
        )
        zero_p = np.zeros(self.base.action_space[1].shape, dtype=np.float32)
        pitch_cmds = []
        tilts = []
        for _ in range(40):
            infant_action = np.zeros(self.base.action_space[0].shape, dtype=np.float32)
            _, pitch = self.env.gaze_object(
                self.base.infant, oid, yaw_link="head", action=infant_action
            )
            pitch_cmds.append(pitch)
            tilts.append(self.base.get_joint_angle(self.base.infant, "head_tilt_joint"))
            infant_action = _normalize_infant_action(self.base, infant_action)
            self.base.step((infant_action, zero_p))

        late_cmds = pitch_cmds[20:]
        flips = 0
        for a, b in zip(late_cmds, late_cmds[1:]):
            if a * b < 0 and abs(a) > 1e-4 and abs(b) > 1e-4:
                flips += 1
        tilt_std_deg = math.degrees(float(np.std(tilts[20:])))
        self.assertEqual(flips, 0)
        self.assertLess(tilt_std_deg, 1.5)
