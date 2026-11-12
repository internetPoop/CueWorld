"""バッチ E: 維持スコアの穴を形だけ固定する。2.7 と 5.0 の値そのものは固定しない。"""

import unittest

import gymnasium as gym
import numpy as np

from cueworld.protocol_manager import ProtocolManager
from cueworld.protocols.base import MAINTAIN_SCORE_EARLY, MAINTAIN_SCORE_LATE
from cueworld.protocols.dyadic import DyadicProtocol
from cueworld.protocols.household_solo import HouseholdSoloProtocol
from cueworld.protocols.joint_attention import JointAttentionProtocol
from cueworld.protocols.object_giving import ObjectGivingProtocol
from cueworld.protocols.observance import ObservanceProtocol
from cueworld.switch_eval import gaze_and_reach, gaze_object, pointing_at, reach_at, zeros


class _MockObject:
    def __init__(self, oid: int, concept_id: str, is_platform: bool = False):
        self.id = oid
        self.concept_id = concept_id
        self.is_platform = is_platform
        self.name = concept_id


class _MockEnv(gym.Env):
    action_num = 17
    eval_mode = False
    active_signals = None
    target_obj_id = None

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = (
            gym.spaces.Box(-1.0, 1.0, shape=(17,), dtype=np.float32),
            gym.spaces.Box(-1.0, 1.0, shape=(17,), dtype=np.float32),
        )
        self.step_count = 0
        self.parent = {"body": 1}
        self.infant = {"body": 2, "held_obj": None}
        self.all_objects = {
            101: _MockObject(101, "red_cube"),
            102: _MockObject(102, "green_cube"),
        }
        self.active_objects = [101, 102]
        self.trackable_targets = [101, 102, self.parent["body"]]
        self.reach_radius = 0.4
        self._in_reach = set()
        self.voice_volume_threshold = 0.25
        self.household_task_candidates = []
        self.stand_dist = 0.5
        self.place_dist = 0.4

    def object_in_agent_reach(self, agent, oid):
        return oid in self._in_reach

    def reset(self, **kwargs):
        self.step_count = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        return np.zeros(1, dtype=np.float32), [0.0, 0.0], False, False, {}


class TestMaintainScore(unittest.TestCase):
    def setUp(self):
        self.env = _MockEnv()
        self.ja = JointAttentionProtocol(self.env, cue_type="pointing")
        self.giving = ObjectGivingProtocol(self.env)
        self.dyadic = DyadicProtocol(self.env)
        self.observance = ObservanceProtocol(self.env)
        self.household = HouseholdSoloProtocol(self.env)
        self.manager = ProtocolManager(
            self.env,
            fixed_protocol="observance",
            fixed_context={},
            decay_rate=0.5,
        )

    def test_retry_uses_early_regardless_of_engaged(self):
        self.ja.start({"fixed_target": 101})
        self.ja.retry_count = 1
        self.ja._calculate_persistence(gaze_object(self.manager))
        self.assertEqual(self.ja.persistence, MAINTAIN_SCORE_EARLY)

    def test_not_engaged_uses_early(self):
        self.ja.start({"fixed_target": 101})
        self.ja.retry_count = 0
        self.env.step_count = self.ja.start_step + self.ja.typical_steps
        self.ja._calculate_persistence(reach_at(self.manager))
        self.assertEqual(self.ja.persistence, MAINTAIN_SCORE_EARLY)

    def test_engaged_other_object_uses_late(self):
        self.ja.start({"fixed_target": 101})
        self.ja.retry_count = 0
        signals = gaze_and_reach(self.manager, gaze_idx=0, reach_idx=1)
        self.ja._calculate_persistence(signals)
        self.assertEqual(self.ja.persistence, MAINTAIN_SCORE_LATE)
        self.assertLess(MAINTAIN_SCORE_EARLY, MAINTAIN_SCORE_LATE)

    def test_engaged_same_object_complete_form_uses_early(self):
        self.ja.start({"fixed_target": 101})
        self.ja.retry_count = 0
        signals = gaze_and_reach(self.manager, gaze_idx=0, reach_idx=0)
        self.ja._calculate_persistence(signals)
        self.assertEqual(self.ja.persistence, MAINTAIN_SCORE_EARLY)

    def test_giving_engaged_look_at_target(self):
        self.giving.start({"fixed_target": 101})
        self.giving.retry_count = 0
        self.giving._calculate_persistence(gaze_object(self.manager))
        self.assertEqual(self.giving.persistence, MAINTAIN_SCORE_LATE)
        self.giving._calculate_persistence(zeros(self.manager))
        self.assertEqual(self.giving.persistence, MAINTAIN_SCORE_EARLY)

    def test_dyadic_imitating_uses_late(self):
        self.dyadic.retry_count = 0
        self.dyadic._infant_has_responded = False
        self.dyadic.state = "performing_parent_cue"
        self.dyadic._calculate_persistence({})
        self.assertEqual(self.dyadic.persistence, MAINTAIN_SCORE_EARLY)
        self.dyadic.state = "imitating"
        self.dyadic._calculate_persistence({})
        self.assertEqual(self.dyadic.persistence, MAINTAIN_SCORE_LATE)
        self.dyadic.state = "echoing"
        self.dyadic._calculate_persistence({})
        self.assertEqual(self.dyadic.persistence, MAINTAIN_SCORE_LATE)

    def test_observance_stays_one(self):
        self.observance._calculate_persistence({})
        self.assertEqual(self.observance.persistence, 1.0)

    def test_household_stays_five(self):
        self.household._calculate_persistence({})
        self.assertEqual(self.household.persistence, 5.0)

    def test_household_maintain_beats_pointing(self):
        self.manager.switch_protocol("household_solo", {})
        self.manager.current_protocol._calculate_persistence({})
        choices, probs = self.manager._response_probs(pointing_at(self.manager))
        pmap = {name: float(p) for name, p in zip(choices, probs)}
        self.assertGreater(pmap["maintain"], pmap.get("joint_attention", 0.0))
        self.assertGreater(pmap["maintain"], 0.9)

    def test_retry_switch_prob_exceeds_engaged_other(self):
        self.manager.switch_protocol(
            "joint_attention", {"infant_initiative": False, "fixed_target": 101}
        )
        proto = self.manager.current_protocol
        reach = reach_at(self.manager)
        other = gaze_and_reach(self.manager, gaze_idx=0, reach_idx=1)

        proto.retry_count = 1
        proto._calculate_persistence(reach)
        _, probs_retry = self.manager._response_probs(reach)

        proto.retry_count = 0
        proto._calculate_persistence(other)
        _, probs_late = self.manager._response_probs(other)

        p_retry_switch = 1.0 - float(probs_retry[0])
        p_late_switch = 1.0 - float(probs_late[0])
        self.assertGreater(p_retry_switch, p_late_switch)

    def test_gaze_alone_does_not_raise_ja(self):
        base = self.manager._calculate_signal_strength(zeros(self.manager))
        ja_idx = self.manager.response_protocol.index("joint_attention")
        score = self.manager._calculate_signal_strength(gaze_object(self.manager))
        self.assertAlmostEqual(float(score[ja_idx]), float(base[ja_idx]), places=5)


if __name__ == "__main__":
    unittest.main()
