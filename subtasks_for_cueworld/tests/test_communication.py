import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from envs.communication import CommunicationEnv


def test_gymnasium_env_checker_passes():
    env = CommunicationEnv(max_steps=8, seed=0)
    check_env(env, skip_render_check=True)


def test_rejects_invalid_constructor_args():
    with pytest.raises(ValueError):
        CommunicationEnv(max_steps=0)
    with pytest.raises(ValueError):
        CommunicationEnv(max_steps=8, n_meaningful=0)
    with pytest.raises(ValueError):
        CommunicationEnv(max_steps=8, n_meaningful=9)
    with pytest.raises(ValueError):
        CommunicationEnv(max_steps=8, obs_noise=-0.01)


def test_reset_obs_and_info():
    env = CommunicationEnv(max_steps=4, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (6,)
    assert np.allclose(obs, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert info["infant_action"] == 0
    assert info["parent_action"] == 0
    assert info["infant_energy"] == 4
    assert info["parent_energy"] == 4
    assert info["trigger_id"] is None
    assert info["reply_step"] == 0
    assert info["yoke"] is False
    assert info["trigger_count"] == 0
    assert info["reply_heard_count"] == 0


def test_noop_recovers_energy():
    env = CommunicationEnv(max_steps=10, seed=0)
    env.reset(seed=0)
    for _ in range(4):
        obs, reward, _, _, info = env.step(0)
        assert reward == 0.0
        assert info["infant_energy"] == 4
        assert abs(obs[0] - 1.0) < 1e-6


def test_signal_drains_energy():
    env = CommunicationEnv(max_steps=20, seed=0)
    env.reset(seed=0)

    for i in range(4):
        obs, _, _, _, info = env.step(1)  # symbol 0
        assert info["infant_action"] == 1
        assert info["infant_energy"] == 3 - i
        assert abs(obs[0] - info["infant_energy"] / 4.0) < 1e-6

    assert info["infant_energy"] == 0
    obs, _, _, _, info = env.step(1)
    assert info["infant_action"] == 0  # forced noop
    assert info["infant_energy"] == 1
    assert abs(obs[0] - 0.25) < 1e-6
    # Infant silent: observe parent random signal (who=-1) or mutual silence (who=0)
    if info["parent_action"] == 0:
        assert abs(obs[1] - 0.0) < 1e-6
    else:
        assert abs(obs[1] - (-1.0)) < 1e-6


def test_self_signal_in_obs():
    env = CommunicationEnv(max_steps=8, seed=0)
    env.reset(seed=0)
    for symbol in range(4):
        action = symbol + 1
        obs, _, _, _, info = env.step(action)
        assert info["infant_action"] == action
        assert abs(obs[1] - 1.0) < 1e-6
        expected = np.zeros(4, dtype=np.float32)
        expected[symbol] = 1.0
        assert np.allclose(obs[2:], expected)


def test_parent_random_when_infant_silent():
    env = CommunicationEnv(max_steps=200, seed=0)
    env.reset(seed=0)
    symbols = []
    for _ in range(200):
        obs, reward, _, _, info = env.step(0)
        assert reward == 0.0
        if info["parent_action"] == 0:
            assert abs(obs[1] - 0.0) < 1e-6
            assert np.allclose(obs[2:], 0.0)
        else:
            assert abs(obs[1] - (-1.0)) < 1e-6
            symbol = info["parent_action"] - 1
            symbols.append(symbol)
            expected = np.zeros(4, dtype=np.float32)
            expected[symbol] = 1.0
            assert np.allclose(obs[2:], expected)
    assert len(set(symbols)) > 1


def test_trigger_fires_reply():
    env = CommunicationEnv(max_steps=20, seed=0)
    env.reset(seed=0)

    # Trigger (0, 0): actions 1, 1
    obs, _, _, _, info = env.step(1)
    assert info["trigger_id"] is None
    assert info["trigger_count"] == 0
    assert abs(obs[1] - 1.0) < 1e-6

    obs, _, _, _, info = env.step(1)
    assert info["trigger_id"] == 0
    assert info["trigger_count"] == 1
    assert abs(obs[1] - 1.0) < 1e-6

    # Reply (2, 0) over next 2 silent steps
    obs, _, _, _, info = env.step(0)
    assert info["reply_step"] == 1
    assert info["parent_action"] == 3  # symbol 2
    assert abs(obs[1] - (-1.0)) < 1e-6
    expected = np.zeros(4, dtype=np.float32)
    expected[2] = 1.0
    assert np.allclose(obs[2:], expected)

    obs, _, _, _, info = env.step(0)
    assert info["reply_step"] == 2
    assert info["parent_action"] == 1  # symbol 0
    assert info["reply_heard_count"] == 1
    expected = np.zeros(4, dtype=np.float32)
    expected[0] = 1.0
    assert np.allclose(obs[2:], expected)


def test_invalid_pair_no_reply():
    env = CommunicationEnv(max_steps=20, seed=0, n_meaningful=1)
    env.reset(seed=0)

    # (0, 1) is not meaningful when n_meaningful=1
    env.step(1)  # symbol 0
    obs, _, _, _, info = env.step(2)  # symbol 1
    assert info["trigger_id"] is None
    assert info["trigger_count"] == 0

    # Next silent steps should be random parent, not fixed reply (2, 0)
    replies = []
    for _ in range(2):
        obs, _, _, _, info = env.step(0)
        assert info["reply_step"] == 0
        replies.append(info["parent_action"])
    # Fixed reply would be (3, 1) i.e. actions 3 then 1. Random may match by chance,
    # so check over more steps that no trigger_count / reply_step appears.
    assert info["trigger_count"] == 0
    for _ in range(10):
        _, _, _, _, info = env.step(0)
        assert info["reply_step"] == 0
        assert info["trigger_count"] == 0


def test_sliding_window():
    env = CommunicationEnv(max_steps=20, seed=0)
    env.reset(seed=0)

    # (2, 0, 0): first pair (2,0) invalid, then window (0,0) valid
    env.step(3)  # symbol 2
    obs, _, _, _, info = env.step(1)  # symbol 0 → pair (2,0) invalid, buffer=[0]
    assert info["trigger_id"] is None
    assert info["trigger_count"] == 0

    obs, _, _, _, info = env.step(1)  # symbol 0 → pair (0,0) valid
    assert info["trigger_id"] == 0
    assert info["trigger_count"] == 1

    obs, _, _, _, info = env.step(0)
    assert info["reply_step"] == 1
    assert info["parent_action"] == 3  # symbol 2


def test_noop_breaks_sequence():
    env = CommunicationEnv(max_steps=20, seed=0)
    env.reset(seed=0)

    env.step(1)  # symbol 0
    env.step(0)  # noop clears buffer
    obs, _, _, _, info = env.step(1)  # symbol 0 alone
    assert info["trigger_id"] is None
    assert info["trigger_count"] == 0

    for _ in range(2):
        _, _, _, _, info = env.step(0)
        assert info["reply_step"] == 0


def test_reply_clears_buffer():
    env = CommunicationEnv(max_steps=40, seed=0)
    env.reset(seed=0)

    # Fire trigger (0,0)
    env.step(1)
    env.step(1)

    # During reply, emit signals — these must not start a new trigger
    obs, _, _, _, info = env.step(1)  # talk during reply step 1
    assert info["reply_step"] == 1
    obs, _, _, _, info = env.step(1)  # talk during reply step 2
    assert info["reply_step"] == 2
    assert info["trigger_count"] == 1

    # Recover energy after 4 signals
    for _ in range(4):
        env.step(0)

    # After reply, one signal alone must not complete a pair from reply-time signals
    obs, _, _, _, info = env.step(1)
    assert info["trigger_id"] is None
    assert info["trigger_count"] == 1

    # Need two fresh signals for next trigger
    obs, _, _, _, info = env.step(1)
    assert info["trigger_id"] == 0
    assert info["trigger_count"] == 2


def test_reply_heard_count():
    env = CommunicationEnv(max_steps=20, seed=0)
    env.reset(seed=0)
    env.step(1)
    env.step(1)
    env.step(0)
    _, _, _, _, info = env.step(0)
    assert info["reply_heard_count"] == 1


def test_reply_heard_count_not_incremented_when_talking():
    env = CommunicationEnv(max_steps=40, seed=0)
    env.reset(seed=0)
    env.step(1)
    env.step(1)
    # Talk during both reply steps
    _, _, _, _, info = env.step(2)
    assert info["reply_step"] == 1
    _, _, _, _, info = env.step(2)
    assert info["reply_step"] == 2
    assert info["reply_heard_count"] == 0

    # Recover energy
    for _ in range(4):
        env.step(0)

    # Talk only on first reply step
    env.step(1)
    env.step(1)
    _, _, _, _, info = env.step(2)
    assert info["reply_step"] == 1
    _, _, _, _, info = env.step(0)
    assert info["reply_step"] == 2
    assert info["reply_heard_count"] == 0

    # Recover energy
    for _ in range(4):
        env.step(0)

    # Talk only on second reply step
    env.step(1)
    env.step(1)
    _, _, _, _, info = env.step(0)
    assert info["reply_step"] == 1
    _, _, _, _, info = env.step(2)
    assert info["reply_step"] == 2
    assert info["reply_heard_count"] == 0


def test_n_meaningful_controls_active_pairs():
    env = CommunicationEnv(max_steps=20, seed=0, n_meaningful=2)
    env.reset(seed=0)

    # Second pair (0, 1) → reply (2, 1)
    env.step(1)  # symbol 0
    _, _, _, _, info = env.step(2)  # symbol 1
    assert info["trigger_id"] == 1
    assert info["trigger_count"] == 1

    _, _, _, _, info = env.step(0)
    assert info["reply_step"] == 1
    assert info["parent_action"] == 3  # symbol 2
    _, _, _, _, info = env.step(0)
    assert info["reply_step"] == 2
    assert info["parent_action"] == 2  # symbol 1


def test_obs_noise_on_signal_only():
    env = CommunicationEnv(max_steps=10, obs_noise=0.25, seed=0)
    env.reset(seed=0)

    obs, _, _, _, info = env.step(1)
    assert abs(obs[0] - info["infant_energy"] / 4.0) < 1e-6
    assert abs(obs[1] - 1.0) < 1e-6
    # Signal part should not be exact one-hot
    assert not np.allclose(obs[2:], [1.0, 0.0, 0.0, 0.0])

    env2 = CommunicationEnv(max_steps=10, obs_noise=0.25, seed=0)
    env2.reset(seed=0)
    # Silent step: who exact, signal part noisy zeros
    obs, _, _, _, info = env2.step(0)
    assert abs(obs[0] - info["infant_energy"] / 4.0) < 1e-6
    if info["parent_action"] == 0:
        assert abs(obs[1] - 0.0) < 1e-6
    else:
        assert abs(obs[1] - (-1.0)) < 1e-6
    # who and energy clean; noise only on last 4
    assert not np.allclose(obs[2:], np.round(obs[2:]))


def test_yoke_replays_parent_actions():
    env = CommunicationEnv(max_steps=6, seed=0)
    env.reset(seed=0)

    # Record parent actions under a fixed infant policy
    recorded = []
    infant_actions = [1, 1, 0, 0, 0, 0]
    for a in infant_actions:
        _, _, _, truncated, info = env.step(a)
        recorded.append(info["parent_action"])
        if truncated:
            break
    tape = np.asarray(recorded, dtype=np.int64)
    assert tape.shape == (6,)
    # Natural run should have triggered a reply
    assert tape[2] == 3 and tape[3] == 1

    # Yoke with different infant actions: parent tape is independent
    obs0, info0 = env.reset(options={"yoke_parent_actions": tape})
    assert np.allclose(obs0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert info0["yoke"] is True

    yoke_infant = [0, 0, 0, 0, 0, 0]
    for i, a in enumerate(yoke_infant):
        obs, reward, terminated, truncated, info = env.step(a)
        assert reward == 0.0
        assert terminated is False
        assert info["yoke"] is True
        assert info["parent_action"] == int(tape[i])
        assert info["trigger_id"] is None
        assert info["trigger_count"] == 0
        assert info["reply_step"] == 0
        # Infant silent → observe taped parent signal (or silence)
        if tape[i] == 0:
            assert abs(obs[1] - 0.0) < 1e-6
        else:
            assert abs(obs[1] - (-1.0)) < 1e-6
            expected = np.zeros(4, dtype=np.float32)
            expected[int(tape[i]) - 1] = 1.0
            assert np.allclose(obs[2:], expected)
        if i + 1 == env.max_steps:
            assert truncated is True
        else:
            assert truncated is False


def test_yoke_shape_validation():
    env = CommunicationEnv(max_steps=3, seed=0)
    with pytest.raises(ValueError):
        env.reset(options={"yoke_parent_actions": np.zeros(2, dtype=np.int64)})
    with pytest.raises(ValueError):
        env.reset(options={"yoke_parent_actions": np.zeros((3, 1), dtype=np.int64)})
    with pytest.raises(ValueError):
        env.reset(options={"yoke_parent_actions": np.array([0, 1, 5], dtype=np.int64)})


def test_reward_always_zero():
    env = CommunicationEnv(max_steps=20, seed=0)
    env.reset(seed=0)
    for action in range(5):
        _, reward, _, _, _ = env.step(action)
        assert reward == 0.0
    # Including after trigger/reply
    env.step(1)
    env.step(1)
    for _ in range(2):
        _, reward, _, _, _ = env.step(0)
        assert reward == 0.0


def test_truncation_at_max_steps():
    env = CommunicationEnv(max_steps=2, seed=0)
    env.reset(seed=0)
    env.step(0)
    _, _, terminated, truncated, _ = env.step(0)
    assert terminated is False
    assert truncated is True
    with pytest.raises(RuntimeError):
        env.step(0)


def test_parent_energy_prevents_signal():
    env = CommunicationEnv(max_steps=40, seed=0)
    env.reset(seed=0)

    # Infant silent: parent emits until energy hits 0 on an emit step.
    hit_zero_after_emit = False
    for _ in range(20):
        _, _, _, _, info = env.step(0)
        if info["parent_action"] != 0 and info["parent_energy"] == 0:
            hit_zero_after_emit = True
            break
    assert hit_zero_after_emit

    # Next silent step: energy was 0, so parent must noop and recover to 1
    obs, _, _, _, info = env.step(0)
    assert info["parent_action"] == 0
    assert info["parent_energy"] == 1
    assert abs(obs[1] - 0.0) < 1e-6
