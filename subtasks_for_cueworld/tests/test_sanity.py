import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from envs.sanity import SanityEnv


def test_gymnasium_env_checker_passes():
    env = SanityEnv(max_steps=8, seed=0)
    check_env(env, skip_render_check=True)


def test_rejects_invalid_constructor_args():
    with pytest.raises(ValueError):
        SanityEnv(max_steps=8, n_actions=2)
    with pytest.raises(ValueError):
        SanityEnv(max_steps=8, n_symbols=1)
    with pytest.raises(ValueError):
        SanityEnv(max_steps=8, symbol_noise=-0.1)
    with pytest.raises(ValueError):
        SanityEnv(max_steps=8, symbol_noise=1.1)
    with pytest.raises(ValueError):
        SanityEnv(max_steps=8, obs_noise=-0.01)


def test_roles_stable_across_reset_and_reset_obs_is_zero():
    env = SanityEnv(max_steps=4, seed=7)
    det = env.deterministic_action
    rew = env.reward_action
    sym = env.deterministic_symbol

    obs, info = env.reset(seed=1)
    assert obs.shape == (4,)
    assert np.allclose(obs, 0.0)
    assert info["symbol"] is None
    assert env.deterministic_action == det
    assert env.reward_action == rew
    assert env.deterministic_symbol == sym

    env.step(det)
    obs2, info2 = env.reset(seed=99)
    assert np.allclose(obs2, 0.0)
    assert info2["symbol"] is None
    assert env.deterministic_action == det
    assert env.reward_action == rew
    assert env.deterministic_symbol == sym


def test_rejects_bad_yoke_shape():
    env = SanityEnv(max_steps=3, n_symbols=4, seed=0)
    with pytest.raises(ValueError):
        env.reset(options={"yoke_observations": np.zeros((2, 4), dtype=np.float32)})
    with pytest.raises(ValueError):
        env.reset(options={"yoke_observations": np.zeros((3, 3), dtype=np.float32)})


def test_deterministic_action_fixed_onehot_reward_off():
    env = SanityEnv(
        max_steps=20,
        n_actions=3,
        n_symbols=4,
        reward_mode=False,
        symbol_noise=0.0,
        obs_noise=0.0,
        seed=3,
    )
    env.reset(seed=0)
    det = env.deterministic_action
    expected = np.zeros(4, dtype=np.float32)
    expected[env.deterministic_symbol] = 1.0

    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(det)
        assert reward == 0.0
        assert terminated is False
        assert np.allclose(obs, expected)
        assert info["symbol"] == env.deterministic_symbol
        assert info["yoke"] is False


def test_other_actions_are_not_constant_and_reward_zero():
    env = SanityEnv(
        max_steps=200,
        n_actions=3,
        n_symbols=4,
        reward_mode=False,
        symbol_noise=0.0,
        obs_noise=0.0,
        seed=5,
    )
    env.reset(seed=0)
    det = env.deterministic_action
    other = next(a for a in range(env.n_actions) if a != det)

    symbols = []
    for _ in range(40):
        obs, reward, _, _, info = env.step(other)
        assert reward == 0.0
        assert info["symbol"] is not None
        symbols.append(info["symbol"])
        assert abs(obs.sum() - 1.0) < 1e-5
        assert np.count_nonzero(np.isclose(obs, 1.0)) == 1

    assert len(set(symbols)) > 1


def test_reward_mode_only_reward_action_gets_one():
    env = SanityEnv(
        max_steps=60,
        n_actions=4,
        n_symbols=4,
        reward_mode=True,
        symbol_noise=0.0,
        obs_noise=0.0,
        seed=11,
    )
    env.reset(seed=0)
    det = env.deterministic_action
    rew = env.reward_action
    assert det != rew

    for action in range(env.n_actions):
        obs, reward, _, _, info = env.step(action)
        if action == rew:
            assert reward == 1.0
        else:
            assert reward == 0.0
        assert info["symbol"] is not None

    # reward action symbols are not constant
    symbols = []
    for _ in range(30):
        _, _, _, _, info = env.step(rew)
        symbols.append(info["symbol"])
    assert len(set(symbols)) > 1


def test_yoke_replays_recorded_vectors_and_reward_follows_current_action():
    env = SanityEnv(
        max_steps=5,
        n_actions=3,
        n_symbols=4,
        reward_mode=True,
        symbol_noise=0.0,
        obs_noise=0.0,
        seed=13,
    )
    env.reset(seed=0)
    det = env.deterministic_action
    rew = env.reward_action
    other = next(a for a in range(env.n_actions) if a not in (det, rew))

    recorded = []
    for _ in range(env.max_steps):
        obs, _, _, truncated, _ = env.step(det)
        recorded.append(obs.copy())
        if truncated:
            break
    tape = np.stack(recorded, axis=0)
    assert tape.shape == (env.max_steps, env.n_symbols)

    obs0, info0 = env.reset(options={"yoke_observations": tape})
    assert np.allclose(obs0, 0.0)
    assert info0["yoke"] is True
    assert info0["symbol"] is None

    # press different actions; vectors must match the tape
    actions = [other, rew, other, det, rew]
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.allclose(obs, tape[i])
        assert info["yoke"] is True
        assert info["symbol"] is None
        assert terminated is False
        if action == rew:
            assert reward == 1.0
        else:
            assert reward == 0.0
        if i + 1 == env.max_steps:
            assert truncated is True
        else:
            assert truncated is False


def test_yoke_replays_obs_noise_vectors_exactly():
    env = SanityEnv(
        max_steps=4,
        n_actions=3,
        n_symbols=4,
        reward_mode=False,
        symbol_noise=0.0,
        obs_noise=0.25,
        seed=17,
    )
    env.reset(seed=0)
    det = env.deterministic_action

    recorded = []
    for _ in range(env.max_steps):
        obs, _, _, _, _ = env.step(det)
        recorded.append(obs.copy())
    tape = np.stack(recorded, axis=0)

    # noisy vectors should not be exact one-hots
    assert not np.allclose(tape, np.round(tape))

    other = next(a for a in range(env.n_actions) if a != det)
    env.reset(options={"yoke_observations": tape})
    for i in range(env.max_steps):
        obs, reward, _, _, info = env.step(other)
        assert np.array_equal(obs, tape[i])
        assert reward == 0.0
        assert info["symbol"] is None


def test_step_after_truncation_raises():
    env = SanityEnv(max_steps=2, seed=0)
    env.reset(seed=0)
    env.step(0)
    _, _, _, truncated, _ = env.step(0)
    assert truncated is True
    with pytest.raises(RuntimeError):
        env.step(0)
