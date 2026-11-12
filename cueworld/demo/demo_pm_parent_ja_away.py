"""見守りから、幼児が物を見ていない状態で親が共同注意を始めるデモ。"""
from __future__ import annotations

from collections import deque

import numpy as np

from demo_catalog_common import (
    DemoRecorder,
    SpeechGate,
    find_object,
    find_signal_detector,
    ja_engage_action,
    ja_result,
    look_parent_action,
    make_env,
    neutralize_appeal,
    slide_object_placements,
    step_parent,
    word_match_label,
    write_result_json,
    TAIL_AFTER_DONE,
)


STEM = "demo_pm_parent_ja_away"
TITLE = "親始動・物を見ていない"
SITUATION_JA = (
    "親は見守りから始める。幼児は親の方を見ていて、立方体は見ていない。"
    "指差し・リーチ・単語・二項の訴えは出さない。"
    "物を見る・触る・持つ手がかりも立てない。"
    "訴えくじは止め、親始動の候補は共同注意だけにする。"
    "未試行だと共同注意は ZPD で候補外なので、デモ側で発達記録だけ開けて選ばせる。"
    "手がかりが無いと始まりにくいので、待つスコアを下げて動画のあいだに親が始めるようにする。"
    "幼児は物を見ていないので対象は視線から決まらない。デモでは赤い立方体に固定する。"
    "親が指差しを始めたら幼児はそこを見て、言い終わってから対象語を言う。"
)
SITUATION_EN = (
    "The parent starts in observance. The infant looks at the parent and not at a cube, "
    "and does not produce pointing, reach, word, or dyadic appeals. "
    "Looking, touching, and holding cues are off. "
    "The response lottery is off and the only initiative candidate is joint attention. "
    "Level records are opened so joint attention is inside the ZPD. "
    "The wait score is lowered so the parent still starts during the clip without a cue. "
    "The target is fixed to the red cube because the infant is not looking at an object. "
    "Once the parent points, the infant looks at it and names it after the parent finishes speaking."
)

# 手がかり 0、tau 0.2 だと、待つスコア 0.36 で開始確率は判定 1 回あたり約 0.14。
WAIT_WITHOUT_CUE = 0.36


def install_away_signals(manager, runtime: dict) -> None:
    detector = manager.env
    orig = detector.get_normalized_signals

    def wrapped(ignored_targets=None):
        sigs = orig(ignored_targets=ignored_targets)
        mode = runtime.get("mode") or "away"
        if mode == "idle":
            return neutralize_appeal(sigs, manager)
        n = len(getattr(manager, "trackable_targets", []) or [])
        out = neutralize_appeal(sigs, manager)
        out["gaze_targets"] = np.zeros(n, dtype=float)
        out["gaze_target_stability"] = 0.0
        out["touching"] = 0.0
        out["holding"] = 0.0
        out["looking_at_parent_eye"] = 0.0
        return out

    detector.get_normalized_signals = wrapped


def pin_red_target(protocol, red) -> None:
    orig = protocol.start

    def start(context):
        context = dict(context)
        context["fixed_target"] = red
        return orig(context)

    protocol.start = start


def main():
    print(f"Starting {STEM}...")
    env = make_env(
        slide_object_placements(),
        max_steps=1400,
        seed=42,
        wait_base_score=WAIT_WITHOUT_CUE,
    )
    env.request_protocol = [n for n in env.request_protocol if n == "joint_attention"]
    n_levels = len(env.level_try_counts)
    env.level_try_counts[:] = [env.base_try_count] * n_levels
    env.level_scores[:] = [0.5] * n_levels
    env.check_for_response = lambda signals: (False, "", {})
    obs, info = env.reset(seed=42)
    red = find_object(env, "RedCube")
    print(f"Red cube: {red}")
    pin_red_target(env.protocols["joint_attention"], red)
    runtime = {"mode": "away"}
    install_away_signals(env, runtime)

    rec = DemoRecorder(env)
    rec.append_voices(obs)
    rec.capture(info)

    speech_queue = deque()
    speech_started = [False]
    gate = SpeechGate()
    tail = None
    ja_started = False
    gaze_cleared = False
    t = 0
    while t < 1100:
        name = getattr(env, "current_protocol_name", None)
        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None
        if name == "joint_attention":
            ja_started = True
            runtime["mode"] = "idle"
            if not gaze_cleared:
                detector = find_signal_detector(env)
                if detector is not None and hasattr(detector, "gaze_targets"):
                    detector.gaze_targets.clear()
                gaze_cleared = True
            infant_action = ja_engage_action(
                env, red, speech_queue, speech_started, gate.ready
            )
        else:
            runtime["mode"] = "away"
            infant_action = look_parent_action(env)

        obs, reward, done, trunc, info, q_before = step_parent(env, info, infant_action)
        rec.after_step(obs, t, q_before)
        if rec.parent_phonemes:
            gate.update(
                rec.parent_phonemes[-1],
                getattr(env, "current_protocol_name", None) == "joint_attention",
            )
        rec.capture(info)

        name = getattr(env, "current_protocol_name", None)
        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None
        if t % 25 == 0 or (ja_started and name == "joint_attention" and t < 5):
            print(
                f"t={t:03d} proto={name} state={state} "
                f"init={getattr(env, 'current_infant_initiative', None)} "
                f"ready={gate.ready} speech_left={len(speech_queue)}"
            )
        left_ja = ja_started and name != "joint_attention"
        finished = left_ja and speech_started[0] and len(speech_queue) == 0
        if finished:
            # 待つスコアを下げているので、戻った直後に次が始まりやすい。最初の一回で切る。
            env.check_for_initiative = lambda signals: (False, "", {})
        if finished or (done and ja_started):
            if tail is None:
                tail = TAIL_AFTER_DONE
            tail -= 1
            if tail <= 0:
                print(f"Stop at t={t}")
                break
        t += 1

    result = ja_result(env, rec, t + 1, prefer="joint_attention")
    plays = [h for h in env.protocol_history if h.get("name") == "joint_attention"]
    print("===== Plays =====")
    for play in plays:
        print(play)
    print("===== Summary =====")
    print(result)
    ok = (
        len(plays) >= 1
        and plays[-1].get("success") is True
        and plays[-1].get("infant_initiative") is False
        and plays[-1].get("target_id") == red
        and plays[-1].get("kind") == "pointing"
        and word_match_label(plays[-1].get("word_match")) == "correct"
    )
    env.close()
    videos = rec.save_videos(STEM)
    lines_path = rec.write_lines(
        STEM,
        title=TITLE,
        situation_ja=SITUATION_JA,
        situation_en=SITUATION_EN,
        result=result,
    )
    write_result_json(
        STEM,
        {
            "stem": STEM,
            "title": TITLE,
            "situation_ja": SITUATION_JA,
            "situation_en": SITUATION_EN,
            "result": result,
            "videos": videos,
            "lines": lines_path,
        },
    )
    if not ok:
        raise SystemExit(f"{STEM} did not show a parent-started joint attention on the red cube")
    print("Done!")


if __name__ == "__main__":
    main()
