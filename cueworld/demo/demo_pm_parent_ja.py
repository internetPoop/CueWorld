"""見守りから、幼児の訴えなしで親が共同注意を始めるデモ。fixed_protocol は使わない。"""
from __future__ import annotations

from collections import deque

from demo_catalog_common import (
    DemoRecorder,
    SpeechGate,
    close_object_placements,
    find_object,
    install_signal_override,
    ja_engage_action,
    ja_result,
    look_object_action,
    make_env,
    step_parent,
    write_result_json,
    TAIL_AFTER_DONE,
)


STEM = "demo_pm_parent_ja"
TITLE = "親始動の共同注意"
SITUATION_JA = (
    "親は見守りから始める。幼児は赤い立方体を見るだけで、指差し・リーチ・単語・二項の訴えは出さない。"
    "訴えくじは止め、親始動の候補は共同注意だけにする。"
    "未試行だと共同注意は ZPD で候補外なので、デモ側で発達記録だけ開けて選ばせる。"
    "幼児は親が言い終わってから対象語を言う。"
)
SITUATION_EN = (
    "The parent starts in observance. The infant only looks at the red cube "
    "and does not produce pointing, reach, word, or dyadic appeals. "
    "The response lottery is off and the only initiative candidate is joint attention. "
    "Level records are opened so joint attention is inside the ZPD. "
    "The infant names the object after the parent finishes speaking."
)


def main():
    print(f"Starting {STEM}...")
    env = make_env(close_object_placements(), max_steps=1200, seed=42, wait_base_score=0.72)
    env.request_protocol = [n for n in env.request_protocol if n == "joint_attention"]
    # 未試行だと ZPD の上限が 0 で、level 2 の共同注意が親始動の候補に入らない。
    n_levels = len(env.level_try_counts)
    env.level_try_counts[:] = [env.base_try_count] * n_levels
    env.level_scores[:] = [0.5] * n_levels
    # 訴えが無いときも response くじはごく小さく他を引く。親始動だけを見せる。
    env.check_for_response = lambda signals: (False, "", {})
    obs, info = env.reset(seed=42)
    red = find_object(env, "RedCube")
    print(f"Red cube: {red}")
    runtime = {"mode": "gaze", "inject_id": red}
    install_signal_override(env, runtime)

    rec = DemoRecorder(env)
    rec.append_voices(obs)
    rec.capture(info)

    speech_queue = deque()
    speech_started = [False]
    gate = SpeechGate()
    tail = None
    ja_started = False
    t = 0
    while t < 900:
        name = getattr(env, "current_protocol_name", None)
        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None
        if name == "joint_attention":
            ja_started = True
            runtime["mode"] = "idle"
            infant_action = ja_engage_action(
                env, red, speech_queue, speech_started, gate.ready
            )
        else:
            runtime["mode"] = "gaze"
            runtime["inject_id"] = red
            infant_action = look_object_action(env, red)

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
        if t % 25 == 0:
            print(
                f"t={t:03d} proto={name} state={state} "
                f"init={getattr(env, 'current_infant_initiative', None)} "
                f"ready={gate.ready} speech_left={len(speech_queue)}"
            )
        left_ja = ja_started and name != "joint_attention"
        finished = left_ja and speech_started[0] and len(speech_queue) == 0
        if finished or (done and ja_started):
            if tail is None:
                tail = TAIL_AFTER_DONE
            tail -= 1
            if tail <= 0:
                print(f"Stop at t={t}")
                break
        t += 1

    result = ja_result(env, rec, t + 1, prefer="joint_attention")
    print("===== Summary =====")
    print(result)
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
    print("Done!")


if __name__ == "__main__":
    main()
