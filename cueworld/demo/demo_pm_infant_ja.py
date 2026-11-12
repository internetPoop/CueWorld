"""見守りから、幼児の指差しで共同注意が始まるデモ。fixed_protocol は使わない。"""
from __future__ import annotations

from collections import deque

from demo_catalog_common import (
    DemoRecorder,
    SpeechGate,
    find_object,
    install_signal_override,
    ja_engage_action,
    ja_result,
    look_object_action,
    look_parent_action,
    make_env,
    slide_object_placements,
    step_parent,
    write_result_json,
    TAIL_AFTER_DONE,
)


STEM = "demo_pm_infant_ja"
TITLE = "幼児の指差しから共同注意"
SITUATION_JA = (
    "親は見守りから始める。幼児が赤い立方体を指差すと、"
    "プロトコルマネージャの訴えくじが共同注意を選ぶ。"
    "幼児は対象を見て、親が「これ、あかいしかく」を言い終わってから対象語を言う。"
)
SITUATION_EN = (
    "The parent starts in observance. The infant points at the red cube, "
    "and the protocol manager's response lottery starts joint attention. "
    "The infant looks at the object and names it after the parent finishes labeling."
)


def main():
    print(f"Starting {STEM}...")
    env = make_env(slide_object_placements(), max_steps=900, seed=42)
    obs, info = env.reset(seed=42)
    red = find_object(env, "RedCube")
    print(f"Red cube: {red}")
    runtime = {"mode": "idle", "inject_id": red}
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
    while t < 700:
        name = getattr(env, "current_protocol_name", None)
        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None
        if name == "observance" and t >= 20 and not ja_started:
            runtime["mode"] = "point"
            runtime["inject_id"] = red
            infant_action = look_object_action(env, red, point=True)
        elif name == "joint_attention":
            ja_started = True
            runtime["mode"] = "idle"
            allow = state in ("looking", "pointing", "shaking", "post_shared_attention", "completed")
            infant_action = ja_engage_action(
                env, red, speech_queue, speech_started, gate.ready
            )
        else:
            runtime["mode"] = "idle"
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
