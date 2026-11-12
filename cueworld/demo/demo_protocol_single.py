"""コミュニケーションプロトコルの単体デモ。種類は --kind で選ぶ。"""
from __future__ import annotations

import argparse
from collections import deque

from demo_catalog_common import (
    DemoRecorder,
    SpeechGate,
    dyadic_respond_action,
    find_object,
    giving_engage_action,
    giving_object_placements,
    ja_engage_action,
    ja_result,
    look_parent_action,
    make_env,
    skip_point_while_looking,
    slide_object_placements,
    step_parent,
    write_result_json,
    TAIL_AFTER_DONE,
)


KINDS = {
    "ja_looking": {
        "stem": "demo_single_ja_looking",
        "title": "共同注意・見るだけ",
        "protocol": "joint_attention",
        "situation_ja": "親は見るだけで注意を向ける。幼児は赤い立方体を見て、親が言い終わってから対象語を言う。",
        "situation_en": "The parent orients by looking only. The infant looks at the red cube and names it after the parent finishes speaking.",
    },
    "ja_pointing": {
        "stem": "demo_single_ja_pointing",
        "title": "共同注意・指差し",
        "protocol": "joint_attention",
        "situation_ja": "親は赤い立方体を指差す。幼児は対象を見て、親が言い終わってから対象語を言う。",
        "situation_en": "The parent points at the red cube. The infant looks at it and names it after the parent finishes speaking.",
    },
    "ja_shake": {
        "stem": "demo_single_ja_shake",
        "title": "共同注意・掲げて見せる",
        "protocol": "joint_attention",
        "situation_ja": "親は赤い立方体を取りに行き、幼児の前で掲げて見せる。手首は振らない。幼児は対象を見て、親が言い終わってから対象語を言う。",
        "situation_en": "The parent fetches the red cube and holds it up without shaking. The infant looks at it and names it after the parent finishes speaking.",
    },
    "dyadic_voice": {
        "stem": "demo_single_dyadic_voice",
        "title": "二項・声",
        "protocol": "dyadic",
        "situation_ja": "親が音素の列を一度見せ、幼児が声と掲げで返す。親はその列を一度返す。",
        "situation_en": "The parent shows a short phoneme sequence once. The infant answers with voice and a raised arm, and the parent echoes that sequence once.",
    },
    "dyadic_arm": {
        "stem": "demo_single_dyadic_arm",
        "title": "二項・腕を上げる",
        "protocol": "dyadic",
        "situation_ja": "親が腕を振って見せ、幼児が上げて返す。成功すると見守りに戻る。",
        "situation_en": "The parent waves an arm. The infant raises an arm back. On success the parent returns to observance.",
    },
    "dyadic_hand": {
        "stem": "demo_single_dyadic_hand",
        "title": "二項・手の開閉",
        "protocol": "dyadic",
        "situation_ja": "親が腕を振ってから手を開閉する。幼児も上げて開閉して返す。成功すると見守りに戻る。",
        "situation_en": "The parent waves an arm and opens and closes the hand. The infant answers the same way. On success the parent returns to observance.",
    },
    "giving": {
        "stem": "demo_single_giving",
        "title": "受け渡し",
        "protocol": "object_giving",
        "situation_ja": "親は届かない赤い立方体を取りに行き、「どうぞ」と言って幼児の到達圏に置く。幼児が拾うと「すごい」と言う。",
        "situation_en": "The parent fetches the out-of-reach red cube, says here you go, and places it in reach. When the infant picks it up the parent says well done.",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Single communication-protocol demo")
    parser.add_argument("--kind", required=True, choices=sorted(KINDS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_frames", type=int, default=800)
    return parser.parse_args()


def context_for(kind: str, red) -> dict:
    if kind == "ja_looking":
        return {
            "fixed_target": red,
            "cue_type": "looking",
            "intent_type": "informative",
            "infant_initiative": False,
        }
    if kind == "ja_pointing":
        return {
            "fixed_target": red,
            "cue_type": "pointing",
            "intent_type": "informative",
            "infant_initiative": False,
        }
    if kind == "ja_shake":
        return {
            "fixed_target": red,
            "cue_type": "shake",
            "intent_type": "informative",
            "infant_initiative": False,
        }
    if kind == "dyadic_voice":
        return {
            "infant_initiative": False,
            "parent_channels": ["voice"],
            "target_phoneme": "p0",
        }
    if kind == "dyadic_arm":
        return {
            "infant_initiative": False,
            "parent_channels": ["arm_raise"],
        }
    if kind == "dyadic_hand":
        return {
            "infant_initiative": False,
            "parent_channels": ["arm_raise", "hand_open_close"],
        }
    return {
        "fixed_target": red,
        "infant_initiative": False,
    }


def main():
    args = parse_args()
    spec = KINDS[args.kind]
    stem = spec["stem"]
    print(f"Starting {stem}...")
    placements = giving_object_placements() if args.kind == "giving" else slide_object_placements()
    env = make_env(
        placements,
        max_steps=max(args.max_frames + 100, 900),
        seed=args.seed,
        fixed_protocol=None,
    )
    obs, info = env.reset(seed=args.seed)
    red = find_object(env, "RedCube")
    print(f"Red cube: {red}")
    env.set_fixed_protocol(spec["protocol"], context_for(args.kind, red))
    if args.kind == "ja_looking":
        skip_point_while_looking(env)

    rec = DemoRecorder(env)
    rec.append_voices(obs)
    rec.capture(info)

    speech_queue = deque()
    speech_started = [False]
    gate = SpeechGate()
    reaching_phase = [False]
    tail = None
    t = 0
    while t < args.max_frames:
        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None
        name = getattr(env, "current_protocol_name", None)

        if spec["protocol"] == "joint_attention":
            infant_action = ja_engage_action(
                env, red, speech_queue, speech_started, gate.ready
            )
        elif spec["protocol"] == "object_giving":
            infant_action = giving_engage_action(env, red, reaching_phase)
        elif args.kind == "dyadic_voice":
            infant_action = dyadic_respond_action(env, t, voice=True)
        elif args.kind == "dyadic_hand":
            infant_action = dyadic_respond_action(env, t, hand=True)
        elif args.kind == "dyadic_arm":
            infant_action = dyadic_respond_action(env, t)
        else:
            infant_action = look_parent_action(env)

        obs, reward, done, trunc, info, q_before = step_parent(env, info, infant_action)
        rec.after_step(obs, t, q_before)
        if spec["protocol"] == "joint_attention" and rec.parent_phonemes:
            gate.update(
                rec.parent_phonemes[-1],
                state in ("looking", "pointing", "shaking", "post_shared_attention", "completed"),
            )
        rec.capture(info)

        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None
        if t % 25 == 0:
            print(
                f"t={t:03d} proto={name} state={state} "
                f"speech_left={len(speech_queue)} done={done}"
            )
        play_done = bool(done or trunc or state in ("completed", "failure"))
        infant_ok = spec["protocol"] != "joint_attention" or (
            speech_started[0] and len(speech_queue) == 0
        )
        if play_done and infant_ok:
            if tail is None:
                tail = TAIL_AFTER_DONE
            tail -= 1
            if tail <= 0:
                print(f"Stop at t={t}")
                break
        t += 1

    result = ja_result(env, rec, t + 1)
    print("===== Summary =====")
    print(result)
    env.close()
    videos = rec.save_videos(stem)
    lines_path = rec.write_lines(
        stem,
        title=spec["title"],
        situation_ja=spec["situation_ja"],
        situation_en=spec["situation_en"],
        result=result,
    )
    write_result_json(
        stem,
        {
            "stem": stem,
            "title": spec["title"],
            "situation_ja": spec["situation_ja"],
            "situation_en": spec["situation_en"],
            "result": result,
            "videos": videos,
            "lines": lines_path,
        },
    )
    print("Done!")


if __name__ == "__main__":
    main()
