"""見守りを挟み、訴えでプロトコルを4回始める通しデモ。切替は7回。"""
from __future__ import annotations

from collections import deque

import pybullet as p

from demo_catalog_common import (
    DemoRecorder,
    SpeechGate,
    dyadic_respond_action,
    find_object,
    giving_engage_action,
    install_signal_override,
    ja_engage_action,
    ja_result,
    look_object_action,
    look_parent_action,
    make_env,
    protocol_target_id,
    raise_action,
    silence_infant_voice,
    sequence_object_placements,
    step_parent,
    write_result_json,
    IDLE_AFTER_PLAY,
    TAIL_AFTER_DONE,
)


STEM = "demo_pm_sequence"
TITLE = "動的切替の通し"
SITUATION_JA = (
    "親はプロトコルマネージャのくじで選ぶ。幼児は完成形の訴えだけを出し分ける。"
    "見守り → 赤い立方体への指差しで共同注意 → 見守り → "
    "届かない青い立方体へのリーチで受け渡し → 見守り → "
    "腕を上げて二項 → 見守り → 緑の立方体への指差しで共同注意。"
)
SITUATION_EN = (
    "The parent chooses with the protocol manager lottery. "
    "The infant produces only complete-form appeals: "
    "pointing at the red cube for joint attention, reaching for the far blue cube "
    "for object giving, raising an arm for dyadic, then pointing at the green cube "
    "for a second joint attention. Observance is between each play."
)

# 次に出したい訴え。見守りに戻って IDLE_AFTER_PLAY 経ったら出す。
APPEALS = ("point_red", "reach_blue", "dyadic", "point_green")


def _body_snapshot(env, body_id):
    cid = env.client_id
    pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=cid)
    n = p.getNumJoints(body_id, physicsClientId=cid)
    joints = [p.getJointState(body_id, i, physicsClientId=cid)[0] for i in range(n)]
    return pos, orn, joints


def _restore_body(env, body_id, snap) -> None:
    """二項のあと、親が幼児の眼前に残ると次の共同注意で視線が親で止まる。"""
    pos, orn, joints = snap
    cid = env.client_id
    p.resetBasePositionAndOrientation(body_id, pos, orn, physicsClientId=cid)
    for index, angle in enumerate(joints):
        p.resetJointState(body_id, index, angle, physicsClientId=cid)


def main():
    print(f"Starting {STEM}...")
    env = make_env(sequence_object_placements(), max_steps=2500, seed=42)
    obs, info = env.reset(seed=42)
    parent_home = _body_snapshot(env, env.parent["body"])
    infant_home = _body_snapshot(env, env.infant["body"])
    red = find_object(env, "RedCube")
    green = find_object(env, "GreenCube")
    blue = find_object(env, "BlueCube")
    print(f"objects red={red} green={green} blue={blue}")

    runtime = {"mode": "idle", "inject_id": None}
    install_signal_override(env, runtime)
    # 訴えを出す前の見守りでは、残りの小さい確率で別プロトコルが始まらないようにする。
    response_gate = {"open": False}
    orig_check = env.check_for_response

    def gated_check(signals):
        if not response_gate["open"]:
            return False, "", {}
        return orig_check(signals)

    env.check_for_response = gated_check

    rec = DemoRecorder(env)
    rec.append_voices(obs)
    rec.capture(info)

    speech_queue = deque()
    speech_started = [False]
    gate = SpeechGate()
    reaching_phase = [False]
    idle_count = 0
    appeal_i = 0
    tail = None
    t = 0
    while t < 2200:
        name = getattr(env, "current_protocol_name", None)
        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None
        play_target = protocol_target_id(protocol)

        if name == "observance":
            if appeal_i < len(APPEALS):
                idle_count += 1
            runtime["mode"] = "idle"
            infant_action = look_parent_action(env)
            if appeal_i < len(APPEALS) and idle_count >= IDLE_AFTER_PLAY:
                kind = APPEALS[appeal_i]
                if kind == "point_red":
                    runtime["mode"] = "point"
                    runtime["inject_id"] = red
                    infant_action = look_object_action(env, red, point=True)
                elif kind == "reach_blue":
                    runtime["mode"] = "reach"
                    runtime["inject_id"] = blue
                    infant_action = look_object_action(env, blue, reach=True)
                elif kind == "dyadic":
                    runtime["mode"] = "dyadic"
                    runtime["inject_id"] = None
                    infant_action = raise_action(env)
                elif kind == "point_green":
                    runtime["mode"] = "point"
                    runtime["inject_id"] = green
                    infant_action = look_object_action(env, green, point=True)
        elif name == "joint_attention":
            runtime["mode"] = "idle"
            target = play_target or (green if APPEALS[min(appeal_i, 3)] == "point_green" else red)
            infant_action = ja_engage_action(
                env, target, speech_queue, speech_started, gate.ready
            )
        elif name == "object_giving":
            runtime["mode"] = "idle"
            infant_action = giving_engage_action(env, play_target or blue, reaching_phase)
        elif name == "dyadic":
            runtime["mode"] = "idle"
            infant_action = dyadic_respond_action(env, t)
        else:
            runtime["mode"] = "idle"
            infant_action = look_parent_action(env)

        response_gate["open"] = not (
            name == "observance" and appeal_i < len(APPEALS) and idle_count < IDLE_AFTER_PLAY
        )
        obs, reward, done, trunc, info, q_before = step_parent(env, info, infant_action)
        rec.after_step(obs, t, q_before)
        name = getattr(env, "current_protocol_name", None)
        if rec.parent_phonemes:
            gate.update(rec.parent_phonemes[-1], name == "joint_attention")
        rec.capture(info)

        # 関わりが始まっていたら、その訴えは消費する。見守りへ戻ったら次の訴えへ。
        if name in ("joint_attention", "object_giving", "dyadic"):
            if appeal_i < len(APPEALS):
                expected = {
                    "point_red": "joint_attention",
                    "reach_blue": "object_giving",
                    "dyadic": "dyadic",
                    "point_green": "joint_attention",
                }[APPEALS[appeal_i]]
                if name == expected:
                    # 開始した訴えは消費せず、終了して見守りに戻るまで保持。
                    pass
        if rec.switches and rec.switches[-1]["t"] == t:
            arrived = rec.switches[-1]["to"]
            left = rec.switches[-1]["from"]
            if left in ("joint_attention", "object_giving", "dyadic") and arrived == "observance":
                if left == "dyadic":
                    _restore_body(env, env.parent["body"], parent_home)
                    _restore_body(env, env.infant["body"], infant_home)
                if appeal_i < len(APPEALS):
                    appeal_i += 1
                    idle_count = 0
                    speech_queue.clear()
                    speech_started[0] = False
                    gate.ready = False
                    gate.heard = False
                    gate.silent = 0
                    reaching_phase[0] = False
                    silence_infant_voice(env)
            if arrived in ("joint_attention", "object_giving", "dyadic"):
                idle_count = 0

        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None
        if t % 25 == 0:
            print(
                f"t={t:03d} proto={name} state={state} appeal={appeal_i} "
                f"switches={len(rec.switches)} speech_left={len(speech_queue)}"
            )
        last_done = appeal_i >= len(APPEALS) and name == "observance"
        if last_done:
            if tail is None:
                tail = TAIL_AFTER_DONE
            tail -= 1
            if tail <= 0:
                print(f"Stop at t={t}")
                break
        t += 1

    result = ja_result(env, rec, t + 1)
    result["appeals_done"] = appeal_i
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
