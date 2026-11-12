"""赤い立方体への共同注意が失敗し、緑への共同注意が成功するデモ。"""
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
    protocol_target_id,
    silence_infant_voice,
    slide_object_placements,
    step_parent,
    word_match_label,
    write_result_json,
    IDLE_AFTER_PLAY,
    TAIL_AFTER_DONE,
)


STEM = "demo_pm_ja_retarget"
TITLE = "別の物への共同注意"
SITUATION_JA = (
    "親は見守りから始める。幼児が赤い立方体を指差すと共同注意が始まる。"
    "幼児は対象を見ないので、指差しのまま失敗して見守りに戻る。"
    "失敗した物は除外される。続けて幼児が緑の立方体を指差すと、"
    "別の物への共同注意が始まり、幼児は親が言い終わってから対象語を言って成功する。"
)
SITUATION_EN = (
    "The parent starts in observance. The infant points at the red cube and "
    "joint attention begins. The infant never looks at it, so pointing fails "
    "and the parent returns to observance. That object is then blocked. "
    "The infant points at the green cube, joint attention starts on the new object, "
    "and the infant names it after the parent finishes speaking."
)


def log_repeated_templates(protocol, recorder) -> None:
    """同じテンプレの言い直しも台本に残す。"""
    orig = protocol.queue_utterance
    seen = {"n": 0}

    def wrapped(template_key, concept_id=None):
        seen["n"] += 1
        if seen["n"] > 1:
            recorder.prev_utt_key = None
        return orig(template_key, concept_id)

    protocol.queue_utterance = wrapped


def reset_speech(speech_queue, speech_started, gate) -> None:
    speech_queue.clear()
    speech_started[0] = False
    gate.ready = False
    gate.heard = False
    gate.silent = 0


def main():
    print(f"Starting {STEM}...")
    env = make_env(slide_object_placements(), max_steps=1600, seed=42)
    obs, info = env.reset(seed=42)
    red = find_object(env, "RedCube")
    green = find_object(env, "GreenCube")
    print(f"objects red={red} green={green}")
    runtime = {"mode": "idle", "inject_id": None}
    install_signal_override(env, runtime)

    rec = DemoRecorder(env)
    log_repeated_templates(env.protocols["joint_attention"], rec)
    rec.append_voices(obs)
    rec.capture(info)

    speech_queue = deque()
    speech_started = [False]
    gate = SpeechGate()
    phase = "appeal_red"
    idle_count = 0
    tail = None
    t = 0
    while t < 1100:
        name = getattr(env, "current_protocol_name", None)
        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None
        target = protocol_target_id(protocol)

        if phase == "appeal_red":
            if name == "observance" and t >= 20:
                runtime["mode"] = "point"
                runtime["inject_id"] = red
                infant_action = look_object_action(env, red, point=True)
            else:
                runtime["mode"] = "idle"
                infant_action = look_parent_action(env)
        elif phase == "fail_ja":
            runtime["mode"] = "idle"
            infant_action = look_parent_action(env)
        elif phase == "gap":
            runtime["mode"] = "idle"
            infant_action = look_parent_action(env)
        elif phase == "appeal_green":
            if name == "joint_attention":
                phase = "success_ja"
                runtime["mode"] = "idle"
                infant_action = ja_engage_action(
                    env, target or green, speech_queue, speech_started, gate.ready
                )
            else:
                runtime["mode"] = "point"
                runtime["inject_id"] = green
                infant_action = look_object_action(env, green, point=True)
        else:
            runtime["mode"] = "idle"
            infant_action = ja_engage_action(
                env, target or green, speech_queue, speech_started, gate.ready
            )

        obs, reward, done, trunc, info, q_before = step_parent(env, info, infant_action)
        rec.after_step(obs, t, q_before)
        name = getattr(env, "current_protocol_name", None)
        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None
        if rec.parent_phonemes:
            gate.update(
                rec.parent_phonemes[-1],
                name == "joint_attention" and phase == "success_ja",
            )
        rec.capture(info)

        if phase == "appeal_red" and name == "joint_attention":
            phase = "fail_ja"
            print(f"t={t:03d} first JA target={protocol_target_id(protocol)}")
        elif phase == "fail_ja" and name == "observance":
            phase = "gap"
            idle_count = 0
            reset_speech(speech_queue, speech_started, gate)
            silence_infant_voice(env)
            print(f"t={t:03d} first JA ended blocked={list(env.blocked_pairs)}")
        elif phase == "gap":
            idle_count += 1
            if idle_count >= IDLE_AFTER_PLAY:
                phase = "appeal_green"
                reset_speech(speech_queue, speech_started, gate)
        elif phase == "success_ja" and name != "joint_attention":
            finished = speech_started[0] and len(speech_queue) == 0
            if finished or done:
                if tail is None:
                    tail = TAIL_AFTER_DONE
                tail -= 1
                if tail <= 0:
                    print(f"Stop at t={t}")
                    break

        if t % 25 == 0:
            print(
                f"t={t:03d} phase={phase} proto={name} state={state} "
                f"target={protocol_target_id(protocol)} ready={gate.ready} "
                f"speech_left={len(speech_queue)}"
            )
        t += 1

    result = ja_result(env, rec, t + 1, prefer="joint_attention")
    plays = [h for h in env.protocol_history if h.get("name") == "joint_attention"]
    print("===== Plays =====")
    for play in plays:
        print(play)
    print("===== Summary =====")
    print(result)
    ok = (
        len(plays) >= 2
        and plays[0].get("success") is False
        and plays[0].get("kind") == "pointing"
        and plays[0].get("target_id") == red
        and plays[-1].get("success") is True
        and plays[-1].get("kind") == "pointing"
        and plays[-1].get("target_id") == green
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
        raise SystemExit(f"{STEM} did not show red failure then green success")
    print("Done!")


if __name__ == "__main__":
    main()
