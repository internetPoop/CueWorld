"""指差しの共同注意が失敗し、同じ物を掲げて見せる共同注意が成功するデモ。"""
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


STEM = "demo_pm_ja_point_to_shake"
TITLE = "指差しのあと掲げて見せる"
SITUATION_JA = (
    "親は見守りから始める。幼児が赤い立方体を指差すと、指差しの共同注意が始まる。"
    "幼児は対象を見ないので失敗して見守りに戻る。"
    "次の共同注意は同じ赤い立方体を、最初から見せる合図で始める。"
    "親は取りに行って掲げて見せる。手首は振らない。"
    "幼児はその立方体を目で追い、親が言い終わってから対象語を言う。"
)
SITUATION_EN = (
    "The parent starts in observance. The infant points at the red cube and "
    "a pointing joint attention begins. The infant never looks at it, so the "
    "play fails and the parent returns to observance. The next joint attention "
    "uses the same red cube and starts by showing it. The parent fetches it "
    "and holds it up without shaking. The infant follows the cube with their "
    "gaze and names it after the parent finishes speaking."
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
    env = make_env(slide_object_placements(), max_steps=2000, seed=42)
    obs, info = env.reset(seed=42)
    red = find_object(env, "RedCube")
    print(f"Red cube: {red}")
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
    while t < 1400:
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
        elif phase in ("fail_ja", "gap"):
            runtime["mode"] = "idle"
            infant_action = look_parent_action(env)
        elif phase == "shake":
            # 取る前から立方体を見ておく。掲げたあとに視線を移すと、
            # 親の顔の履歴が残って、振っている物への注視がタイムアウトする。
            runtime["mode"] = "idle"
            infant_action = ja_engage_action(
                env, target or red, speech_queue, speech_started, gate.ready
            )
        else:
            runtime["mode"] = "idle"
            infant_action = look_parent_action(env)

        if phase == "gap" and idle_count + 1 >= IDLE_AFTER_PLAY and name == "observance":
            # くじは合図を指差しに固定している。同じ物の次の共同注意だけ、始める前に振るへ上書きする。
            env.check_for_response = lambda signals: (False, "", {})
            env.check_for_initiative = lambda signals: (False, "", {})
            env.current_infant_initiative = False
            env.switch_protocol(
                "joint_attention",
                {
                    "infant_initiative": False,
                    "cue_type": "shake",
                    "intent_type": "informative",
                    "fixed_target": red,
                    "infant_signals": {},
                    "global_ignored_targets": [],
                },
            )
            phase = "shake"
            reset_speech(speech_queue, speech_started, gate)
            rec.prev_utt_key = None
            print(
                f"t={t:03d} start shake cue={env.current_protocol.cue_type} "
                f"target={protocol_target_id(env.current_protocol)}"
            )

        obs, reward, done, trunc, info, q_before = step_parent(env, info, infant_action)
        rec.after_step(obs, t, q_before)
        name = getattr(env, "current_protocol_name", None)
        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None
        if rec.parent_phonemes:
            gate.update(
                rec.parent_phonemes[-1],
                name == "joint_attention" and phase == "shake",
            )
        rec.capture(info)

        if phase == "appeal_red" and name == "joint_attention":
            phase = "fail_ja"
            print(f"t={t:03d} first JA cue={getattr(protocol, 'cue_type', None)}")
        elif phase == "fail_ja" and name == "observance":
            phase = "gap"
            idle_count = 0
            reset_speech(speech_queue, speech_started, gate)
            silence_infant_voice(env)
            print(f"t={t:03d} pointing JA ended")
        elif phase == "gap":
            idle_count += 1
        elif phase == "shake" and name != "joint_attention":
            finished = (speech_started[0] and len(speech_queue) == 0) or not speech_started[0]
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
                f"cue={getattr(protocol, 'cue_type', None)} ready={gate.ready} "
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
        and plays[-1].get("kind") == "shake"
        and plays[-1].get("target_id") == red
        and plays[-1].get("infant_initiative") is False
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
        raise SystemExit(f"{STEM} did not show pointing failure then shake success")
    print("Done!")


if __name__ == "__main__":
    main()
