"""CommunicationEnv の依存ありと yoke を、同じ幼児の行動列で録画する。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".mplconfig"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

from envs.communication import CommunicationEnv

FONT_PATH = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
FONT = font_manager.FontProperties(fname=FONT_PATH)

HERE = Path(__file__).resolve().parent
GIF_PATH = HERE / "communication_demo.gif"
PNG_PATH = HERE / "communication_timeline.png"

# 0=沈黙, 1=記号0, 2=記号1
INFANT_ACTIONS = [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 2]

COLORS = {
    0: "#E4E4E4",
    1: "#4C78A8",
    2: "#F58518",
    3: "#54A24B",
    4: "#B279A2",
}
SYMBOL_NAME = {0: "静", 1: "0", 2: "1", 3: "2", 4: "3"}


def rollout(env: CommunicationEnv, actions: list[int], *, options=None):
    env.reset(seed=0, options=options)
    rows = []
    for action in actions:
        obs, _, _, _, info = env.step(action)
        rows.append(
            {
                "infant": info["infant_action"],
                "parent": info["parent_action"],
                "infant_energy": info["infant_energy"],
                "parent_energy": info["parent_energy"],
                "trigger_id": info["trigger_id"],
                "reply_step": info["reply_step"],
                "trigger_count": info["trigger_count"],
                "reply_heard_count": info["reply_heard_count"],
                "who": float(obs[1]),
                "yoke": info["yoke"],
            }
        )
    return rows


def record_silent_parent(n: int) -> np.ndarray:
    env = CommunicationEnv(max_steps=n, n_meaningful=1, seed=1)
    env.reset(seed=1)
    tape = []
    for _ in range(n):
        _, _, _, _, info = env.step(0)
        tape.append(info["parent_action"])
    return np.asarray(tape, dtype=np.int64)


def caption(row: dict, prev_heard: int) -> str:
    if row["yoke"] and row["trigger_id"] is None and row["infant"] != 0 and row["reply_step"] == 0:
        if row["infant"] == 1:
            return "yoke。幼児は記号 0 を出しているが、親の列は別エピソードの再生なので塊は届かない。"
        return "yoke。話しているあいだ見えるのは自分の信号だけ。"
    if row["trigger_id"] is not None:
        return "記号 0 が 2 つ続いた。塊が届き、次の 2 step で親は 2, 0 を返す。"
    if row["reply_step"] == 1 and row["infant"] == 0:
        return "返事の 1 文字目。幼児は沈黙しているので、相手の信号が見える。"
    if row["reply_step"] == 2 and row["reply_heard_count"] > prev_heard:
        return "返事の 2 文字目まで聞けた。この 2 step の観測は (2, 0) に決まっている。"
    if row["reply_step"] > 0 and row["infant"] != 0:
        return "返事の最中に幼児が話した。親の行には返事が出るが、幼児が見るのは自分の信号。"
    if row["infant"] == 0 and row["parent"] != 0 and row["reply_step"] == 0:
        return "幼児は沈黙。親はばらばらな信号を出していて、その信号が見える。"
    if row["infant"] == 0 and row["parent"] == 0:
        return "どちらも沈黙。信号は見えない。"
    if row["infant"] != 0 and row["reply_step"] == 0:
        return "幼児が信号を出している。親は黙って受け取り、幼児が見るのは自分の信号。"
    return ""


def draw_frame(records: list[dict], t: int, phase: str, caption_text: str):
    n = len(records)
    fig_w = max(11.5, 1.05 * n + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, 5.6), dpi=120)
    ax.set_xlim(-1.7, n + 0.3)
    ax.set_ylim(-1.55, 4.35)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        n / 2 - 0.5,
        4.05,
        phase,
        ha="center",
        va="center",
        fontsize=16,
        fontproperties=FONT,
        color="#222222",
    )
    ax.text(
        n / 2 - 0.5,
        3.45,
        caption_text,
        ha="center",
        va="center",
        fontsize=11,
        fontproperties=FONT,
        color="#333333",
        wrap=True,
    )

    row_y = {"infant": 2.15, "parent": 0.85}
    ax.text(-1.55, row_y["infant"] + 0.28, "幼児", fontproperties=FONT, fontsize=13, va="center")
    ax.text(-1.55, row_y["parent"] + 0.28, "親", fontproperties=FONT, fontsize=13, va="center")

    current = records[t]
    for i, row in enumerate(records):
        filled = i <= t
        for key, action_key in (("infant", "infant"), ("parent", "parent")):
            action = row[action_key] if filled else None
            color = COLORS[action] if action is not None else "white"
            edge = "#222222" if i == t else "#BBBBBB"
            width = 2.2 if i == t else 0.8
            y = row_y[key]
            patch = FancyBboxPatch(
                (i, y),
                0.86,
                0.72,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=color,
                edgecolor=edge,
                linewidth=width,
            )
            ax.add_patch(patch)
            if filled:
                ink = "#FFFFFF" if action != 0 else "#444444"
                ax.text(
                    i + 0.43,
                    y + 0.36,
                    SYMBOL_NAME[action],
                    ha="center",
                    va="center",
                    color=ink,
                    fontsize=13,
                    fontproperties=FONT,
                )
        if filled and row["trigger_id"] is not None:
            ax.plot([i + 0.43], [row_y["infant"] + 0.86], marker="v", color="#C59B08", markersize=8)
        if filled and row["reply_step"] > 0:
            ax.plot([i + 0.43], [row_y["parent"] - 0.16], marker="^", color="#E45756", markersize=8)
        ax.text(
            i + 0.43,
            0.28,
            str(i + 1),
            ha="center",
            va="center",
            fontsize=8,
            color="#888888",
            fontproperties=FONT,
        )

    heard = "自分" if current["who"] > 0.5 else ("相手" if current["who"] < -0.5 else "なし")
    ax.text(
        -1.55,
        -0.35,
        f"見ているもの: {heard}",
        fontproperties=FONT,
        fontsize=12,
        color="#222222",
    )
    ax.text(
        3.4,
        -0.35,
        f"届いた塊 {current['trigger_count']}    聞けた返事 {current['reply_heard_count']}",
        fontproperties=FONT,
        fontsize=12,
        color="#222222",
    )
    ax.text(
        -1.55,
        -0.95,
        "エネルギー  " + energy_bar(current["infant_energy"]) + "  幼児     "
        + energy_bar(current["parent_energy"]) + "  親",
        fontproperties=FONT,
        fontsize=12,
        color="#222222",
    )

    legend_x = n - 5.3
    for j, (name, color) in enumerate(
        (
            ("静", COLORS[0]),
            ("0", COLORS[1]),
            ("1", COLORS[2]),
            ("2", COLORS[3]),
            ("3", COLORS[4]),
        )
    ):
        x = legend_x + j * 1.05
        ax.add_patch(
            Rectangle((x, -1.25), 0.32, 0.32, facecolor=color, edgecolor="#888888")
        )
        ax.text(x + 0.4, -1.09, name, fontproperties=FONT, fontsize=10, va="center")

    fig.tight_layout()
    return fig


def energy_bar(value: int) -> str:
    return "●" * value + "○" * (4 - value)


def save_timeline(dependent: list[dict], yoke: list[dict]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 5.2), dpi=140)
    fig.patch.set_facecolor("white")
    for ax, rows, title in (
        (axes[0], dependent, "依存あり。▼は塊が届いた step、▲は返事"),
        (axes[1], yoke, "yoke。同じ幼児の行動でも、塊のあとに決まった返事は来ない"),
    ):
        ax.set_xlim(-1.5, len(rows) + 0.2)
        ax.set_ylim(-0.35, 2.15)
        ax.axis("off")
        ax.set_title(title, fontproperties=FONT, fontsize=12, loc="left", color="#222222")
        for i, row in enumerate(rows):
            for key, y in (("infant", 1.15), ("parent", 0.15)):
                action = row[key]
                ax.add_patch(
                    FancyBboxPatch(
                        (i, y),
                        0.86,
                        0.72,
                        boxstyle="round,pad=0.02,rounding_size=0.08",
                        facecolor=COLORS[action],
                        edgecolor="#CCCCCC",
                        linewidth=0.6,
                    )
                )
                ink = "#FFFFFF" if action != 0 else "#444444"
                ax.text(
                    i + 0.43,
                    y + 0.36,
                    SYMBOL_NAME[action],
                    ha="center",
                    va="center",
                    color=ink,
                    fontsize=11,
                    fontproperties=FONT,
                )
            if row["trigger_id"] is not None:
                ax.plot([i + 0.43], [1.98], marker="v", color="#C59B08", markersize=7)
            if row["reply_step"] > 0:
                ax.plot([i + 0.43], [0.02], marker="^", color="#E45756", markersize=7)
        ax.text(-1.35, 1.5, "幼児", fontproperties=FONT, fontsize=11, va="center")
        ax.text(-1.35, 0.5, "親", fontproperties=FONT, fontsize=11, va="center")
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)


def main() -> None:
    n = len(INFANT_ACTIONS)
    env = CommunicationEnv(max_steps=n, n_meaningful=1, seed=0)
    dependent = rollout(env, INFANT_ACTIONS)

    tape = record_silent_parent(n)
    yoke_env = CommunicationEnv(max_steps=n, n_meaningful=1, seed=0)
    yoke = rollout(yoke_env, INFANT_ACTIONS, options={"yoke_parent_actions": tape})

    save_timeline(dependent, yoke)

    frames = []
    sequences = (
        (dependent, "依存あり"),
        (yoke, "yoke（親の行動は、幼児がずっと沈黙していた別エピソード）"),
    )
    for records, phase in sequences:
        prev_heard = 0
        for t, row in enumerate(records):
            text = caption(row, prev_heard)
            fig = draw_frame(records, t, phase, text)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            frames.append(buf.copy())
            prev_heard = row["reply_heard_count"]
            plt.close(fig)

    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        GIF_PATH,
        save_all=True,
        append_images=images[1:],
        duration=1100,
        loop=0,
        disposal=2,
    )
    print(f"wrote {GIF_PATH}")
    print(f"wrote {PNG_PATH}")


if __name__ == "__main__":
    main()
