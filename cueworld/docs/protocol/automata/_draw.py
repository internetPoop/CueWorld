"""オートマトン図を PNG にする。マークダウンのプレビューは mermaid を描かないため。"""
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "img"
REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

STATE = ("#e8f1fb", "#2f5f8a")
OK = ("#e7f6ec", "#2e7d4f")
BAD = ("#fdecea", "#b33a3a")
INK = "#1b2838"
ARROW = "#3d4f63"


class Fig:
    def __init__(self, w, h):
        self.im = Image.new("RGB", (w, h), "#ffffff")
        self.d = ImageDraw.Draw(self.im)
        self.w, self.h = w, h

    def font(self, size, bold=False):
        return ImageFont.truetype(BOLD if bold else REG, size, index=0)

    def text_size(self, text, font):
        b = self.d.textbbox((0, 0), text, font=font)
        return b[2] - b[0], b[3] - b[1]

    def text_center(self, cx, cy, lines, size=16, fill=INK, bold=False, spacing=3):
        font = self.font(size, bold=bold)
        metrics = [self.text_size(line, font) for line in lines]
        total = sum(h for _, h in metrics) + spacing * (len(lines) - 1)
        y = cy - total / 2
        for line, (w, h) in zip(lines, metrics):
            self.d.text((cx - w / 2, y - 1), line, font=font, fill=fill)
            y += h + spacing

    def box(self, x, y, w, h, lines, kind="state", size=16):
        fill, outline = {"state": STATE, "ok": OK, "bad": BAD}[kind]
        radius = h // 2 if kind in ("ok", "bad") else 14
        self.d.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill, outline=outline, width=2)
        self.text_center(x + w / 2, y + h / 2, lines, size=size)
        return (x, y, w, h)

    def diamond(self, cx, cy, w, h, lines, size=15):
        pts = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)]
        self.d.polygon(pts, fill="#fff6df")
        self.d.line(pts + [pts[0]], fill="#b8860b", width=2)
        self.text_center(cx, cy, lines, size=size)
        return (cx - w / 2, cy - h / 2, w, h)

    def side(self, box, edge, t=0.5):
        x, y, w, h = box
        if edge == "n":
            return (x + w * t, y)
        if edge == "s":
            return (x + w * t, y + h)
        if edge == "w":
            return (x, y + h * t)
        if edge == "e":
            return (x + w, y + h * t)
        raise ValueError(edge)

    def arrow(self, points):
        self.d.line(points, fill=ARROW, width=2, joint="curve")
        x2, y2 = points[-1]
        x1, y1 = points[-2]
        ang = math.atan2(y2 - y1, x2 - x1)
        size = 10
        p1 = (x2 + size * math.cos(ang + 2.6), y2 + size * math.sin(ang + 2.6))
        p2 = (x2 + size * math.cos(ang - 2.6), y2 + size * math.sin(ang - 2.6))
        self.d.polygon([(x2, y2), p1, p2], fill=ARROW)

    def label(self, x, y, text, anchor="mm"):
        font = self.font(13)
        w, h = self.text_size(text, font)
        if anchor == "mm":
            bx, by = x - w / 2, y - h / 2
        elif anchor == "lm":
            bx, by = x, y - h / 2
        elif anchor == "rm":
            bx, by = x - w, y - h / 2
        elif anchor == "lt":
            bx, by = x, y
        else:
            raise ValueError(anchor)
        pad = 2
        self.d.rectangle([bx - pad, by - pad, bx + w + pad, by + h + pad], fill="#ffffff")
        self.d.text((bx, by), text, font=font, fill="#3d4f63")

    def loop_east(self, box, text, where="right"):
        x, y, w, h = box
        y1 = y + h * 0.28
        y2 = y + h * 0.74
        x2 = x + w + 22
        self.arrow([(x + w, y1), (x2, y1), (x2, y2), (x + w, y2)])
        if where == "above":
            self.label(x2, y1 - 14, text)
        else:
            self.label(x2 + 6, (y1 + y2) / 2, text, "lm")

    def loop_north(self, box, text):
        x, y, w, h = box
        x1 = x + w * 0.32
        x2 = x + w * 0.68
        y2 = y - 26
        self.arrow([(x1, y), (x1, y2), (x2, y2), (x2, y)])
        self.label((x1 + x2) / 2, y2 - 16, text)

    def loop_west(self, box, text):
        x, y, w, h = box
        y1 = y + h * 0.28
        y2 = y + h * 0.74
        x2 = x - 22
        self.arrow([(x, y1), (x2, y1), (x2, y2), (x, y2)])
        self.label(x2 - 6, (y1 + y2) / 2, text, "rm")

    def legend(self, x, y, lang="ja"):
        if lang == "en":
            names = ("State", "Branch", "Success", "Failure")
        else:
            names = ("状態", "分岐", "成功終了", "失敗終了")
        items = [
            (STATE, names[0]),
            (("#fff6df", "#b8860b"), names[1]),
            (OK, names[2]),
            (BAD, names[3]),
        ]
        font = self.font(13)
        cx = x
        for (fill, outline), name in items:
            self.d.rounded_rectangle([cx, y, cx + 18, y + 14], radius=3, fill=fill, outline=outline, width=2)
            self.d.text((cx + 24, y - 2), name, font=font, fill=INK)
            cx += 24 + self.text_size(name, font)[0] + 16

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / name
        self.im.save(path, "PNG")
        print(path)


def dyadic():
    f = Fig(1240, 860)
    f.legend(36, 18)
    start = f.box(520, 48, 160, 42, ["開始"], "ok", size=16)
    entry = f.diamond(600, 168, 210, 96, ["誰が始めたか"])
    imit = f.box(36, 340, 250, 70, ["imitating", "真似る"])
    wait = f.box(380, 340, 290, 70, ["waiting_infant", "幼児の合図を待つ"])
    cue = f.box(820, 340, 300, 70, ["performing_parent_cue", "親が合図を見せる"])
    resp = f.box(820, 500, 300, 70, ["waiting_infant_response", "反応を待つ"])
    ack = f.box(820, 660, 300, 64, ["acknowledging", "すごい"])
    fail = f.box(425, 510, 200, 46, ["failure"], "bad")
    done = f.box(61, 750, 200, 46, ["completed"], "ok")

    f.arrow([f.side(start, "s"), f.side(entry, "n")])
    f.arrow([f.side(entry, "s", 0.2), (f.side(entry, "s", 0.2)[0], 290), (161, 290), f.side(imit, "n")])
    f.label(250, 274, "幼児始動・音素あり")
    f.arrow([f.side(entry, "s"), (f.side(entry, "s")[0], 310), f.side(wait, "n")])
    f.label(640, 300, "音素なし", "lm")
    f.arrow([f.side(entry, "s", 0.8), (f.side(entry, "s", 0.8)[0], 290), (970, 290), f.side(cue, "n")])
    f.label(860, 274, "親始動")

    f.arrow([f.side(wait, "w"), f.side(imit, "e")])
    f.label(333, 362, "閾値以上")
    f.loop_east(wait, "50 step・1回目", where="above")
    f.arrow([f.side(wait, "s"), f.side(fail, "n")])
    f.label(545, 470, "50 step・2回目")
    f.arrow([f.side(imit, "s"), (161, 773), f.side(done, "n")])
    f.label(175, 560, "チャネル完了", "lm")

    f.arrow([f.side(cue, "s"), f.side(resp, "n")])
    f.label(990, 460, "キュー完了")
    f.arrow([f.side(resp, "s"), f.side(ack, "n")])
    f.label(990, 620, "反応あり")
    f.arrow([(1120, 550), (1188, 550), (1188, 375), (1120, 375)])
    f.label(1176, 460, "50 step・1回目", "rm")
    f.arrow([f.side(resp, "w"), f.side(fail, "e")])
    f.label(720, 522, "50 step・2回目")
    f.arrow([f.side(ack, "s"), (970, 773), f.side(done, "e")])
    f.label(620, 758, "発話キューが空")
    f.save("dyadic.png")


def observance():
    f = Fig(720, 180)
    f.legend(24, 16)
    start = f.box(40, 70, 120, 44, ["開始"], "ok")
    obs = f.box(240, 56, 280, 72, ["observing", "顔を幼児へ、腕を戻す"])
    f.arrow([f.side(start, "e"), f.side(obs, "w")])
    f.loop_east(obs, "毎 step")
    f.save("observance.png")


def prerelease():
    f = Fig(980, 220)
    f.legend(24, 16)
    hold = f.diamond(150, 120, 200, 100, ["持っている？"])
    rel = f.box(390, 90, 180, 60, ["指を開く"])
    fsm = f.box(700, 40, 230, 48, ["本体の状態機械"], "ok", size=15)
    fail = f.box(730, 140, 170, 46, ["failure"], "bad")
    f.arrow([f.side(hold, "n"), (150, 40), (815, 40), f.side(fsm, "n")])
    f.label(420, 28, "持っていない・持ち続けてよい")
    f.arrow([f.side(hold, "e"), f.side(rel, "w")])
    f.label(300, 108, "手放す")
    f.arrow([f.side(rel, "e"), (640, 120), (640, 64), f.side(fsm, "w")])
    f.label(648, 96, "離れた", "lm")
    f.arrow([f.side(rel, "s"), (480, 163), f.side(fail, "w")])
    f.label(560, 150, "30 step で離れない")
    f.save("prerelease.png")


def household():
    f = Fig(1140, 1140)
    f.legend(24, 16)
    start = f.box(400, 44, 140, 42, ["開始"], "ok")
    task = f.diamond(470, 150, 190, 86, ["タスク候補"])
    item = f.diamond(470, 300, 210, 90, ["item 選択の直後"])
    hes = f.box(30, 400, 190, 52, ["hesitating"])
    ins_i = f.box(760, 400, 220, 52, ["inspecting_item"])
    moving = f.box(330, 470, 280, 52, ["moving_to_item"])
    grasping = f.box(330, 590, 280, 52, ["grasping_item"])
    grasped = f.diamond(470, 740, 200, 86, ["掴んだ直後"])
    ins_b = f.box(760, 714, 230, 52, ["inspecting_board"])
    to_board = f.box(330, 850, 280, 52, ["moving_to_board"])
    placing = f.box(330, 960, 280, 52, ["placing_on_board"])
    check = f.box(370, 1070, 200, 48, ["check_done"])
    done = f.box(780, 1072, 200, 46, ["completed"], "ok")

    f.arrow([f.side(start, "s"), f.side(task, "n")])
    f.arrow([f.side(task, "e"), (900, 150), (900, 1095), f.side(done, "n")])
    f.label(910, 170, "なし・不能", "lm")
    f.arrow([f.side(task, "s"), f.side(item, "n")])
    f.label(490, 220, "1件選ぶ", "lm")

    f.arrow([f.side(item, "w"), (125, 300), f.side(hes, "n")])
    f.label(140, 340, "迷い", "lm")
    f.arrow([f.side(hes, "e"), (250, 426), (250, 496), f.side(moving, "w")])
    f.label(200, 460, "25 step")
    f.arrow([f.side(item, "e"), (870, 300), f.side(ins_i, "n")])
    f.label(760, 340, "item を見る")
    f.arrow([f.side(ins_i, "s"), (870, 496), f.side(moving, "e")])
    f.label(880, 460, "40 step", "lm")
    f.arrow([f.side(item, "s"), f.side(moving, "n")])
    f.label(490, 430, "直接", "lm")

    f.arrow([f.side(moving, "s"), f.side(grasping, "n")])
    f.label(490, 555, "距離 0.6 未満", "lm")
    f.arrow([f.side(grasping, "s"), f.side(grasped, "n")])
    f.label(490, 680, "掴めた", "lm")
    f.arrow([f.side(grasping, "w"), (16, 616), (16, 268), (365, 268), f.side(item, "w")])
    f.label(24, 480, "100 step・別 item", "lm")
    f.arrow([f.side(grasping, "e"), (1100, 616), (1100, 1095), f.side(done, "e")])
    f.label(900, 640, "100 step・候補なし")

    f.arrow([f.side(grasped, "e"), f.side(ins_b, "w")])
    f.label(680, 728, "board を見る")
    f.arrow([f.side(ins_b, "s"), (875, 876), f.side(to_board, "e")])
    f.label(890, 820, "35 step", "lm")
    f.arrow([f.side(grasped, "s"), f.side(to_board, "n")])
    f.label(490, 810, "直接", "lm")
    f.arrow([f.side(to_board, "s"), f.side(placing, "n")])
    f.label(490, 930, "距離 0.45 未満", "lm")
    f.arrow([f.side(placing, "e"), (700, 986), (700, 876), f.side(to_board, "e", 0.7)])
    f.label(710, 930, "200 step・まだ持っている", "lm")
    f.arrow([f.side(placing, "w"), (70, 986), (70, 248), (430, 248), f.side(item, "n")])
    f.label(78, 700, "200 step・別 item", "lm")
    f.arrow([f.side(placing, "s"), f.side(check, "n")])
    f.label(490, 1040, "board に乗った", "lm")
    f.arrow([f.side(check, "e"), f.side(done, "w")])
    f.label(640, 1084, "1個")
    f.save("household_solo.png")


def object_giving():
    f = Fig(980, 1180)
    f.legend(24, 16)
    start = f.box(390, 48, 140, 42, ["開始"], "ok")
    fetching = f.box(300, 140, 320, 64, ["fetching", "取りに行く"])
    approach = f.box(300, 280, 320, 64, ["approach", "幼児へ近づく"])
    showing = f.box(40, 430, 250, 64, ["showing", "掲げて見せる"])
    placing = f.box(390, 430, 280, 64, ["placing", "到達圏に置く"])
    waiting = f.box(390, 600, 280, 64, ["waiting_infant", "拾うのを待つ"])
    reaction = f.box(390, 780, 280, 64, ["reaction", "すごい + 対象語"])
    done = f.box(430, 920, 200, 46, ["completed"], "ok")
    fail = f.box(740, 290, 190, 46, ["failure"], "bad")

    f.arrow([f.side(start, "s"), f.side(fetching, "n")])
    f.arrow([f.side(fetching, "s"), f.side(approach, "n")])
    f.label(480, 230, "対象を持った")
    f.arrow([f.side(fetching, "e"), (700, 172), (700, 313), f.side(fail, "w")])
    f.label(710, 230, "116 step", "lm")
    f.arrow([f.side(approach, "s", 0.72), (590, 400), f.side(placing, "n")])
    f.label(600, 370, "届いた・通常", "lm")
    f.arrow([f.side(approach, "s", 0.28), (370, 400), f.side(showing, "n")])
    f.label(230, 378, "届いた・escalated")
    f.arrow([f.side(approach, "e"), f.side(fail, "w")])
    f.label(700, 300, "52 step")
    f.arrow([f.side(showing, "e"), f.side(placing, "w")])
    f.label(320, 450, "24 step")
    f.arrow([f.side(showing, "s"), (165, 820), f.side(reaction, "w")])
    f.label(175, 640, "幼児が持った", "lm")
    f.arrow([f.side(showing, "w"), (48, 462), (48, 172), f.side(fetching, "w")])
    f.label(56, 300, "落とした", "lm")
    f.arrow([f.side(placing, "s"), f.side(waiting, "n")])
    f.label(550, 560, "8 step 後・圏内", "lm")
    f.arrow([f.side(placing, "e"), (760, 462), (760, 336), f.side(fail, "s")])
    f.label(770, 400, "圏外・84 step", "lm")
    f.loop_east(waiting, "RETRY")
    f.arrow([f.side(waiting, "w", 0.35), (16, 622), (16, 172), f.side(fetching, "w", 0.35)])
    f.label(24, 520, "ESCALATE・対象変更", "lm")
    f.arrow([f.side(waiting, "e"), (760, 632), (835, 632), f.side(fail, "s")])
    f.label(760, 610, "FAILED")
    f.arrow([f.side(waiting, "s"), f.side(reaction, "n")])
    f.label(550, 720, "幼児が持った")
    f.arrow([f.side(reaction, "s"), f.side(done, "n")])
    f.label(550, 870, "言い切った")
    f.save("object_giving.png")


def joint_attention():
    f = Fig(1100, 1020)
    f.legend(24, 16)
    start = f.box(470, 36, 140, 42, ["開始"], "ok")
    init = f.diamond(540, 140, 170, 84, ["init"])
    fetching = f.box(40, 250, 230, 60, ["fetching", "取りに行く"])
    approach = f.box(400, 250, 260, 60, ["approach"])
    looking = f.box(40, 470, 190, 52, ["looking"])
    pointing = f.box(300, 470, 190, 52, ["pointing"])
    shaking = f.box(560, 470, 190, 52, ["shaking"])
    post = f.box(250, 700, 330, 60, ["post_shared_attention", "締め"])
    done = f.box(315, 840, 200, 46, ["completed"], "ok")
    fail = f.box(860, 820, 180, 46, ["failure"], "bad")

    f.arrow([f.side(start, "s"), f.side(init, "n")])
    f.arrow([f.side(init, "w"), (155, 140), f.side(fetching, "n")])
    f.label(180, 118, "親始動・shake")
    f.arrow([f.side(init, "s"), f.side(approach, "n")])
    f.label(560, 210, "それ以外", "lm")
    f.arrow([f.side(fetching, "e"), f.side(approach, "w")])
    f.label(340, 268, "持った")
    f.arrow([f.side(fetching, "w"), (18, 280), (18, 960), (950, 960), f.side(fail, "s")])
    f.label(26, 560, "116 step", "lm")
    f.arrow([f.side(approach, "s", 0.15), (135, 400), f.side(looking, "n")])
    f.label(150, 400, "looking", "lm")
    f.arrow([f.side(approach, "s", 0.45), (395, 400), f.side(pointing, "n")])
    f.label(410, 400, "pointing", "lm")
    f.arrow([f.side(approach, "s", 0.85), (655, 400), f.side(shaking, "n")])
    f.label(670, 400, "shake", "lm")
    f.arrow([f.side(approach, "e"), (1070, 280), (1070, 960), f.side(fail, "s")])
    f.label(1040, 400, "52 step", "rm")

    f.arrow([f.side(looking, "e"), f.side(pointing, "w")])
    f.label(265, 484, "1回目")
    f.loop_north(pointing, "1回目")
    f.loop_north(shaking, "1回目")
    f.arrow([f.side(looking, "s"), (135, 640), f.side(post, "n", 0.15)])
    f.arrow([f.side(pointing, "s"), (395, 640), f.side(post, "n", 0.45)])
    f.label(160, 620, "注視成功", "lm")
    f.arrow([f.side(shaking, "s"), (655, 640), f.side(post, "n", 0.85)])
    # 落とした: shaking の上を右へ出て、approach の上を左へ
    f.arrow([f.side(shaking, "n", 0.8), (712, 220), (270, 220), f.side(fetching, "e", 0.3)])
    f.label(480, 204, "落とした")
    f.arrow([f.side(pointing, "e", 0.75), (520, 509), (520, 900), (860, 900), f.side(fail, "s")])
    f.label(620, 890, "2回目")
    f.arrow([f.side(shaking, "s", 0.8), (712, 900), f.side(fail, "s", 0.3)])
    f.arrow([f.side(post, "s"), f.side(done, "n")])
    f.label(430, 800, "待ち終了", "lm")
    f.save("joint_attention.png")


def attention():
    f = Fig(1040, 460)
    f.legend(24, 16)
    never = f.box(40, 90, 240, 70, ["まだ見ていない"])
    holding = f.box(400, 90, 250, 70, ["見ている", "hold を数える"])
    left = f.box(400, 280, 250, 64, ["目を逸らした"])
    success = f.box(800, 102, 180, 46, ["success"], "ok")
    timeout = f.box(800, 290, 180, 46, ["timeout"], "bad")
    f.arrow([f.side(never, "e"), f.side(holding, "w")])
    f.label(350, 112, "対象")
    f.loop_north(never, "対象以外")
    f.arrow([f.side(never, "s"), (160, 400), (890, 400), f.side(timeout, "s")])
    f.label(400, 384, "見ないまま 50 step")
    f.loop_north(holding, "30 未満・親の顔")
    f.arrow([f.side(holding, "e"), f.side(success, "w")])
    f.label(690, 112, "hold が 30")
    f.arrow([f.side(holding, "s", 0.7), f.side(left, "n", 0.7)])
    f.label(600, 220, "それ以外")
    f.arrow([f.side(left, "n", 0.25), (462, 230), (462, 160), f.side(holding, "s", 0.25)])
    f.label(360, 210, "対象・親の顔")
    f.loop_west(left, "50 未満")
    f.arrow([f.side(left, "e"), f.side(timeout, "w")])
    f.label(690, 300, "離脱 50 step")
    f.save("attention.png")


def manager_check():
    f = Fig(980, 640)
    f.legend(24, 16)
    tick = f.box(380, 52, 200, 48, ["10 step ごと"])
    blocked = f.diamond(480, 180, 230, 96, ["割り込みを止める？"])
    stay = f.box(40, 156, 240, 52, ["今のプロトコルのまま"])
    response = f.diamond(480, 340, 200, 90, ["訴えのくじ"])
    play = f.box(740, 250, 200, 64, ["二項・渡す", "共同注意"])
    clock = f.diamond(250, 500, 190, 90, ["家事の期限？"])
    house = f.box(40, 560, 160, 48, ["家事を始める"])
    init = f.diamond(620, 500, 200, 90, ["親始動のくじ"])

    f.arrow([f.side(tick, "s"), f.side(blocked, "n")])
    f.arrow([f.side(blocked, "w"), f.side(stay, "e")])
    f.label(300, 168, "止める")
    f.arrow([f.side(blocked, "s"), f.side(response, "n")])
    f.label(500, 260, "止めない", "lm")
    f.arrow([f.side(response, "e"), (700, 340), (840, 340), f.side(play, "s")])
    f.label(700, 320, "別の関わり")
    f.arrow([f.side(response, "w"), (200, 340), (160, 208)])
    f.label(180, 280, "維持・見守り以外", "lm")
    f.arrow([f.side(response, "s", 0.3), f.side(clock, "n")])
    f.label(300, 420, "維持・見守り")
    f.arrow([f.side(clock, "s"), f.side(house, "n")])
    f.label(160, 560, "期限")
    f.arrow([f.side(clock, "e"), f.side(init, "w")])
    f.label(430, 488, "まだ")
    f.arrow([f.side(init, "n"), (620, 282), f.side(play, "w")])
    f.label(680, 400, "遊びを引く", "lm")
    f.arrow([f.side(init, "s"), (620, 580), (200, 580), f.side(stay, "s")])
    f.label(400, 592, "待つ")
    f.save("manager_check.png")


def manager_finish():
    f = Fig(860, 280)
    f.legend(24, 16)
    fin = f.box(40, 110, 230, 52, ["プロトコルが終わった"])
    kind = f.diamond(420, 136, 180, 100, ["終わったのは"])
    obs = f.box(620, 60, 180, 48, ["見守り"], "ok")
    house = f.box(620, 170, 180, 48, ["家事"])
    f.arrow([f.side(fin, "e"), f.side(kind, "w")])
    f.arrow([f.side(kind, "n"), (420, 84), f.side(obs, "w")])
    f.label(520, 70, "家事、または期限前")
    f.arrow([f.side(kind, "s"), (420, 194), f.side(house, "w")])
    f.label(520, 206, "関わり・家事の期限")
    f.save("manager_finish.png")


def joint_attention_simple():
    """見る → 指差す、見続けたら締め、もう一度外れたら失敗。"""
    f = Fig(920, 560)
    f.legend(24, 16)
    start = f.box(340, 52, 140, 42, ["開始"], "ok")
    look = f.box(80, 160, 240, 64, ["見る"])
    point = f.box(500, 160, 240, 64, ["指差す"])
    close = f.box(80, 340, 240, 64, ["締め"])
    done = f.box(100, 470, 200, 46, ["成功"], "ok")
    fail = f.box(520, 360, 200, 46, ["失敗"], "bad")

    f.arrow([f.side(start, "s"), (410, 140), (200, 140), f.side(look, "n")])
    f.arrow([f.side(look, "e"), f.side(point, "w")])
    f.label(410, 176, "1回外れた")
    f.arrow([f.side(look, "s"), f.side(close, "n")])
    f.label(220, 270, "見続けた")
    f.label(220, 292, "親の顔は秒数を止めない")
    f.arrow([f.side(point, "s", 0.25), (560, 300), f.side(close, "e")])
    f.label(430, 300, "見続けた")
    f.arrow([f.side(point, "s", 0.75), f.side(fail, "n")])
    f.label(700, 300, "もう一度外れた")
    f.label(700, 322, "対象は変えない")
    f.arrow([f.side(close, "s"), f.side(done, "n")])
    f.save("joint_attention_simple.png")


def object_giving_simple():
    """取って届く場所に置く。拾わなければ梯子、拾えば成功。"""
    f = Fig(980, 640)
    f.legend(24, 16)
    start = f.box(250, 52, 140, 42, ["開始"], "ok")
    fetch = f.box(190, 130, 260, 56, ["対象を取る"])
    place = f.box(190, 230, 260, 64, ["届く場所に置く"])
    wait = f.box(190, 350, 260, 56, ["拾うのを待つ"])
    praise = f.box(190, 460, 260, 56, ["すごいと言う"])
    done = f.box(220, 560, 200, 46, ["成功"], "ok")
    ladder = f.box(620, 250, 280, 150, [
        "拾わないとき",
        "1. 言い直す",
        "2. 掲げて見せる",
        "3. 別の物",
        "4. 諦める",
    ], size=15)
    fail = f.box(660, 470, 200, 46, ["失敗"], "bad")

    f.arrow([f.side(start, "s"), f.side(fetch, "n")])
    f.arrow([f.side(fetch, "s"), f.side(place, "n")])
    f.arrow([f.side(place, "s"), f.side(wait, "n")])
    f.arrow([f.side(wait, "s"), f.side(praise, "n")])
    f.label(340, 430, "拾った")
    f.arrow([f.side(praise, "s"), f.side(done, "n")])
    f.arrow([f.side(wait, "e"), (560, 378), (560, 325), f.side(ladder, "w")])
    f.label(500, 360, "拾わない")
    f.arrow([f.side(ladder, "w", 0.28), (560, 292), (560, 262), f.side(place, "e")])
    f.label(500, 270, "1〜3 はもう一度")
    f.arrow([f.side(ladder, "s"), f.side(fail, "n")])
    f.label(780, 440, "4. 諦める")
    f.save("object_giving_simple.png")


def joint_attention_simple_en():
    f = Fig(980, 580)
    f.legend(24, 16, lang="en")
    start = f.box(130, 52, 140, 42, ["Start"], "ok")
    look = f.box(60, 160, 280, 64, ["Look"])
    point = f.box(560, 160, 280, 64, ["Point"])
    close = f.box(60, 360, 280, 64, ["Closing"])
    done = f.box(100, 490, 200, 46, ["Success"], "ok")
    fail = f.box(600, 380, 200, 46, ["Failure"], "bad")

    f.arrow([f.side(start, "s"), f.side(look, "n")])
    f.arrow([f.side(look, "e"), f.side(point, "w")])
    f.label(450, 176, "Looked away once")
    f.arrow([f.side(look, "s"), f.side(close, "n")])
    f.label(214, 270, "Kept looking", "lm")
    f.label(214, 292, "A look at the face pauses the count", "lm")
    f.arrow([f.side(point, "s", 0.2), (616, 320), f.side(close, "e")])
    f.label(500, 330, "Kept looking")
    f.arrow([f.side(point, "s", 0.75), f.side(fail, "n")])
    f.label(790, 300, "Looked away again", "lm")
    f.label(790, 322, "Same object", "lm")
    f.arrow([f.side(close, "s"), f.side(done, "n")])
    f.save("joint_attention_simple_en.png")


def object_giving_simple_en():
    f = Fig(1100, 660)
    f.legend(24, 16, lang="en")
    start = f.box(230, 52, 140, 42, ["Start"], "ok")
    fetch = f.box(140, 130, 320, 56, ["Take the object"])
    place = f.box(140, 230, 320, 64, ["Place it within reach"])
    wait = f.box(140, 350, 320, 56, ["Wait for a pickup"])
    praise = f.box(140, 470, 320, 56, ["Praise"])
    done = f.box(200, 570, 200, 46, ["Success"], "ok")
    ladder = f.box(680, 240, 360, 160, [
        "If it is not picked up",
        "1. Say it again",
        "2. Hold it up",
        "3. Another object",
        "4. Give up",
    ], size=15)
    fail = f.box(760, 480, 200, 46, ["Failure"], "bad")

    f.arrow([f.side(start, "s"), f.side(fetch, "n")])
    f.arrow([f.side(fetch, "s"), f.side(place, "n")])
    f.arrow([f.side(place, "s"), f.side(wait, "n")])
    f.arrow([f.side(wait, "s"), f.side(praise, "n")])
    f.label(320, 440, "Picked up")
    f.arrow([f.side(praise, "s"), f.side(done, "n")])
    f.arrow([f.side(wait, "e"), (620, 378), (620, 320), f.side(ladder, "w")])
    f.label(560, 360, "Not picked up")
    f.arrow([f.side(ladder, "w", 0.25), (620, 280), (620, 262), f.side(place, "e")])
    f.label(560, 248, "1–3: place again")
    f.arrow([f.side(ladder, "s"), f.side(fail, "n")])
    f.label(880, 450, "4. Give up")
    f.save("object_giving_simple_en.png")


def joint_attention_speech_en():
    """親が合図の最初に言うことと、幼児の単語への一度きりの返事。"""
    f = Fig(1180, 860)
    f.legend(24, 16, lang="en")
    head = f.font(15, bold=True)

    f.d.text((40, 48), "When the cue starts", font=head, fill=INK)
    purpose = f.diamond(590, 155, 200, 90, ["Purpose"])
    teach = f.box(40, 280, 280, 58, ["This + the name"])
    share = f.box(450, 280, 280, 58, ["Great"])
    quiet = f.box(860, 280, 280, 58, ["No words"])
    f.arrow([f.side(purpose, "w"), (180, 155), f.side(teach, "n")])
    f.label(300, 140, "To name it")
    f.arrow([f.side(purpose, "s"), f.side(share, "n")])
    f.label(640, 230, "To share", "lm")
    f.arrow([f.side(purpose, "e"), (1000, 155), f.side(quiet, "n")])
    f.label(760, 140, "Just to look")

    f.d.text((40, 380), "If the infant says a word", font=head, fill=INK)
    heard = f.diamond(200, 500, 230, 96, ["New word?"])
    none = f.box(70, 660, 260, 52, ["No reply"])
    second = f.box(40, 760, 280, 52, ["No second reply"])
    which = f.diamond(760, 560, 210, 96, ["How close?"])
    match = f.box(400, 720, 220, 58, ["Great + the name"])
    near = f.box(645, 720, 230, 58, ["Yes + the name"])
    other = f.box(930, 720, 230, 58, ["Not that + the name"])

    f.arrow([f.side(heard, "s"), f.side(none, "n")])
    f.label(214, 618, "None, or the same", "lm")
    f.arrow([f.side(heard, "w"), (24, 500), (24, 786), f.side(second, "w")])
    f.label(36, 548, "Already answered", "lm")
    f.arrow([f.side(heard, "e"), (760, 500), f.side(which, "n")])
    f.label(470, 484, "First time")
    f.arrow([f.side(which, "w"), (510, 560), f.side(match, "n")])
    f.label(500, 544, "The name")
    f.arrow([f.side(which, "s"), f.side(near, "n")])
    f.label(900, 660, "Close", "lm")
    f.arrow([f.side(which, "e"), (1045, 560), f.side(other, "n")])
    f.label(1032, 640, "Other", "rm")

    note = f.font(14)
    f.d.text(
        (360, 810),
        "One reply only. None of these decide success. A steady look does.",
        font=note,
        fill="#3d4f63",
    )
    f.save("joint_attention_speech_en.png")


if __name__ == "__main__":
    dyadic()
    observance()
    prerelease()
    household()
    object_giving()
    joint_attention()
    attention()
    manager_check()
    manager_finish()
    joint_attention_simple()
    object_giving_simple()
    joint_attention_simple_en()
    object_giving_simple_en()
    joint_attention_speech_en()
