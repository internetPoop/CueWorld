#!/usr/bin/env python3
"""Render the five isolation-test backup slides (16:9).

Wording follows cueworld/docs/next.md §10 and the isolation agreements:
ja_looking_isolation.md, giving_isolation.md, dyadic_raise_isolation.md,
voice_isolation.md, oracle_isolation.md.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
W, H = 1920, 1080

INK = (28, 28, 28)
MUTED = (78, 78, 78)
FAINT = (128, 128, 128)
LINE = (214, 214, 214)
PAPER = (255, 255, 255)
TEAL = (22, 112, 108)
TEAL_SOFT = (226, 241, 239)
CORAL = (196, 84, 62)
CORAL_SOFT = (250, 234, 228)
CUBE = (214, 78, 58)

FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FONT_B = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

F_KICK = ImageFont.truetype(FONT_B, 18)
F_NAME = ImageFont.truetype(FONT, 20)
F_TITLE = ImageFont.truetype(FONT_B, 44)
F_LEAD = ImageFont.truetype(FONT, 24)
F_H = ImageFont.truetype(FONT_B, 18)
F_BODY = ImageFont.truetype(FONT, 22)
F_BODY_B = ImageFont.truetype(FONT_B, 22)
F_SMALL = ImageFont.truetype(FONT, 18)
F_SMALL_B = ImageFont.truetype(FONT_B, 18)
F_FORM = ImageFont.truetype(FONT_B, 30)
F_FOOT = ImageFont.truetype(FONT, 20)
F_Q = ImageFont.truetype(FONT_B, 48)
F_XL = ImageFont.truetype(FONT_B, 40)
F_L = ImageFont.truetype(FONT, 32)
F_LB = ImageFont.truetype(FONT_B, 32)
F_MD = ImageFont.truetype(FONT, 28)

MX = 72
COL_TOP = 292
FOOT_Y = 1008
GAP = 48


def text_w(text: str, font) -> float:
    return font.getlength(text)


def wrap(text: str, font, max_w: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = word if not cur else f"{cur} {word}"
        if text_w(trial, font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_text(draw, xy, string, font, fill, clip=None) -> tuple[int, int, int, int]:
    draw.text(xy, string, font=font, fill=fill)
    box = draw.textbbox(xy, string, font=font)
    if clip is not None:
        x0, y0, x1, y1 = clip
        if box[0] < x0 - 1 or box[1] < y0 - 1 or box[2] > x1 + 1 or box[3] > y1 + 1:
            print(f"OVERFLOW {string!r} bbox={box} clip={clip}")
    return box


def paragraph(draw, text, font, fill, x, y, max_w, leading, clip=None) -> int:
    for line in wrap(text, font, max_w):
        draw_text(draw, (x, y), line, font, fill, clip)
        y += leading
    return y


def arrow(draw, start, end, fill, width=4, head=12):
    draw.line([start, end], fill=fill, width=width)
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    p1 = (end[0] + head * math.cos(ang + math.radians(152)),
          end[1] + head * math.sin(ang + math.radians(152)))
    p2 = (end[0] + head * math.cos(ang - math.radians(152)),
          end[1] + head * math.sin(ang - math.radians(152)))
    draw.polygon([end, p1, p2], fill=fill)


def dashed(draw, start, end, fill, width=2, dash=8, gap=6):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy) or 1
    ux, uy = dx / length, dy / length
    t = 0.0
    while t < length:
        t2 = min(t + dash, length)
        draw.line(
            [(start[0] + ux * t, start[1] + uy * t),
             (start[0] + ux * t2, start[1] + uy * t2)],
            fill=fill,
            width=width,
        )
        t += dash + gap


def cube(draw, x, y, size=36):
    draw.rounded_rectangle([x, y, x + size, y + size], radius=3, fill=CUBE)


def chip(draw, x, y, label, fill, ink) -> int:
    pad_x, pad_y = 14, 8
    tw = text_w(label, F_SMALL_B)
    w, h = tw + pad_x * 2, 34
    draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=fill)
    draw_text(draw, (x + pad_x, y + pad_y - 1), label, F_SMALL_B, ink)
    return int(w)


def header(draw, index: str, name: str, title: list[str], lead: str):
    draw_text(draw, (MX, 40), "BACKUP", F_KICK, TEAL)
    right = f"{index}    {name.upper()}"
    draw_text(draw, (W - MX - text_w(right, F_NAME), 38), right, F_NAME, FAINT)
    y = 86
    for line in title:
        draw_text(draw, (MX, y), line, F_TITLE, INK, (MX, 70, W - MX, 230))
        y += 54
    y += 6
    y = paragraph(draw, lead, F_LEAD, MUTED, MX, y, W - 2 * MX, 32, (MX, y, W - MX, COL_TOP - 8))
    rule_y = COL_TOP - 28
    draw.line([(MX, rule_y), (W - MX, rule_y)], fill=LINE, width=2)
    return rule_y


def footer(draw, line: str):
    draw.line([(MX, FOOT_Y), (W - MX, FOOT_Y)], fill=LINE, width=2)
    paragraph(draw, line, F_FOOT, MUTED, MX, FOOT_Y + 16, W - 2 * MX, 28, (MX, FOOT_Y, W - MX, H - 16))


def columns():
    inner = W - 2 * MX
    width = (inner - 2 * GAP) / 3
    xs = [MX, MX + width + GAP, MX + 2 * (width + GAP)]
    return xs, width


def col_heads(draw, titles):
    xs, width = columns()
    y = COL_TOP
    for x, title in zip(xs, titles):
        draw_text(draw, (x, y), title, F_H, INK)
    for i in range(1, 3):
        x = xs[i] - GAP / 2
        draw.line([(x, COL_TOP), (x, FOOT_Y - 28)], fill=LINE, width=2)
    return xs, width, y + 40


WHO = {
    "parent": (TEAL, TEAL_SOFT, "Parent"),
    "infant": (CORAL, CORAL_SOFT, "Infant"),
    "end": (INK, (238, 238, 238), "End"),
}


def timeline(draw, x, y, width, steps, clip):
    body_x = x + 118
    max_w = width - 118
    gap = 22
    blocks = []
    for who, sentence in steps:
        lines = wrap(sentence, F_BODY, max_w)
        blocks.append((who, lines, 8 + len(lines) * 30))
    cx = x + 18
    top_dot = None
    bottom_dot = None
    laid = []
    cursor = y
    for who, lines, height in blocks:
        dot_y = cursor + 14
        laid.append((who, lines, cursor, dot_y))
        if top_dot is None:
            top_dot = dot_y
        bottom_dot = dot_y
        cursor += height + gap
    if top_dot is not None:
        draw.line([(cx, top_dot), (cx, bottom_dot)], fill=LINE, width=3)
    for who, lines, cursor, dot_y in laid:
        color, soft, label = WHO[who]
        draw.ellipse([cx - 9, dot_y - 9, cx + 9, dot_y + 9], fill=color)
        tag_w = text_w(label, F_SMALL_B)
        draw_text(draw, (x + 40, cursor + 2), label, F_SMALL_B, color, clip)
        ty = cursor + 2
        for line in lines:
            draw_text(draw, (body_x, ty), line, F_BODY, INK, clip)
            ty += 30
        _ = tag_w


def spec(draw, x, y, width, rows, clip):
    cursor = y + 8
    max_w = width
    for label, sentence in rows:
        draw_text(draw, (x, cursor), label, F_SMALL_B, TEAL, clip)
        cursor += 28
        cursor = paragraph(draw, sentence, F_BODY, INK, x, cursor, max_w, 30, clip)
        cursor += 26


def _screen_deg(p0, p1) -> float:
    return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) % 360


def _short_mid(a: float, b: float) -> float:
    delta = (b - a + 180) % 360 - 180
    return (a + delta / 2) % 360


def draw_giving_score(draw, x, y, width):
    formula = "e  =  d  ×  (1 + (1 − cos θ))"
    draw_text(draw, (x, y), formula, F_FORM, INK, (x, y, x + width, y + 48))
    base = y + 58
    palm = (x + 8, base + 28)
    draw.rounded_rectangle(
        [palm[0], palm[1], palm[0] + 78, palm[1] + 52],
        radius=26, fill=CORAL_SOFT, outline=CORAL, width=3,
    )
    draw_text(draw, (palm[0] + 16, palm[1] + 14), "palm", F_SMALL_B, CORAL)
    origin = (palm[0] + 74, palm[1] + 26)
    cube_x = x + width - 78
    cube(draw, cube_x, base + 22, 44)
    obj_c = (cube_x - 2, base + 44)
    dashed(draw, origin, obj_c, TEAL, width=3)
    face = (origin[0] + 64 * math.cos(math.radians(42)),
            origin[1] + 64 * math.sin(math.radians(42)))
    arrow(draw, origin, face, CORAL, width=4, head=11)
    a_obj = _screen_deg(origin, obj_c)
    a_face = _screen_deg(origin, face)
    radius = 36
    draw.arc(
        [origin[0] - radius, origin[1] - radius, origin[0] + radius, origin[1] + radius],
        start=a_obj, end=a_face, fill=TEAL, width=3,
    )
    mid = _short_mid(a_obj, a_face)
    draw_text(
        draw,
        (origin[0] + (radius + 8) * math.cos(math.radians(mid)) - 6,
         origin[1] + (radius + 8) * math.sin(math.radians(mid)) - 12),
        "θ", F_BODY_B, TEAL,
    )
    draw_text(draw, (origin[0] + 96, origin[1] - 28), "d", F_BODY_B, TEAL)
    c1 = chip(draw, x, base + 128, "Facing it:  e = d", TEAL_SOFT, TEAL)
    chip(draw, x + c1 + 12, base + 128, "Facing away:  e = 3d", CORAL_SOFT, CORAL)
    return base + 178


def draw_raise_score(draw, x, y, width):
    formula = "(h before  −  h now)  /  0.4 m"
    draw_text(draw, (x, y), formula, F_FORM, INK, (x, y, x + width, y + 46))
    left = x + 36
    top = y + 62
    bottom = y + 230
    band_y = y + 168
    draw.rectangle([left, top, left + 86, band_y], fill=TEAL_SOFT)
    draw.line([(left - 16, band_y), (left + 110, band_y)], fill=TEAL, width=3)
    draw.ellipse([left + 28, band_y + 40, left + 46, band_y + 58], fill=CORAL)
    arrow(draw, (left + 37, band_y + 40), (left + 37, band_y + 8), CORAL, width=3, head=10)
    draw_text(draw, (left + 130, top + 18), "In the band. No ceiling.", F_BODY, INK)
    draw_text(draw, (left + 130, band_y - 16), "Eye height − 0.4 m", F_SMALL_B, TEAL)
    draw_text(draw, (left + 130, band_y + 28), "gap h", F_BODY_B, CORAL)
    return bottom


def behavior_slide(spec_rows, steps, score_fn, score_lines, meta, footer_line):
    img = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(img)
    header(draw, meta["index"], meta["name"], meta["title"], meta["lead"])
    xs, width, y = col_heads(draw, ["THE MOVE", "THE SCORE", "THE INFANT"])
    clips = [(xs[i], COL_TOP, xs[i] + width, FOOT_Y - 8) for i in range(3)]
    timeline(draw, xs[0], y, width, steps, clips[0])
    diagram_bottom = score_fn(draw, xs[1], y, width)
    cursor = diagram_bottom + 18
    for line in score_lines:
        cursor = paragraph(draw, line, F_BODY, INK, xs[1], cursor, width, 30, clips[1])
        cursor += 12
    spec(draw, xs[2], y, width, spec_rows, clips[2])
    footer(draw, footer_line)
    img.save(OUT / meta["file"], "PNG")
    print("wrote", meta["file"])


def _kicker(draw, index: str, name: str):
    draw_text(draw, (MX, 28), "BACKUP", F_LB, TEAL)
    right = f"{index}    {name.upper()}"
    draw_text(draw, (W - MX - text_w(right, F_MD), 32), right, F_MD, FAINT)


def _cx(draw, cx, y, string, font, fill):
    draw_text(draw, (cx - text_w(string, font) / 2, y), string, font, fill)


def _fact_row(draw, y, height, items):
    gap = 24
    width = (W - 2 * MX - gap * (len(items) - 1)) / len(items)
    for i, (label, value) in enumerate(items):
        x = MX + i * (width + gap)
        draw.rounded_rectangle([x, y, x + width, y + height], radius=18, fill=(244, 247, 247))
        draw_text(draw, (x + 28, y + 22), label, F_MD, TEAL)
        draw_text(draw, (x + 28, y + 68), value, F_XL, INK, (x + 16, y, x + width - 16, y + height))


def looking_cue_slide():
    """Parent cue selects one object among several. Gaze toward that object scores."""
    img = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(img)
    _kicker(draw, "01", "Looking together")
    draw_text(draw, (MX, 84), "After the parent cues an object,", F_Q, INK)
    draw_text(draw, (MX, 148), "looking at that object is rewarded.", F_Q, INK)

    gray = (186, 186, 186)
    size = 140
    col = 1180
    cubes = [(col, 150, gray), (col, 390, CUBE), (col, 680, gray)]
    for x, y, fill in cubes:
        if fill == CUBE:
            draw.rounded_rectangle(
                [x - 12, y - 12, x + size + 12, y + size + 12],
                radius=16, outline=TEAL, width=6,
            )
        draw.rounded_rectangle([x, y, x + size, y + size], radius=8, fill=fill)
    draw_text(draw, (col + size + 28, 430), "cued", F_XL, TEAL)

    parent = (300, 460)
    infant = (300, 680)
    dashed(draw, parent, (col - 6, 460), TEAL, width=6, dash=16, gap=12)
    arrow(draw, infant, (col - 6, 500), CORAL, width=8, head=22)
    draw.ellipse([parent[0] - 20, parent[1] - 20, parent[0] + 20, parent[1] + 20], fill=TEAL)
    draw.ellipse([infant[0] - 20, infant[1] - 20, infant[0] + 20, infant[1] + 20], fill=CORAL)
    draw_text(draw, (parent[0] - 150, parent[1] - 22), "Parent", F_LB, TEAL)
    draw_text(draw, (infant[0] - 140, infant[1] - 22), "Infant", F_LB, CORAL)
    _cx(draw, 720, 400, "cue", F_XL, TEAL)
    _cx(draw, 720, 620, "gaze", F_XL, CORAL)

    _fact_row(draw, 890, 140, [
        ("REWARD FROM", "the cue"),
        ("SCORES", "the cued object"),
        ("THE HOLD", "+2"),
    ])
    img.save(OUT / "slide_backup_looking_cue.png", "PNG")
    print("wrote slide_backup_looking_cue.png")


def looking_slide():
    img = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(img)
    _kicker(draw, "01", "Looking together")
    draw_text(draw, (MX, 84), "Can the infant hold its gaze on the cued object", F_Q, INK)
    draw_text(draw, (MX, 148), "for about 3 seconds?", F_Q, INK)

    eye = (300, 390)
    cube_xy = (1500, 230)
    cube_size = 120
    cube(draw, cube_xy[0], cube_xy[1], cube_size)
    target = (cube_xy[0] - 4, cube_xy[1] + cube_size / 2)
    gaze = (720, 520)
    dashed(draw, eye, target, TEAL, width=5, dash=16, gap=12)
    arrow(draw, eye, gaze, CORAL, width=8, head=22)
    a_obj = _screen_deg(eye, target)
    a_gaze = _screen_deg(eye, gaze)
    radius = 120
    draw.arc(
        [eye[0] - radius, eye[1] - radius, eye[0] + radius, eye[1] + radius],
        start=a_obj, end=a_gaze, fill=TEAL, width=6,
    )
    mid = _short_mid(a_obj, a_gaze)
    label_r = radius + 28
    _cx(
        draw,
        eye[0] + label_r * math.cos(math.radians(mid)),
        eye[1] + label_r * math.sin(math.radians(mid)) - 20,
        "θ", F_XL, TEAL,
    )
    draw.ellipse([eye[0] - 18, eye[1] - 18, eye[0] + 18, eye[1] + 18], fill=CORAL)
    draw_text(draw, (eye[0] - 110, eye[1] - 18), "eyes", F_LB, CORAL)
    draw_text(draw, (cube_xy[0] + 8, cube_xy[1] + cube_size + 12), "object", F_LB, CUBE)
    draw_text(draw, (gaze[0] - 16, gaze[1] + 16), "gaze", F_LB, CORAL)
    mx = (eye[0] + target[0]) / 2
    my = (eye[1] + target[1]) / 2
    dx, dy = target[0] - eye[0], target[1] - eye[1]
    length = math.hypot(dx, dy)
    ox, oy = -dy / length, dx / length
    if oy > 0:
        ox, oy = -ox, -oy
    _cx(draw, mx + ox * 42, my + oy * 42 - 16, "cue", F_LB, TEAL)

    _cx(draw, W / 2, 600, "(θ before  −  θ now)  /  180°", F_XL, INK)
    bar_y = 730
    draw.rounded_rectangle([260, bar_y, 1660, bar_y + 18], radius=9, fill=TEAL_SOFT)
    draw_text(draw, (260, bar_y - 58), "1", F_XL, INK)
    draw_text(draw, (320, bar_y - 50), "behind", F_LB, MUTED)
    zero = "0"
    draw_text(draw, (1660 - text_w(zero, F_XL), bar_y - 58), zero, F_XL, INK)
    on_obj = "on the object"
    draw_text(draw, (1660 - text_w(on_obj, F_LB) - 56, bar_y - 50), on_obj, F_LB, MUTED)
    _cx(draw, W / 2, bar_y + 36, "closing  →", F_LB, CORAL)

    _fact_row(draw, 880, 150, [
        ("DOES", "torso and head"),
        ("SEES", "the camera"),
        ("AT THE HOLD", "+2"),
        ("IF THE CUE CHANGES", "angle stays"),
    ])
    img.save(OUT / "slide_backup_looking.png", "PNG")
    print("wrote slide_backup_looking.png")


def voice_slide():
    img = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(img)
    _kicker(draw, "04", "Voice")
    draw_text(draw, (MX, 84), "Can the infant repeat one word when", F_Q, INK)
    draw_text(draw, (MX, 148), "the recognition cost is the reward?", F_Q, INK)

    draw_text(draw, (MX, 230), "Parent", F_LB, TEAL)
    box_w, box_h, gap = 168, 100, 18
    x = MX
    y = 278
    for _ in range(5):
        draw.rounded_rectangle([x, y, x + box_w, y + box_h], radius=16, fill=TEAL_SOFT)
        _cx(draw, x + box_w / 2, y + 30, "name", F_XL, TEAL)
        x += box_w + gap
    draw_text(draw, (MX, y + box_h + 12), "one name, five times", F_L, TEAL)

    voice_x = 1280
    draw_text(draw, (voice_x, 230), "Infant", F_LB, CORAL)
    draw.rounded_rectangle([voice_x, y, voice_x + 280, y + box_h], radius=16, fill=CORAL_SOFT, outline=CORAL, width=4)
    _cx(draw, voice_x + 140, y + 30, "voice", F_XL, CORAL)
    draw_text(draw, (voice_x, y + box_h + 12), "no reply", F_L, CORAL)
    arrow(draw, (x - gap + 8, y + box_h / 2), (voice_x - 24, y + box_h / 2), INK, width=6, head=18)

    _cx(draw, W / 2, 500, "1  −  (cost  −  c*)", F_Q, INK)

    cards = [
        (MX, "NO HINT", "0", "raw cost"),
        (W / 2 + 20, "HINT", "−1", "after the hint"),
    ]
    for cx0, title, value, note in cards:
        draw.rounded_rectangle([cx0, 590, cx0 + 860, 860], radius=20, fill=(244, 247, 247))
        draw_text(draw, (cx0 + 40, 612), title, F_LB, TEAL)
        draw_text(draw, (cx0 + 40, 662), "c*", F_L, MUTED)
        draw_text(draw, (cx0 + 110, 704), value, F_Q, INK)
        draw_text(draw, (cx0 + 40, 792), note, F_L, MUTED)
        draw_text(draw, (cx0 + 480, 784), "match  →  +1", F_XL, TEAL)

    draw.rounded_rectangle([MX, 900, W - MX, 1000], radius=16, fill=TEAL_SOFT)
    _cx(draw, W / 2, 932, "No decision scores 0.    The learner still sees the parent’s action.", F_L, TEAL)
    img.save(OUT / "slide_backup_voice.png", "PNG")
    print("wrote slide_backup_voice.png")


def oracle_slide():
    img = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(img)
    _kicker(draw, "05", "Oracle observation")
    draw_text(draw, (MX, 84), "Can numbers stand in for the camera?", F_Q, INK)

    frame = (180, 210, 1080, 720)
    draw.rounded_rectangle(frame, radius=8, outline=INK, width=4)
    _cx(draw, (frame[0] + frame[2]) / 2, frame[1] - 52, "+1", F_XL, INK)
    _cx(draw, (frame[0] + frame[2]) / 2, frame[3] + 10, "−1", F_XL, INK)
    draw_text(draw, (frame[0] - 78, (frame[1] + frame[3]) / 2 - 24), "−1", F_XL, INK)
    draw_text(draw, (frame[2] + 18, (frame[1] + frame[3]) / 2 - 24), "+1", F_XL, INK)

    cube(draw, 420, 360, 110)
    draw_text(draw, (420, 482), "object", F_LB, CUBE)
    eye = (760, 560)
    draw.ellipse([eye[0] - 20, eye[1] - 20, eye[0] + 20, eye[1] + 20], fill=TEAL)
    draw_text(draw, (eye[0] + 28, eye[1] - 18), "eye", F_LB, TEAL)
    finger = (860, 660)
    draw.rounded_rectangle([finger[0], finger[1], finger[0] + 28, finger[1] + 44], radius=8, fill=TEAL)
    draw_text(draw, (finger[0] + 40, finger[1] + 6), "finger", F_LB, TEAL)
    arrow(draw, eye, (560, 430), TEAL, width=6, head=16)
    draw_text(draw, (600, 470), "look", F_LB, TEAL)

    draw.rounded_rectangle([180, 820, 260, 900], radius=6, fill=(190, 190, 190))
    draw_text(draw, (280, 840), "(−2, −2)    unseen", F_XL, MUTED)

    rx = 1220
    blocks = [
        (250, "OBJECTS", "x, y, distance", "color, shape"),
        (460, "PARENT", "eye", "fingertip"),
        (670, "CUE", "look, point, show", "raised within 0.12 m"),
    ]
    for by, title, a, b in blocks:
        draw_text(draw, (rx, by), title, F_LB, TEAL)
        draw_text(draw, (rx, by + 48), a, F_XL, INK)
        draw_text(draw, (rx, by + 104), b, F_XL, INK)

    draw.rounded_rectangle([MX, 960, W - MX, 1040], radius=16, fill=TEAL_SOFT)
    _cx(draw, W / 2, 984, "Replaces the image.    Not in the normal run.", F_L, TEAL)
    img.save(OUT / "slide_backup_oracle.png", "PNG")
    print("wrote slide_backup_oracle.png")


def main():
    looking_cue_slide()
    looking_slide()
    behavior_slide(
        spec_rows=[
            ("DOES", "Torso, head tilt, shoulder, elbow, wrist, and the fingers."),
            ("SEES", "The camera image, joint angles, and touch."),
            ("REWARD", "The dense score alone. Preference points stay at zero."),
            ("STILL OPEN", "Which room layout to train first."),
        ],
        steps=[
            ("parent", "Places one object within reach. Says nothing."),
            ("infant", "Reaches, and closes the hand on it."),
            ("end", "The episode ends when that object is held. Gaze is not required."),
        ],
        score_fn=draw_giving_score,
        score_lines=[
            "d is the palm-to-object distance. θ is how far the palm faces from it.",
            "Each step scores how much e fell on that step. 1 m counts as 1.",
            "+2 once, at the grasp. Nothing before the wait, and nothing after.",
        ],
        meta={
            "file": "slide_backup_giving.png",
            "index": "02",
            "name": "Giving",
            "title": ["Can the infant reach a placed object and pick it up?"],
            "lead": "Giving, stopped at the grasp. No lift, no second object, and no parent speech.",
        },
        footer_line="Two waits of about 3.4 seconds, then a miss. The object stays where it was placed. This test has not been trained yet.",
    )
    behavior_slide(
        spec_rows=[
            ("DOES", "Torso, shoulder pitch, and elbow."),
            ("SEES", "The camera image, and those joint angles."),
            ("REWARD", "The dense score alone. Preference points stay at zero."),
            ("STILL OPEN", "Which room layout to train first. There is no object, so a new target does not move the room."),
        ],
        steps=[
            ("parent", "Raises one arm and waves it."),
            ("infant", "Lifts its own hand into the eye-level band."),
            ("infant", "The hand is in the band on 8 of the last 10 steps."),
            ("end", "The episode ends. If the demonstration ends first, it is a miss."),
        ],
        score_fn=draw_raise_score,
        score_lines=[
            "h is how far the hand sits under the band. Inside the band, h is 0.",
            "Each step scores how much h closed on that step. 0.4 m counts as 1.",
            "Higher, once inside, adds nothing. +2 once, when the rate first reaches 0.8.",
        ],
        meta={
            "file": "slide_backup_raise.png",
            "index": "03",
            "name": "Turn-taking, raise",
            "title": ["Can the infant hold a hand in the eye-level band", "while the parent demonstrates?"],
            "lead": "Turn-taking, raise only. An open hand and the voice do not count.",
        },
        footer_line="The parent has up to 20 steps to arrive, then 90 steps of the raise. This test has not been trained yet.",
    )
    voice_slide()
    oracle_slide()


if __name__ == "__main__":
    main()
