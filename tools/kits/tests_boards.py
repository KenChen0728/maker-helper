"""雷切測試板：軸孔測試尺、接頭測試板（3 mm、5 mm）。"""
import math, os, numpy as np
from kitlib import *
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)

# ---------------- 軸孔測試尺 ----------------
def axle_ruler():
    rows = [("竹籤", 3.0), ("木棒", 4.0), ("木棒", 5.0), ("木棒", 6.0)]
    deltas = [-.2, -.1, 0, .1, .2, .3, .4, .5, .6]
    sp, x0, y0, rh = 12.0, 24.0, 14.0, 15.0
    W = x0 + len(deltas) * sp + 4; H = y0 + len(rows) * rh + 2
    sh = Sheet(W + 8, H + 8, "軸孔測試尺")
    ox, oy = 4, 4
    holes = []; eng = []
    eng += text("軸孔測試尺　轉得順＝輪軸孔　卡緊不轉＝輪子孔", 3.0, W / 2, 7.5)
    for j, (nm, d) in enumerate(rows):
        cy = y0 + j * rh + 4.5
        eng += text(f"{nm}", 2.6, 11, cy - .4); eng += text(f"{d:g}", 2.6, 11, cy + 3.4)
        for i, dl in enumerate(deltas):
            cx = x0 + i * sp + sp / 2
            holes.append(circle(cx, cy, (d + dl) / 2))
            lab = ("0" if dl == 0 else (f"+{dl:.1f}" if dl > 0 else f"{dl:.1f}")).replace("0.", ".")
            eng += text(lab, 2.1, cx, cy + 3.3 + d / 2 + .6, bold=False)
    sh.part("軸孔測試尺", rrect(0, 0, W, H, 3), holes, eng, dx=ox, dy=oy)
    return sh

# ---------------- 接頭測試板 ----------------
def tab_loop(kind, t, v):
    """直立的小片（寬 20、身體高 16），凸榫朝下。回傳（外框, 孔, 雕刻字, 插槽長, 底板要不要特別的槽長）"""
    B, Hb = 20.0, (22.0 if kind == "snap" else 16.0); cx = B / 2; holes = []; lab = ""
    body_l = [(0, 0), (B, 0), (B, Hb)]; body_r = [(0, Hb)]
    if kind == "taper":                 # 梯形榫：前端窄好插，根部寬卡緊
        root, tip = 12 + v, 11.6
        tab = [(cx + root / 2, Hb), (cx + tip / 2, Hb + t - .3), (cx + tip / 2 - .3, Hb + t), (cx - tip / 2 + .3, Hb + t), (cx - tip / 2, Hb + t - .3), (cx - root / 2, Hb)]
        lab = f"梯{v:.1f}".replace("0.", "."); slot = 12.0
    elif kind == "bump":                # 榫邊凸點：兩側各一個半圓凸點
        Rb = .8; yc = Hb + t / 2; hw = 6.0
        dy = math.sqrt(Rb ** 2 - (Rb - v) ** 2)
        right = [(cx + hw, Hb)]
        for a in np.linspace(-1, 1, 15):
            y = yc + a * dy; right.append((cx + hw - (Rb - v) + math.sqrt(max(Rb ** 2 - (y - yc) ** 2, 0)), y))
        right += [(cx + hw, Hb + t - .3), (cx + hw - .3, Hb + t)]
        left = [(2 * cx - x, y) for (x, y) in right[::-1]]
        tab = right + left
        lab = f"點{v:.1f}".replace("0.", "."); slot = 12.0
    elif kind == "pin":                 # 竹籤插銷：凸榫穿出去，插一根竹籤
        e = 7.0; hw = 6.0
        tab = [(cx + hw, Hb), (cx + hw, Hb + t + e - .5), (cx + hw - .5, Hb + t + e), (cx - hw + .5, Hb + t + e), (cx - hw, Hb + t + e - .5), (cx - hw, Hb)]
        r = 3.3 / 2; holes.append(circle(cx, Hb + (t - v) + r, r))
        lab = f"銷{v:.1f}".replace("0.", ".").replace("銷.0", "銷0"); slot = 12.4
    elif kind == "wedge":               # 楔子：凸榫穿出去，方孔插一片斜的楔子
        e = 9.0; hw = 6.0
        tab = [(cx + hw, Hb), (cx + hw, Hb + t + e - .5), (cx + hw - .5, Hb + t + e), (cx - hw + .5, Hb + t + e), (cx - hw, Hb + t + e - .5), (cx - hw, Hb)]
        hx = (t + .4) / 2; holes.append(rect(cx - hx, Hb + (t - v), 2 * hx, 5.0))
        lab = f"楔{v:.1f}".replace("0.", "."); slot = 12.4
    elif kind == "snap":                # 懸臂卡扣：兩根細臂，尖端有倒鉤
        w, g, b, L_bar = v, 1.6, .8, 3.0
        xo = g / 2 + w; yc = Hb + t + .2; ytip = yc + L_bar; slit_top = Hb - 10
        rs = g / 2
        right = [(cx + xo, Hb), (cx + xo, yc), (cx + xo + b, yc), (cx + xo, ytip), (cx + g / 2, ytip)]
        arc = [(cx + rs * math.cos(a), slit_top + rs * math.sin(a)) for a in np.linspace(0, -math.pi, 12)]
        right += [(cx + g / 2, slit_top)] + arc[1:-1] + [(cx - g / 2, slit_top)]
        left = [(cx - g / 2, ytip), (cx - xo, ytip), (cx - xo - b, yc), (cx - xo, yc), (cx - xo, Hb)]
        tab = right + left
        lab = f"扣{v:.1f}"; slot = 2 * w + g + .3
    outer = np.array(body_l + tab + body_r, float)
    eng = text(lab, 3.4, cx, 8.6 if kind != "snap" else 7.6)
    return outer, holes, eng, slot, lab

def joint_board(t):
    sh = Sheet(300, 210, f"接頭測試板 {t:g} mm")
    groups = [("taper", [.2, .3, .4, .5]), ("bump", [.3, .4, .5]), ("pin", [0, .2, .4]), ("wedge", [.3, .6]), ("snap", [1.6, 2.0, 2.4])]
    items = [(k, v) for k, vs in groups for v in vs]
    # 底板：兩條，每條放 7～8 個槽
    rowsA = items[:7]; rowsB = items[7:]
    def base(items_, y, title, note):
        n = len(items_); pitch = 21.0; W = n * pitch + 8; Hh = 34.0
        holes = []; eng = []
        for i, (k, v) in enumerate(items_):
            o, hs, e, slot, lab = tab_loop(k, t, v)
            cx = 4 + i * pitch + pitch / 2; sy = 9.0
            holes.append(rect(cx - slot / 2, sy, slot, t + .3))
            eng += text(lab, 2.6, cx, sy + t + .3 + 5.0)
        eng += text(title, 2.8, 4, 6.2, anchor="start"); eng += text(note, 2.2, W - 4, Hh - 3.2, anchor="end", bold=False)
        sh.part(title, rrect(0, 0, W, Hh, 2), holes, eng, dx=6, dy=y)
        return W
    base(rowsA, 4, f"底板 A（{t:g} mm）", "選好插、拔不掉的那一個")
    base(rowsB, 42, f"底板 B（{t:g} mm）", "選卡得緊、又不會裂的那一個")
    # 直立的小片
    x, y, rowh = 6.0, 82.0, 0
    for k, v in items:
        o, hs, e, slot, lab = tab_loop(k, t, v)
        hgt = o[:, 1].max()
        if x + 22 > 294: x, y = 6.0, y + rowh + 3; rowh = 0
        sh.part(lab, o, hs, e, dx=x, dy=y); x += 23; rowh = max(rowh, hgt)
    # 楔子 2 片（長 22，寬 3 → 7）
    y2 = y + rowh + 4
    for i in range(2):
        wl = np.array([(0, 0), (22, 0), (22, 7), (0, 3)], float)
        sh.part(f"楔子{i+1}", wl, (), text("楔", 2.4, 14, 5.3), dx=6 + i * 26, dy=y2)
    return sh

# ---------------- 十字軸測試板 ----------------
def cross_axle_test(t=3.0, aw=10.0):
    """十字軸（兩片 aw 寬的木條十字插接）：試軸承孔要多大才轉得順、十字孔要多緊才卡得住。"""
    sh = Sheet(300, 210, f"十字軸測試板 {t:g} mm")
    base_d = math.hypot(aw, t)
    bear = [.2, .4, .6, .8, 1.0]; fits = [-.1, -.2, -.3, -.4, -.5]
    W = len(bear) * 20 + 8; H = 30
    holes = []; eng = []
    for i, c in enumerate(bear):
        cx = 4 + 10 + i * 20; holes.append(circle(cx, 13, (base_d + c) / 2)); eng += text(f"+{c:.1f}".replace("0.", "."), 2.6, cx, 27.0)
    eng += text("軸承孔：轉得順", 2.4, W - 4, 4.2, anchor="end", bold=False)
    sh.part("軸承孔", rrect(0, 0, W, H, 2), holes, eng, dx=4, dy=4)
    holes = []; eng = []
    for i, f in enumerate(fits):
        cx = 4 + 10 + i * 20; holes.append(plus_hole(cx, 13, aw + f, t + .3)); eng += text(("0" if f == 0 else f"{f:.1f}").replace("0.", "."), 2.6, cx, 27.0)
    eng += text("十字孔：卡得住", 2.4, W - 4, 4.2, anchor="end", bold=False)
    sh.part("十字孔", rrect(0, 0, W, H, 2), holes, eng, dx=4, dy=4 + H + 4)
    L = 70.0; mid = L / 2; hw = (t + .2) / 2
    A = np.array([(0, 0), (L, 0), (L, aw), (0, aw), (0, aw / 2 + hw), (mid + .5, aw / 2 + hw), (mid + .5, aw / 2 - hw), (0, aw / 2 - hw)])
    B = np.array([(0, 0), (L, 0), (L, aw / 2 - hw), (mid - .5, aw / 2 - hw), (mid - .5, aw / 2 + hw), (L, aw / 2 + hw), (L, aw), (0, aw)])
    sh.part("十字軸A", A, (), text("A", 3, L - 5, 4.3), dx=4, dy=4 + 2 * (H + 4))
    sh.part("十字軸B", B, (), text("B", 3, 5, 4.3), dx=4, dy=4 + 2 * (H + 4) + aw + 3)
    return sh

if __name__ == "__main__":
    ca = cross_axle_test().fit(); ca.save(f"{OUT}/十字軸測試板_3mm.svg"); ca.preview(f"{OUT}/十字軸測試板_預覽.png", 5)
    ar = axle_ruler().fit(); ar.save(f"{OUT}/軸孔測試尺.svg"); ar.preview(f"{OUT}/軸孔測試尺_預覽.png", 6)
    for t in (3, 5):
        jb = joint_board(t).fit(); jb.save(f"{OUT}/接頭測試板_{t}mm.svg"); jb.preview(f"{OUT}/接頭測試板_{t}mm_預覽.png", 4)
    print("ok", ar.w, ar.h)
