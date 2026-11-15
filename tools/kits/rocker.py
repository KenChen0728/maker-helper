"""搖搖動物筆架（雷切 3 mm）。
兩片弧形底的搖椅腳（上緣是海浪），中間一片座板用凸榫＋竹籤插銷固定；座板上用指接榫組一個筆筒（四片牆，牆的凸榫插進座板）；
筆筒前面的槽插小朋友畫的動物立牌（創客小幫手「立牌」的 8 mm 插腳，槽寬用機器校正的插槽寬度）。
搖椅底的半徑 110 mm：插滿筆時重心也在弧心下面很多，推了一定會搖回來。
"""
import math, os, sys, numpy as np
from PIL import Image
from kitlib import *
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)

T = 3.0; SLOT_C = .3; PIN = 3.3; DELTA = .2; TAB_E = 7.0; TAPER = .3
SNUG = 3.05                         # 立牌的槽：用機器校正的插槽寬度（小幫手預設 3.05）
R_ARC = 110.0; ARC_DEG = 35.0       # 搖椅底的半徑、左右各幾度
BOX = 52.0; BOX_H = 70.0            # 筆筒外徑、高
W_IN = 66.0                         # 兩片搖椅腳內側的距離（立牌最寬 60 mm 才放得進來）
SEAT_Z = 20.0                       # 座板上緣離桌面（不搖的時候）
SEAT_Y = (-44.0, 40.0)              # 座板前後（前面是 −y，放立牌）
BOX_Y0 = -14.0                      # 筆筒前緣
STANDEE_Y = -32.0                   # 立牌的槽
FINGER = 10.0; FCLR = .1            # 指接榫每一指 10 mm，兩端各留 0.1
SC_ST = .225                        # 範例立牌（小黑鮪魚）縮放

def tab_pts(edge, a, b, style, x0, y0, x1, y1, holes):
    """長方形某一邊上的凸榫（和招潮蟹一樣）：'pin' 凸出 T+TAB_E 插竹籤，'taper' 凸出 T 壓緊。"""
    d = T + TAB_E if style == "pin" else T
    if style == "taper": ra, rb, ta, tb = a - TAPER / 2, b + TAPER / 2, a + .2, b - .2
    else: ra, rb, ta, tb = a, b, a, b
    c = .5
    if edge == "L":
        if style == "pin": holes.append(circle(x0 - (T - DELTA) - PIN / 2, (a + b) / 2, PIN / 2))
        return [(x0, rb), (x0 - d + c, tb), (x0 - d, tb - c), (x0 - d, ta + c), (x0 - d + c, ta), (x0, ra)]
    if edge == "R":
        if style == "pin": holes.append(circle(x1 + (T - DELTA) + PIN / 2, (a + b) / 2, PIN / 2))
        return [(x1, ra), (x1 + d - c, ta), (x1 + d, ta + c), (x1 + d, tb - c), (x1 + d - c, tb), (x1, rb)]
    if edge == "B":
        return [(rb, y1), (tb, y1 + d - c), (tb - c, y1 + d), (ta + c, y1 + d), (ta, y1 + d - c), (ra, y1)]
    raise ValueError(edge)

# ---------------- 零件 ----------------
def rocker():
    """搖椅腳（直立，在 y-z 平面）：u＝y+70，v＝60−z。下緣是半徑 R_ARC 的弧，兩端圓頭，上緣是海浪。"""
    U = lambda y: y + 70; V = lambda z: 60 - z
    th = np.radians(np.linspace(-ARC_DEG, ARC_DEG, 90))
    arc = [(R_ARC * math.sin(t), R_ARC - R_ARC * math.cos(t)) for t in th]
    yR, zR = arc[-1]; yL, zL = arc[0]
    ztop = lambda y: 40 + 3.5 * math.cos(2 * math.pi * y / 42) - 9 * (y / 63) ** 2
    top = [(y, ztop(y)) for y in np.linspace(yR - 4, yL + 4, 90)]
    def cap(p0, p1, out_sign):          # 兩端的半圓頭：從 p0（弧的端點）繞到 p1（浪的端點）
        c = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2); r = math.dist(p0, p1) / 2
        a0 = math.atan2(p0[1] - c[1], p0[0] - c[0])
        return [(c[0] + r * math.cos(a0 + out_sign * t), c[1] + r * math.sin(a0 + out_sign * t)) for t in np.linspace(0, math.pi, 16)[1:-1]]
    P = arc + cap((yR, zR), top[0], 1) + top + cap(top[-1], (yL, zL), 1)
    d = Drawing(-2, -2, 145, 66, .04); d.poly([(U(y), V(z)) for (y, z) in P])
    outer = d.loops()[0]
    holes = [rect(U(a) - .2, V(SEAT_Z) - SLOT_C / 2, b - a + .4, T + SLOT_C) for (a, b) in SEAT_TABS]
    return outer, holes, text("搖一搖", 4.0, U(0), V(25)), []

def crsp(P, n=8):
    P = np.asarray(P, float); out = []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, len(P) - 1)]
        for t in np.linspace(0, 1, n, endpoint=False)[1:] if i == 0 else np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t ** 3
            out.append(tuple(.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)))
    return out

SEAT_TABS = ((-36.0, -20.0), (16.0, 32.0))           # 座板兩側凸榫（沿 y）

def seat():
    """座板（水平）：u＝x+W_IN/2，v＝SEAT_Y[1]−y（從上面看，前面在下）。"""
    U = lambda x: x + W_IN / 2; V = lambda y: SEAT_Y[1] - y
    holes = []; pts = [(U(-W_IN / 2), V(SEAT_Y[1])), (U(W_IN / 2), V(SEAT_Y[1]))]
    for (a, b) in sorted(SEAT_TABS, key=lambda t: -t[1]):  # 右邊：沿 −y 走（v 增加）
        pts += tab_pts("R", V(b), V(a), "pin", U(-W_IN / 2), V(SEAT_Y[1]), U(W_IN / 2), V(SEAT_Y[0]), holes)
    pts += [(U(W_IN / 2), V(SEAT_Y[0])), (U(-W_IN / 2), V(SEAT_Y[0]))]
    for (a, b) in sorted(SEAT_TABS, key=lambda t: t[0]):
        pts += tab_pts("L", V(b), V(a), "pin", U(-W_IN / 2), V(SEAT_Y[1]), U(W_IN / 2), V(SEAT_Y[0]), holes)
    outer = np.array(pts, float)
    # 筆筒四面牆的凸榫槽
    for (cx, cy, along_x) in wall_slots():
        for (a, b) in ((-16.0, -6.0), (6.0, 16.0)):
            if along_x: holes.append(rect(U(cx + a), V(cy) - (T + SLOT_C) / 2, b - a, T + SLOT_C))
            else: holes.append(rect(U(cx) - (T + SLOT_C) / 2, V(cy + b), T + SLOT_C, b - a))
    holes.append(rect(U(-4.2), V(STANDEE_Y) - SNUG / 2, 8.4, SNUG))           # 立牌的槽（8 mm 插腳）
    eng = text("插動物", 3.2, U(0), V(STANDEE_Y - 6.5))
    return outer, holes, eng, []

def wall_slots():
    """筆筒四面牆在座板上的位置：(中心 x, 中心 y, 牆沿 x 方向?)"""
    y0, y1 = BOX_Y0, BOX_Y0 + BOX; h = T / 2
    return [(0.0, y0 + h, True), (0.0, y1 - h, True), (-BOX / 2 + h, (y0 + y1) / 2, False), (BOX / 2 - h, (y0 + y1) / 2, False)]

def box_wall(kind):
    """筆筒的牆：kind 'A'（前後，兩邊指頭在 10–20、30–40、50–60）或 'B'（左右，指頭在 0–10、20–30、40–50、60–70）。
    u 沿牆、v＝BOX_H−z。下緣兩個梯形榫插進座板。"""
    inner = BOX - 2 * T; H = BOX_H
    fingers = [(z, z + FINGER) for z in np.arange(0, H, 2 * FINGER)] if kind == "B" else [(z, z + FINGER) for z in np.arange(FINGER, H, 2 * FINGER)]
    fingers = [(a, min(b, H)) for (a, b) in fingers if a < H]
    x0, x1 = T, T + inner
    def side(right):
        pts = []
        zs = sorted(fingers, key=lambda f: -f[0]) if right else sorted(fingers, key=lambda f: f[0])
        for (a, b) in zs:
            a2 = a + (FCLR if a > 0 else 0); b2 = b - (FCLR if b < H else 0)
            if right: pts += [(x1, H - b2), (x1 + T, H - b2), (x1 + T, H - a2), (x1, H - a2)]
            else: pts += [(x0, H - a2), (x0 - T, H - a2), (x0 - T, H - b2), (x0, H - b2)]
        return pts
    holes = []
    top = [(x0, 0), (x1, 0)]
    right = side(True)
    bottom = []
    for (a, b) in ((16.0, 6.0), (-6.0, -16.0)):   # 兩個梯形榫（沿牆的座標，從右往左）
        c = T + inner / 2
        bottom += tab_pts("B", c + b, c + a, "taper", x0, 0, x1, H, holes)
    left = side(False)
    pts = top + right + [(x1, H)] + bottom + [(x0, H)] + left
    pts = [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]
    eng = text("筆筒", 4.0, T + inner / 2, 14) if kind == "A" else []
    return np.array(pts, float), holes, eng, []

def tuna_standee():
    """範例立牌：小黑鮪魚（創客小幫手的立牌也是這樣：下面 8×6 的插腳）。"""
    import tuna as F
    o = F.body_outline() * SC_ST; o[:, 1] = -o[:, 1]
    lo, hi = o.min(0), o.max(0); o = o - [lo[0], lo[1]]
    W, H = hi[0] - lo[0], hi[1] - lo[1]
    d = Drawing(-2, -2, W + 2, H + 8, .04); d.poly([tuple(p) for p in o])
    bx = (62 + 178) / 2 * SC_ST - lo[0]
    d.rect(bx - 4, H - .5, 8, 6.5)
    L = d.loops()
    return L[0], [], [circle(26 * SC_ST - lo[0], (-60 * SC_ST) - lo[1], 1.4)] + text("黑鮪魚", 3.0, bx, H - 4), []

# ---------------- 檢查 ----------------
def physics(n_pens=12, pen_g=8.0, pen_L=140.0, rho=.00062):
    """重心高度（g/mm³ 合板約 0.62 g/cm³）。弧心在 R_ARC，重心要比弧心低才會搖回來。"""
    m = [];
    def area(P): P = np.asarray(P); return .5 * abs(np.dot(P[:, 0], np.roll(P[:, 1], 1)) - np.dot(P[:, 1], np.roll(P[:, 0], 1)))
    ro = rocker()[0]; A_r = area(ro); zc_r = 60 - ro[:, 1].mean()
    m.append((2 * A_r * T * rho, zc_r))
    so = seat()[0]; m.append((area(so) * T * rho, SEAT_Z - T / 2))
    for k in "AABB": wo = box_wall(k)[0]; m.append((area(wo) * T * rho, SEAT_Z + BOX_H / 2))
    so2 = tuna_standee()[0]; m.append((area(so2) * T * rho, SEAT_Z + 12))
    M0 = sum(a for a, b in m); z0 = sum(a * b for a, b in m) / M0
    M1 = M0 + n_pens * pen_g; z1 = (M0 * z0 + n_pens * pen_g * (SEAT_Z + pen_L / 2 * .97)) / M1
    return {"空的重量 g": round(M0), "空的重心 mm": round(z0, 1), f"插 {n_pens} 支筆的重心 mm": round(z1, 1), "弧心高 mm": R_ARC,
            "會搖回來": z1 < R_ARC, "最多搖幾度": ARC_DEG}

def all_parts():
    return [("搖椅腳1", *rocker()), ("搖椅腳2", *rocker()), ("座板", *seat()), ("筆筒前", *box_wall("A")), ("筆筒後", *box_wall("A")),
            ("筆筒左", *box_wall("B")), ("筆筒右", *box_wall("B")), ("範例立牌", *tuna_standee())]

def sheets():
    """一張：上排筆筒四面牆，下排座板、兩片搖椅腳、範例立牌。"""
    sh = Sheet(300, 210, "搖搖動物筆架 3 mm"); x = 4; hmax = 0
    for name, k in (("筆筒前", "A"), ("筆筒後", "A"), ("筆筒左", "B"), ("筆筒右", "B")):
        w = place(sh, name, box_wall(k), x, 4); x += w[0] + 4; hmax = max(hmax, w[1])
    y = 4 + hmax + 4; sw = place(sh, "座板", seat(), 4, y)
    rw = place(sh, "搖椅腳1", rocker(), 4 + sw[0] + 4, y); place(sh, "搖椅腳2", rocker(), 4 + sw[0] + 4, y + rw[1] + 3)
    place(sh, "範例立牌", tuna_standee(), 4 + sw[0] + 4 + rw[0] + 4, y)
    return [("1", sh.fit())]

# ---------------- 3D ----------------
def assembly(theta=0.0, pens=True):
    """theta：往前搖幾度（弧度，正＝往前）。滾動：弧心在 (y＝−R·θ, z＝R)，整個繞弧心轉。"""
    sys.path.insert(0, os.path.join(HERE, "..", "card-images"))
    from wood_car import plate3d, cylinder
    import scene
    items = []
    c, s = math.cos(theta), math.sin(theta)
    def Rm(p):      # 繞 x 軸、在弧心轉 theta，再滾動平移
        p = np.asarray(p, float); y, z = p[..., 1], p[..., 2] - R_ARC
        y2, z2 = y * c - z * s, y * s + z * c
        return np.stack([p[..., 0], y2 - R_ARC * theta, z2 + R_ARC], -1)
    def Rv(v): v = np.asarray(v, float); return np.array([v[0], v[1] * c - v[2] * s, v[1] * s + v[2] * c])
    def add(part, O, ex, ey, en, tone="wood"):
        o, hs, e, sc = part
        tr, dec, d = plate3d(np.asarray(o), [np.asarray(h) for h in hs], T, Rm(O), Rv(ex), Rv(ey), Rv(en), e, sc)
        d["shadow"] = d["shadow"]; items.append((tr, tone, dec, d))
    for x0 in (W_IN / 2 + T, -W_IN / 2): add(rocker(), (x0, -70, 60), (0, 1, 0), (0, 0, -1), (1, 0, 0))
    add(seat(), (-W_IN / 2, SEAT_Y[1], SEAT_Z), (1, 0, 0), (0, -1, 0), (0, 0, 1))
    y0, y1 = BOX_Y0, BOX_Y0 + BOX
    add(box_wall("A"), (-BOX / 2, y0, SEAT_Z + BOX_H), (1, 0, 0), (0, 0, -1), (0, -1, 0))
    add(box_wall("A"), (BOX / 2, y1, SEAT_Z + BOX_H), (-1, 0, 0), (0, 0, -1), (0, 1, 0))
    add(box_wall("B"), (BOX / 2, y0, SEAT_Z + BOX_H), (0, 1, 0), (0, 0, -1), (1, 0, 0))
    add(box_wall("B"), (-BOX / 2, y1, SEAT_Z + BOX_H), (0, -1, 0), (0, 0, -1), (-1, 0, 0))
    so = tuna_standee(); W = so[0][:, 0].max()
    bx = (62 + 178) / 2 * SC_ST - so[0][:, 0].min() if False else None
    tabx = so[0][np.argmax(so[0][:, 1]), 0]
    add(so, (-tabx, STANDEE_Y - T / 2, SEAT_Z + so[0][:, 1].max() - 6.0), (1, 0, 0), (0, 0, -1), (0, -1, 0))
    xp = W_IN / 2 + (T - DELTA) + PIN / 2
    for sx in (1, -1):
        for (a, b) in SEAT_TABS:
            p0, p1 = Rm((sx * xp, (a + b) / 2, SEAT_Z - T / 2 - 7)), Rm((sx * xp, (a + b) / 2, SEAT_Z - T / 2 + 7))
            items.append((cylinder(p0, p1, 1.5, 12), "stick", None, None))
    if pens:
        rng = np.random.default_rng(3); cols = ["pen1", "pen2", "pen3", "pen4", "pen5"]
        for k in range(7):
            bx, by = rng.uniform(-14, 14), rng.uniform(y0 + 10, y1 - 10); tx, ty = rng.uniform(-.22, .22), rng.uniform(-.22, .22)
            p0 = np.array([bx, by, SEAT_Z + 1]); d = np.array([tx, ty, 1.0]); d /= np.linalg.norm(d)
            items.append((cylinder(Rm(p0), Rm(p0 + d * 140), 3.8, 14), cols[k % 5], None, None))
    tris = []; tones = []; part = []; tri_dec = []; decs = []
    for pid, (tr, tone, dec, d) in enumerate(items):
        tris.append(tr); tones += [tone] * len(tr); part += [pid] * len(tr)
        if d is None: tri_dec += [-1] * len(tr)
        else: decs.append(d); tri_dec += [len(decs) - 1 if f else -1 for f in dec]
    for k, v in {"pen1": "#2f6fb5", "pen2": "#e0a422", "pen3": "#c8443c", "pen4": "#3c9a5f", "pen5": "#7a4fb0"}.items(): scene.COL.setdefault(k, v)
    tris = np.concatenate(tris); cols = np.array([scene.hexc(scene.COL.get(t, "#cccccc")) for t in tones])
    sh = np.concatenate([d["shadow"] for d in decs] + [tr for (tr, tone, dec, d) in items if d is None])
    return tris, cols, tones, np.array(part), decs, np.array(tri_dec), sh

def render(out, theta=0.0, pens=True, az=math.pi + .55, el=.48, W=960, H=660, cam_fix=None):
    sys.path.insert(0, os.path.join(HERE, "..", "card-images"))
    import scene
    tris, cols, tones, part, decs, tri_dec, sh = assembly(theta, pens)
    return scene.render_scene(tris, cols, tones, part, ground="kraft", az=az, el=el, decals=decs, tri_decal=tri_dec, shadow_tris=sh, W=W, H=H, out=out, cam_fix=cam_fix)

if __name__ == "__main__":
    print(physics())
    for name, sh in sheets():
        sh.save(f"{OUT}/搖搖動物筆架_{name}_3mm.svg"); sh.preview(f"{OUT}/搖搖動物筆架_{name}_預覽.png", 3); print("sheet", name, sh.w, sh.h, len(sh.parts))
