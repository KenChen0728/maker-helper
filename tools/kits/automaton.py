"""手搖招潮蟹（雷切 3 mm）：盒子框架用凸榫＋竹籤插銷，十字軸穿過兩邊軸承孔，凸輪推動從動桿，把招潮蟹的大螯推上去、放下來。
三顆凸輪可以換：大偏心（揮得高）、小偏心（揮得低）、雙峰（轉一圈揮兩次）。
"""
import math, os, sys, numpy as np
from kitlib import *
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)

T = 3.0; SLOT_C = .3; PIN = 3.3; DELTA = .2; TAB_E = 7.0; TAPER = .3
AW = 10.0                         # 十字軸每片的寬
HUB_FIT = -.2                     # 十字孔長邊比軸寬少多少（越負越緊；試十字軸測試板後改）
BEAR = math.hypot(AW, T) + .6     # 軸承孔直徑（十字軸轉得動）
DOWEL = 4.0; DOWEL_TIGHT = -.1    # 搖把用 4 mm 木棒
SK_TIGHT, SK_LOOSE = 3.1, 3.6     # 樞軸竹籤：身體卡緊、大螯會轉
Z0 = 15.0                         # 整組機構離桌面多高（側板下面當腳，搖把轉到最低才不會敲到桌子）
SHAFT_Z = 31.0 + Z0
X_IN = 40.0; Y_IN = 33.0          # 兩片側板內側 ±40，前後 ±33
BASE_Z = (7.0 + Z0, 10.0 + Z0); TOP_Z = (57.0 + Z0, 60.0 + Z0); WALL_H = 63.0 + Z0
FRONT_Y = (-30.0, -27.0)
CAM_X = -30.0; PIVOT = (-14.0, 74.0); R1 = 16.0   # PIVOT 是蟹身平面圖上的座標（3D 再加 Z0）
AX_X = (-47.0, 56.0)              # 十字軸左右兩端
COLLAR_L = -43.4                  # 左側板外面的軸環（內側面 x）：擋住軸往右滑
SPACER_X0 = (43.4, 46.4, 49.4)    # 右側板外面三個軸環當墊片：搖把要躲開凸榫（凸榫凸到 x=50）
CRANK_X0 = 52.4                   # 搖把內側面
FOOT_Y = (-9.0, 9.0)              # 腳板前後範圍（凸輪最高點會前後跑 ±5）

# ---------- 小工具 ----------
def tabbed_rect(x0, y0, x1, y1, tabs):
    """SVG 座標的長方形，邊上加凸榫。tabs：[(邊 'T'/'R'/'B'/'L', a, b, 樣式 'pin'/'taper'), ...]（a、b 是沿著那一邊的座標）。回傳（外框, 竹籤孔）"""
    holes = []
    def tab_pts(edge, a, b, style):
        d = T + TAB_E if style == "pin" else T
        if style == "taper": ra, rb, ta, tb = a - TAPER / 2, b + TAPER / 2, a + .2, b - .2
        else: ra, rb, ta, tb = a, b, a, b
        c = .5
        if edge == "T":   # 往 −y 凸
            pts = [(ra, y0), (ta, y0 - d + c), (ta + c, y0 - d), (tb - c, y0 - d), (tb, y0 - d + c), (rb, y0)]
            if style == "pin": holes.append(circle((a + b) / 2, y0 - (T - DELTA) - PIN / 2, PIN / 2))
        elif edge == "B":  # 往 +y 凸，沿 −x 走
            pts = [(rb, y1), (tb, y1 + d - c), (tb - c, y1 + d), (ta + c, y1 + d), (ta, y1 + d - c), (ra, y1)]
            if style == "pin": holes.append(circle((a + b) / 2, y1 + (T - DELTA) + PIN / 2, PIN / 2))
        elif edge == "R":  # 往 +x 凸，沿 +y 走
            pts = [(x1, ra), (x1 + d - c, ta), (x1 + d, ta + c), (x1 + d, tb - c), (x1 + d - c, tb), (x1, rb)]
            if style == "pin": holes.append(circle(x1 + (T - DELTA) + PIN / 2, (a + b) / 2, PIN / 2))
        else:              # L：往 −x 凸，沿 −y 走
            pts = [(x0, rb), (x0 - d + c, tb), (x0 - d, tb - c), (x0 - d, ta + c), (x0 - d + c, ta), (x0, ra)]
            if style == "pin": holes.append(circle(x0 - (T - DELTA) - PIN / 2, (a + b) / 2, PIN / 2))
        return pts
    out = [(x0, y0)]
    for e, a, b, st in sorted([t for t in tabs if t[0] == "T"], key=lambda t: t[1]): out += tab_pts(e, a, b, st)
    out.append((x1, y0))
    for e, a, b, st in sorted([t for t in tabs if t[0] == "R"], key=lambda t: t[1]): out += tab_pts(e, a, b, st)
    out.append((x1, y1))
    for e, a, b, st in sorted([t for t in tabs if t[0] == "B"], key=lambda t: -t[1]): out += tab_pts(e, a, b, st)
    out.append((x0, y1))
    for e, a, b, st in sorted([t for t in tabs if t[0] == "L"], key=lambda t: -t[1]): out += tab_pts(e, a, b, st)
    return np.array(out, float), holes

def slot(cx, cy, along, across, style, horizontal=True):
    """配合凸榫的槽：along＝凸榫寬（pin 多 0.4、taper 剛好），across＝板厚方向（T+SLOT_C）。"""
    L = along + (.4 if style == "pin" else 0.0); Wd = across
    return rect(cx - L / 2, cy - Wd / 2, L, Wd) if horizontal else rect(cx - Wd / 2, cy - L / 2, Wd, L)

def cam_outline(kind):
    if kind == "big": return circle(5.0, 0, 14.0)
    if kind == "small": return circle(3.0, 0, 14.0)
    th = np.linspace(0, 2 * math.pi, 240, endpoint=False); r = 15.5 + 3.5 * np.cos(2 * th)
    return np.stack([r * np.cos(th), r * np.sin(th)], 1)

# ---------- 零件（各自的平面座標；SVG：y 往下） ----------
def wall(side):
    """側板。side=+1 右（搖把那邊），−1 左。u 對應 y（從外面看），v＝63−z。"""
    U = (lambda y: y + Y_IN) if side > 0 else (lambda y: Y_IN - y); V = lambda z: WALL_H - z
    d = Drawing(-1, -1, 2 * Y_IN + 1, WALL_H + 1, .04)
    o = rrect(0, 0, 2 * Y_IN, WALL_H, 4); d.poly([tuple(p) for p in o])
    d.ellipse(Y_IN, WALL_H, 19, 12, fill=0)               # 下面挖一個拱，變成兩隻腳
    outer = d.loops()[0]
    holes = [circle(U(0), V(SHAFT_Z), BEAR / 2)]
    holes.append(slot(U(0), V(sum(BASE_Z) / 2), 30, T + SLOT_C, "pin"))
    holes.append(slot(U(0), V(sum(TOP_Z) / 2), 30, T + SLOT_C, "pin"))
    holes.append(slot(U(sum(FRONT_Y) / 2), V(42 + Z0), 12, T + SLOT_C, "pin", horizontal=False))
    eng = text("右" if side > 0 else "左", 3.5, U(24 if side > 0 else -24), V(Z0 + 14))
    if side > 0: eng += text("轉一轉", 4.2, U(0), V(SHAFT_Z - 10))
    return outer, holes, eng, []

def plate(kind):
    """底板（kind='base'）或頂板（'top'）。u＝x+50，v＝33−y。兩側凸榫穿過側板插竹籤。"""
    U = lambda x: x + 50; V = lambda y: Y_IN - y
    outer, holes = tabbed_rect(U(-X_IN), V(Y_IN), U(X_IN), V(-Y_IN), [("L", V(15), V(-15), "pin"), ("R", V(15), V(-15), "pin")])
    for x in (-20, 20): holes.append(slot(U(x), V(sum(FRONT_Y) / 2), 10, T + SLOT_C, "taper"))
    eng = []
    if kind == "top":
        for x in (-8, 8): holes.append(slot(U(x), V(1.5), 8, T + SLOT_C, "taper"))
        holes.append(rect(U(CAM_X) - (T + .4) / 2, V(3.0), T + .4, 9.4))   # 從動桿的導槽
        pass
    return outer, holes, eng, []

def front():
    """前板：看得到凸輪的窗，右邊刻名字。u＝x+50，v＝60−z。"""
    U = lambda x: x + 50; V = lambda z: 60 - z
    outer, holes = tabbed_rect(U(-X_IN), V(57), U(X_IN), V(10),
                               [("L", V(48), V(36), "pin"), ("R", V(48), V(36), "pin"),
                                ("T", U(-25), U(-15), "taper"), ("T", U(15), U(25), "taper"),
                                ("B", U(-25), U(-15), "taper"), ("B", U(15), U(25), "taper")])
    holes.append(rrect(U(-36), V(52), 18, 38, 4))
    score = [np.vstack([p := rrect(U(-12), V(46), 48, 26, 3), p[:1]])]
    eng = text("我的招潮蟹", 6.0, U(12), V(30))
    return outer, holes, eng, score

def axle_strip(which):
    """十字軸：兩片從相反的一端開槽，互相插起來就是十字形。u＝x+45。"""
    L = AX_X[1] - AX_X[0]; mid = L / 2; hw = (T + .2) / 2
    if which == "A":   # 左半邊開槽
        P = [(0, 0), (L, 0), (L, AW), (0, AW), (0, AW / 2 + hw), (mid + .5, AW / 2 + hw), (mid + .5, AW / 2 - hw), (0, AW / 2 - hw)]
    else:              # 右半邊開槽
        P = [(0, 0), (L, 0), (L, AW / 2 - hw), (mid - .5, AW / 2 - hw), (mid - .5, AW / 2 + hw), (L, AW / 2 + hw), (L, AW), (0, AW)]
    eng = text(which, 3.0, 6 if which == "B" else L - 6, 4.3)
    if which == "B":   # 凸輪的位置：套在兩條線中間
        u0 = CAM_X - T / 2 - AX_X[0]
        eng += [rect(u0 - .5, 1.0, .5, AW - 2.0), rect(u0 + T, 1.0, .5, AW - 2.0)] + text("凸輪", 3.0, u0 + T + 7.5, 6.5)
    return np.array(P, float), [], eng, []

def collar():
    return circle(0, 0, 8.0), [plus_hole(0, 0, AW + HUB_FIT, T + SLOT_C)], [], []

def cam(kind):
    lab = {"big": "大", "small": "小", "double": "雙"}[kind]
    off = {"big": 5.0, "small": 3.0, "double": 0.0}[kind]
    return cam_outline(kind), [plus_hole(0, 0, AW + HUB_FIT, T + SLOT_C)], text(lab, 3.6, off + (8 if kind != "double" else 9), 1.6), []

def rod():
    """從動桿（u＝y+6.2，v＝74−z，最低位置）：下面的凸榫插進腳板，上面的叉口托住大螯。"""
    pts = [(-6.2, 43), (-4.7 - TAPER / 2, 43), (-4.5, 40), (1.1, 40), (1.3 + TAPER / 2, 43), (2.8, 43), (2.8, 74), (0.0, 74), (0.0, 70), (-3.4, 70), (-3.4, 74), (-6.2, 74)]
    P = np.array([(y + 6.2, 74 - z) for (y, z) in pts], float)
    return P, [], [], []

def foot():
    """腳板（水平，壓在凸輪上）。u＝x+36，v＝FOOT_Y[1]−y。前後 16 mm：凸輪最高點會前後跑。"""
    outer = rrect(0, 0, 12, FOOT_Y[1] - FOOT_Y[0], 2)
    holes = [rect((CAM_X + 36) - (T + SLOT_C) / 2, FOOT_Y[1] - 1.3, T + SLOT_C, 6.0)]
    return outer, holes, [], []

def crank():
    d = Drawing(-9, -33, 9, 9, .03); d.ellipse(0, 0, 7.5, 7.5); d.ellipse(0, -24, 6, 6); d.poly([(-6, 0), (6, 0), (5, -24), (-5, -24)])
    outer = d.loops()[0]
    return outer, [plus_hole(0, 0, AW + HUB_FIT, T + SLOT_C), circle(0, -24, (DOWEL + DOWEL_TIGHT) / 2)], [], []

def crab_body():
    """招潮蟹的身體（正面），站在頂板上（兩個凸榫）；左肩有大螯的轉軸孔。座標：x、z（z 往上），轉成 u＝x+32、v＝96−z。"""
    d = Drawing(-32, 54, 32, 98, .04)
    P = lambda x, z: (x, 152 - z)       # 畫布上 y 往下：用 152−z 讓上下顛倒，最後再換回來
    d.ellipse(*P(0, 76), 19, 11)
    d.poly([P(-14, 60), P(14, 60), P(14, 70), P(-14, 70)])
    for s in (-1, 1):
        d.line([P(s * 12, 70), P(s * 21, 66), P(s * 24, 60)], 2.4)
        d.line([P(s * 8, 67), P(s * 15, 63), P(s * 17, 60)], 2.2)
        d.line([P(s * 5, 92), P(s * 7, 86)], 2.2); d.ellipse(*P(s * 8, 94), 3.2, 3.2)
    d.line([P(15, 77), P(22, 82)], 4.0); d.ellipse(*P(26, 85), 5.2, 4.2)
    d.poly([P(26, 85), P(33, 88), P(33, 83)], fill=0)
    for x0 in (-12, 4):
        d.poly([P(x0 - TAPER / 2, 60.2), P(x0 + .2, 57), P(x0 + 8 - .2, 57), P(x0 + 8 + TAPER / 2, 60.2)])
    loops = d.loops()
    to_uv = lambda L: np.stack([L[:, 0] + 32, 96 - (152 - L[:, 1])], 1)
    outer = to_uv(loops[0]); holes = [to_uv(l) for l in loops[1:]]
    holes.append(circle(PIVOT[0] + 32, 96 - PIVOT[1], SK_TIGHT / 2))
    eng = [circle(-8 + 32, 96 - 94, 1.3), circle(8 + 32, 96 - 94, 1.3)]
    smile = np.array([(x + 32, 96 - (70.5 + .05 * x * x)) for x in np.linspace(-5, 5, 12)])
    shell = [np.array([(x + 32, 96 - (80 + .01 * x * x)) for x in np.linspace(-11, 11, 16)])]
    return outer, holes, eng, [smile] + shell

def big_claw():
    """招潮蟹的大螯：繞肩膀的竹籤轉。座標以轉軸為原點（x、z），u＝x+46，v＝16−z。"""
    d = Drawing(-46, -14, 8, 18, .04)
    P = lambda x, z: (x, 4 - z)
    d.ellipse(*P(0, 0), 6, 6); d.poly([P(-24, -4), P(0, -4), P(0, 4), P(-24, 5)])
    d.ellipse(*P(-32, 4), 11, 9.5)
    d.poly([P(-30, 2), P(-46, 9), P(-46, -2)], fill=0)
    d.ellipse(*P(-36, -4.5), 7, 3.2)
    loops = d.loops()
    to_uv = lambda L: np.stack([L[:, 0] + 46, 16 - (4 - L[:, 1])], 1)
    outer = to_uv(loops[0]); holes = [to_uv(l) for l in loops[1:]] + [circle(46, 16, SK_LOOSE / 2)]
    score = [np.array([(46 + x, 16 - z) for (x, z) in [(-36, 9), (-30, 11), (-24, 9)]]), np.array([(46 + x, 16 - z) for (x, z) in [(-38, 5), (-31, 6.5)]])]
    return outer, holes, [], score

def washer():
    return circle(0, 0, 3.5), [circle(0, 0, 1.5)], [], []

# ---------- 排版 ----------
def all_parts():
    P = []
    P.append(("右側板", *wall(1))); P.append(("左側板", *wall(-1)))
    P.append(("底板", *plate("base"))); P.append(("頂板", *plate("top"))); P.append(("前板", *front()))
    P.append(("十字軸A", *axle_strip("A"))); P.append(("十字軸B", *axle_strip("B")))
    for i in range(4): P.append((f"軸環{i+1}", *collar()))
    for k in ("big", "small", "double"): P.append((f"凸輪{k}", *cam(k)))
    P.append(("從動桿", *rod())); P.append(("腳板", *foot())); P.append(("搖把", *crank()))
    P.append(("蟹身", *crab_body())); P.append(("大螯", *big_claw())); P.append(("小墊圈", *washer()))
    return P

BOX = ("右側板", "左側板", "底板", "頂板", "前板")
def sheets():
    """兩張：外殼（盒子）、機構和角色。"""
    parts = all_parts(); out = []
    for title, group in (("外殼", [p for p in parts if p[0] in BOX]), ("機構", [p for p in parts if p[0] not in BOX])):
        packs = pack(group, W=292, H=202)
        for i, pk in enumerate(packs):
            sh = Sheet(300, 210, f"手搖招潮蟹 {title}（3 mm）")
            for (name, o, hs, e, sc), dx, dy in pk: sh.part(name, o, hs, e, sc, dx=dx, dy=dy)
            out.append((title + ("" if len(packs) == 1 else str(i + 1)), sh.fit()))
    return out

# ---------- 3D：組起來的樣子 ----------
def assembly(phi=1.2, cam_kind="big"):
    sys.path.insert(0, os.path.join(HERE, "..", "card-images"))
    from wood_car import plate3d, cylinder
    import scene
    items = []
    def add(o, hs, O, ex, ey, en, eng=(), sc=(), tone="wood"):
        tr, dec, d = plate3d(np.asarray(o), [np.asarray(h) for h in hs], T, O, ex, ey, en, eng, sc); items.append((tr, tone, dec, d))
    o, hs, e, sc = wall(1); add(o, hs, (X_IN + T, -Y_IN, WALL_H), (0, 1, 0), (0, 0, -1), (1, 0, 0), e, sc)
    o, hs, e, sc = wall(-1); add(o, hs, (-X_IN - T, Y_IN, WALL_H), (0, -1, 0), (0, 0, -1), (-1, 0, 0), e, sc)
    o, hs, e, sc = plate("base"); add(o, hs, (-50, Y_IN, BASE_Z[1]), (1, 0, 0), (0, -1, 0), (0, 0, 1), e, sc)
    o, hs, e, sc = plate("top"); add(o, hs, (-50, Y_IN, TOP_Z[1]), (1, 0, 0), (0, -1, 0), (0, 0, 1), e, sc)
    o, hs, e, sc = front(); add(o, hs, (-50, FRONT_Y[0], 60 + Z0), (1, 0, 0), (0, 0, -1), (0, -1, 0), e, sc)
    # 竹籤插銷
    xp = X_IN + (T - DELTA) + PIN / 2
    for s in (1, -1):
        for zc in (sum(BASE_Z) / 2, sum(TOP_Z) / 2): items.append((cylinder((s * xp, 0, zc - 7), (s * xp, 0, zc + 7), 1.5), "stick", None, None))
        items.append((cylinder((s * xp, sum(FRONT_Y) / 2 - 7, 42 + Z0), (s * xp, sum(FRONT_Y) / 2 + 7, 42 + Z0), 1.5), "stick", None, None))
    # 十字軸（跟著轉）
    def rotyz(v, a):
        y, z = v; return (y * math.cos(a) - z * math.sin(a), y * math.sin(a) + z * math.cos(a))
    a = phi
    def O_rot(x, du, dv):   # 軸上某一點（局部 u 沿 y、v 沿 −z）轉 a 之後
        y, z = rotyz((du, -dv), a); return np.array([x, y, SHAFT_Z + z])
    ex_r = np.array([0, *rotyz((1, 0), a)]); ey_r = np.array([0, *rotyz((0, -1), a)])
    o, hs, e, sc = axle_strip("A")   # 水平那片：u→x，v→−y（轉 a）
    exA = np.array([1.0, 0, 0]); eyA = -ex_r; enA = np.array([0, *rotyz((0, 1), a)])
    OA = np.array([AX_X[0], 0, SHAFT_Z]) + eyA * (-AW / 2) + enA * (T / 2)
    add(o, hs, OA, exA, eyA, enA)
    o, hs, e, sc = axle_strip("B")   # 直的那片：u→x，v→−z（轉 a）
    eyB = ey_r; enB = -ex_r
    OB = np.array([AX_X[0], 0, SHAFT_Z]) + eyB * (-AW / 2) + enB * (T / 2)
    add(o, hs, OB, exA, eyB, enB)
    for x0 in (COLLAR_L - T,) + SPACER_X0:
        o, hs, e, sc = collar(); add(o, hs, np.array([x0 + T, 0, SHAFT_Z]), ex_r, ey_r, (1, 0, 0))
    o, hs, e, sc = cam(cam_kind); add(o, hs, np.array([CAM_X + T / 2, 0, SHAFT_Z]), ex_r, ey_r, (1, 0, 0), e, sc)
    # 凸輪最高點 → 腳板、從動桿、大螯
    co = cam_outline(cam_kind); zs = [SHAFT_Z + rotyz((u, -v), a)[1] for (u, v) in co]; lift = max(zs) - (40.0 + Z0)
    o, hs, e, sc = foot(); add(o, hs, (-36, FOOT_Y[1], 43 + Z0 + lift), (1, 0, 0), (0, -1, 0), (0, 0, 1))
    o, hs, e, sc = rod(); add(o, hs, (CAM_X + T / 2, -6.2, 74 + Z0 + lift), (0, 1, 0), (0, 0, -1), (1, 0, 0))
    o, hs, e, sc = crank(); add(o, hs, np.array([CRANK_X0 + T, 0, SHAFT_Z]), ex_r, ey_r, (1, 0, 0))
    hx = CRANK_X0 + T; hy, hz = rotyz((0, 24), a)
    items.append((cylinder((hx - T, hy, SHAFT_Z + hz), (hx + 22, hy, SHAFT_Z + hz), DOWEL / 2), "stick", None, None))
    o, hs, e, sc = crab_body(); add(o, hs, (-32, 0, 96 + Z0), (1, 0, 0), (0, 0, -1), (0, -1, 0), e, sc)
    al = math.atan2(lift, R1)   # 大螯抬起的角度
    o, hs, e, sc = big_claw()
    ca, sa = math.cos(al), math.sin(al)
    exC = np.array([ca, 0, -sa]); eyC = np.array([-sa, 0, -ca])   # u 沿 x、v 沿 −z，繞 y 軸轉（左邊抬起）
    OC = np.array([PIVOT[0], -3.2, PIVOT[1] + Z0]) - exC * 46 - eyC * 16
    add(o, hs, OC, exC, eyC, (0, -1, 0), e, sc)
    o, hs, e, sc = washer(); add(o, hs, (PIVOT[0], -6.4, PIVOT[1] + Z0), (1, 0, 0), (0, 0, -1), (0, -1, 0))
    items.append((cylinder((PIVOT[0], -8.0, PIVOT[1] + Z0), (PIVOT[0], 3.5, PIVOT[1] + Z0), 1.5), "stick", None, None))
    tris = []; tones = []; part = []; tri_dec = []; decs = []
    for pid, (tr, tone, dec, d) in enumerate(items):
        tris.append(tr); tones += [tone] * len(tr); part += [pid] * len(tr)
        if d is None: tri_dec += [-1] * len(tr)
        else: decs.append(d); tri_dec += [len(decs) - 1 if f else -1 for f in dec]
    tris = np.concatenate(tris); cols = np.array([scene.hexc(scene.COL.get(t, "#cccccc")) for t in tones])
    sh = np.concatenate([d["shadow"] for d in decs] + [tr for (tr, tone, dec, d) in items if d is None])
    return tris, cols, tones, np.array(part), decs, np.array(tri_dec), sh, lift, al

def render(out, phi=1.2, cam_kind="big", az=math.pi - .5, el=.62, W=960, H=660):
    sys.path.insert(0, os.path.join(HERE, "..", "card-images"))
    import scene
    tris, cols, tones, part, decs, tri_dec, sh, lift, al = assembly(phi, cam_kind)
    return scene.render_scene(tris, cols, tones, part, ground="kraft", az=az, el=el, decals=decs, tri_decal=tri_dec, shadow_tris=sh, W=W, H=H, out=out)

if __name__ == "__main__":
    for name, sh in sheets():
        sh.save(f"{OUT}/手搖招潮蟹_{name}_3mm.svg"); sh.preview(f"{OUT}/手搖招潮蟹_{name}_預覽.png", 4); print("sheet", name, sh.w, sh.h, len(sh.parts))
    if "render" in sys.argv: render(f"{OUT}/手搖招潮蟹_成品.png"); print("render ok")

def motion_check(cam_kind="big", n=720):
    """轉一整圈：凸輪碰不碰底板、腳板撞不撞頂板、接觸點在不在腳板上、搖把會不會掃到凸榫。"""
    co = cam_outline(cam_kind); rows = []
    for k in range(n):
        a = 2 * math.pi * k / n; c, s_ = math.cos(a), math.sin(a)
        Y = np.array([u * c + v * s_ for (u, v) in co]); Z = SHAFT_Z + np.array([u * s_ - v * c for (u, v) in co])
        i = int(np.argmax(Z)); rows.append((Z[i] - (40.0 + Z0), Z.min(), Y[i]))
    R = np.array(rows); lift = R[:, 0]; claw = np.degrees(np.arctan2(lift, R1))
    tab_tip = X_IN + T + TAB_E; play = .4; sweep = 24 + 6
    f = lambda v: round(float(v), 1)
    return {"抬升": (f(lift.min()), f(lift.max())), "大螯角度": (f(claw.min()), f(claw.max())),
            "腳板到頂板": f(TOP_Z[0] - (43 + Z0 + lift.max())), "凸輪到底板": f(R[:, 1].min() - BASE_Z[1]),
            "接觸點前後": (f(R[:, 2].min()), f(R[:, 2].max())),
            "接觸點離腳板邊": f(min(R[:, 2].min() - FOOT_Y[0], FOOT_Y[1] - R[:, 2].max())),
            "搖把到凸榫": f(CRANK_X0 - play - tab_tip), "搖把離桌面": f(SHAFT_Z - sweep)}

def animate(out, cam_kind="big", frames=16, W=640, H=440):
    sys.path.insert(0, os.path.join(HERE, "..", "card-images"))
    import scene
    from PIL import Image
    ims = []; cam = None
    for k in range(frames):
        phi = 2 * math.pi * k / frames
        tris, cols, tones, part, decs, tri_dec, sh, lift, al = assembly(phi, cam_kind)
        im = scene.render_scene(tris, cols, tones, part, ground="kraft", az=math.pi - .5, el=.62, decals=decs, tri_decal=tri_dec, shadow_tris=sh, W=W, H=H, cam_fix=cam)
        if cam is None: cam = scene.LAST_CAM
        ims.append(im.convert("P", palette=Image.ADAPTIVE, colors=200))
    ims[0].save(out, save_all=True, append_images=ims[1:], duration=110, loop=0, optimize=True)
    return out
