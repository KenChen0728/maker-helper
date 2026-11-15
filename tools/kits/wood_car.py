"""全木小車（雷切 3 mm）：兩片側板、三片隔板（凸榫穿出去插竹籤固定）、四個三層木輪（中間夾一條橡皮筋）、墊片、黑鮪魚乘客。"""
import math, os, sys, numpy as np
from PIL import Image, ImageDraw
from kitlib import *
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "card-images"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)

T = 3.0            # 板厚
W_IN = 50.0        # 兩片側板內側的距離（車寬）
L, H = 116.0, 46.0 # 側板長、高
AX = (22.0, 94.0); AZ = 12.0      # 輪軸位置
AXLE = 4.0         # 木棒直徑
AXLE_LOOSE = .4    # 側板、墊片的軸孔：會轉
AXLE_TIGHT = -.1   # 輪子的軸孔：卡緊，跟著軸一起轉
WR = 17.0          # 輪子半徑（中間那片小 1.5 mm，夾橡皮筋）
PIN = 3.3          # 竹籤孔
DELTA = .2         # 竹籤孔的內緣比板厚少多少（越大越緊）
TAB_E = 7.0        # 凸榫穿出側板外面多長
SLOT_C = .3        # 插槽在板厚方向多留的間隙（鬆，好插）
BULK = [(12.0, [(32, 42)], "前"), (58.0, [(19, 27), (32, 42)], "中"), (104.0, [(32, 42)], "後")]
BZ = (4.0, 44.0)   # 隔板上下
NOTCH = 6.0        # 中間隔板上方的乘客槽深

def rrect4(x, y, w, h, rtl, rtr, rbr, rbl):
    """四個角各自的圓角（SVG 座標，y 往下）。"""
    pts = []
    for (cx, cy, r, a0) in ((x + w - rtr, y + rtr, rtr, -90), (x + w - rbr, y + h - rbr, rbr, 0), (x + rbl, y + h - rbl, rbl, 90), (x + rtl, y + rtl, rtl, 180)):
        if r <= 0:
            pts.append([cx + (0 if a0 in (90, 180) else 0), cy]); continue
        n = max(6, int(math.pi / 2 * r / SEG))
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n); pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
    return np.array(pts)

# ---------- 零件（SVG 座標，y 往下；z＝H−y） ----------
def side_plate(mirror):
    """側板。mirror=True 是左邊那片（雕刻面朝外時，車頭在右邊）。"""
    outer = rrect4(0, 0, L, H, 16, 8, 6, 6)
    holes = [circle(x, H - AZ, (AXLE + AXLE_LOOSE) / 2) for x in AX]
    for (xc, tabs, _) in BULK:
        for (z0, z1) in tabs:
            holes.append(rect(xc - (T + SLOT_C) / 2, H - z1 - .2, T + SLOT_C, (z1 - z0) + .4))
    score = [rrect(18, H - 41, 32, 13, 3), rrect(64, H - 41, 34, 13, 3), rrect(30, H - 15, 56, 11, 3)]
    if mirror:
        outer = mirror_x(outer, L / 2); holes = [mirror_x(h, L / 2) for h in holes]; score = [mirror_x(s, L / 2) for s in score]
    eng = text("我的小車", 5.6, L / 2, H - 6.8)
    return outer, holes, eng, [np.vstack([s, s[:1]]) for s in score]

def bulkhead(xc, tabs, label):
    """隔板（u＝橫跨車寬，SVG x＝u+35）。凸榫穿出側板外，竹籤從孔插過去。"""
    u0 = W_IN / 2; ue = u0 + T + TAB_E; ox = ue
    pts_r = [(ox + u0, H - BZ[1])]
    for (z0, z1) in sorted(tabs, key=lambda z: -z[1]):
        pts_r += [(ox + u0, H - z1), (ox + ue - .6, H - z1), (ox + ue, H - z1 + .6), (ox + ue, H - z0 - .6), (ox + ue - .6, H - z0), (ox + u0, H - z0)]
    pts_r += [(ox + u0, H - BZ[0])]
    left = [(2 * ox - x, y) for (x, y) in pts_r[::-1]]
    top = []
    if label == "中":   # 上方中間開一個乘客槽
        hw = (T + .2) / 2; top = [(ox - hw, H - BZ[1]), (ox - hw, H - BZ[1] + NOTCH), (ox + hw, H - BZ[1] + NOTCH), (ox + hw, H - BZ[1])]
    outer = np.array(left + top + pts_r, float)
    holes = []
    for (z0, z1) in tabs:
        zc = (z0 + z1) / 2; uc = u0 + (T - DELTA) + PIN / 2
        holes += [circle(ox + uc, H - zc, PIN / 2), circle(ox - uc, H - zc, PIN / 2)]
    eng = text(label, 7.0, ox, H - 20)
    return outer, holes, eng

def wheel_disc(kind):
    r = WR - (1.5 if kind == "mid" else 0)
    outer = circle(0, 0, r); holes = [circle(0, 0, (AXLE + AXLE_TIGHT) / 2)]
    eng = []; score = []
    if kind == "out":   # 輪蓋：一圈線和五根輪輻
        score.append(np.vstack([circle(0, 0, 11.5), circle(0, 0, 11.5)[:1]]))
        for k in range(5):
            a = k * 2 * math.pi / 5 - math.pi / 2; c, s = math.cos(a), math.sin(a)
            p = np.array([[-.9, 4.0], [.9, 4.0], [.9, 10.0], [-.9, 10.0]]); R = np.array([[c, -s], [s, c]])
            eng.append(p @ np.array([[0, 1], [1, 0]]) @ R.T)
    return outer, holes, eng, score

def spacer():
    return circle(0, 0, 5.0), [circle(0, 0, (AXLE + AXLE_LOOSE) / 2)]

def tuna():
    """黑鮪魚乘客：橫著坐在車上，下方的腳插進中間隔板的槽。"""
    ctl = [(-24, 52), (-18, 56.5), (-6, 60.5), (8, 59.5), (15, 55.5), (19, 54.5), (25, 61.5), (23, 53), (25, 44.5), (19, 51.5), (15, 50.5), (8, 46.5), (-6, 44.5), (-18, 47.5)]
    P = np.array(ctl, float); n = len(P); out = []
    for i in range(n):   # Catmull-Rom 讓輪廓圓順（尾巴尖角保留）
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        sharp = i in (6, 7, 8)
        for tt in np.linspace(0, 1, 10, endpoint=False):
            if sharp: out.append(p1 + (p2 - p1) * tt); continue
            t2, t3 = tt * tt, tt * tt * tt
            out.append(.5 * ((2 * p1) + (-p0 + p2) * tt + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    body = np.array(out)
    # 下方的腳：x −5～5，z 32～46，中間開槽（z 32～38，寬 T+0.2）
    hw = (T + .2) / 2; zb = BZ[1] - 2 * NOTCH
    # 把腳接到身體下緣：找身體下緣最靠近 x=±5 的點，接成一個外框
    lower = [i for i, p in enumerate(body) if p[1] < 52]
    iR = min(lower, key=lambda i: abs(body[i][0] - 5)); iL = min(lower, key=lambda i: abs(body[i][0] + 5))
    n_ = len(body); loop = [body[(iL + k) % n_] for k in range((iR - iL) % n_ + 1)]   # 從左腳往頭、背、尾巴繞到右腳
    xr, xl = body[iR][0], body[iL][0]
    foot = [(xr, zb), (hw, zb), (hw, zb + NOTCH), (-hw, zb + NOTCH), (-hw, zb), (xl, zb)]
    outer = np.array(loop + [np.array(p, float) for p in foot], float)
    svg = np.stack([outer[:, 0] + 26, 64 - outer[:, 1]], 1)
    eye = circle(-15 + 26, 64 - 53.5, 1.4)
    gill = np.array([[-11 + 26, 64 - 57.5], [-9.5 + 26, 64 - 53], [-11 + 26, 64 - 48.5]])
    line = np.array([[-8 + 26, 64 - 52.5], [2 + 26, 64 - 53.5], [12 + 26, 64 - 53]])
    return svg, [], [eye], [gill, line]

# ---------- 排版 ----------
def sheets():
    body = Sheet(300, 210, "全木小車 車身（3 mm）")
    o, hs, e, sc = side_plate(True); body.part("左側板", o, hs, e, sc, dx=4, dy=4)
    o, hs, e, sc = side_plate(False); body.part("右側板", o, hs, e, sc, dx=4 + L + 6, dy=4)
    x = 4.0
    for (xc, tabs, lab) in BULK:
        o, hs, e = bulkhead(xc, tabs, lab); body.part(f"{lab}隔板", o, hs, e, dx=x, dy=4 + H + 6); x += o[:, 0].max() + 4
    o, hs, e, sc = tuna(); body.part("黑鮪魚乘客", o, hs, e, sc, dx=x + 2, dy=4 + H + 2)
    xs = x + 2
    for k in range(4):
        o, hs = spacer(); body.part(f"墊片{k+1}", o, hs, dx=xs + 6 + k * 12, dy=4 + H + 6 + 36)
    wheels = Sheet(300, 210, "全木小車 輪子（3 mm，4 個輪子，每個 3 片）")
    k = 0
    for kind in ("out", "mid", "in"):
        for j in range(4):
            o, hs, e, sc = wheel_disc(kind); col, row = k % 6, k // 6
            wheels.part(f"輪子{kind}{j+1}", o, hs, e, sc, dx=4 + WR + col * (2 * WR + 3), dy=4 + WR + row * (2 * WR + 3)); k += 1
    return body.fit(), wheels.fit()

def standoff_sides():
    """螺絲柱版側板：不用隔板，用 M3×40 母母螺絲柱撐開（車寬 40）。"""
    sh = Sheet(300, 120, "全木小車 螺絲柱版側板（M3×40 螺絲柱 3 支、M3 螺絲 6 顆）")
    for i, mir in enumerate((True, False)):
        outer = rrect4(0, 0, L, H, 16, 8, 6, 6)
        holes = [circle(x, H - AZ, (AXLE + AXLE_LOOSE) / 2) for x in AX] + [circle(12, H - 37, 1.7), circle(58, H - 24, 1.7), circle(104, H - 37, 1.7)]
        if mir: outer = mirror_x(outer, L / 2); holes = [mirror_x(h, L / 2) for h in holes]
        sh.part("側板", outer, holes, text("我的小車", 5.6, L / 2, H - 6.8), dx=4 + i * (L + 6), dy=4)
    return sh.fit()

# ---------- 3D：組起來的樣子（給預覽圖） ----------
def plate3d(outer, holes, t, O, ex, ey, en, eng=(), score=(), res=.1):
    """把平面零件變成板子：正反面各一個大四邊形（用平面圖遮掉孔），側面照輪廓做。雕刻面在 +en。"""
    lo = outer.min(0) - .5; hi = outer.max(0) + .5
    nx, ny = int((hi[0] - lo[0]) / res) + 1, int((hi[1] - lo[1]) / res) + 1
    def raster(loops, evenodd=False):
        acc = np.zeros((ny, nx), bool)
        for l in loops:
            m = Image.new("1", (nx, ny), 0); ImageDraw.Draw(m).polygon([tuple((q - lo) / res) for q in l], fill=1)
            acc = acc ^ np.asarray(m, bool) if evenodd else (acc | np.asarray(m, bool))
        return acc
    inside = raster([outer]) & ~raster(holes)
    engm = raster(list(eng), True) if len(eng) else None
    scm = None
    if len(score):
        im = Image.new("1", (nx, ny), 0); dr = ImageDraw.Draw(im)
        for l in score: dr.line([tuple((q - lo) / res) for q in l], fill=1, width=max(1, int(.35 / res)))
        scm = np.asarray(im, bool)
    O, ex, ey, en = map(np.asarray, (O, ex, ey, en))
    P = lambda u, v, s: O + ex * u + ey * v + en * s
    c = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
    tris = []; dec = []
    for s in (0.0, -t):
        q = [P(u, v, s) for (u, v) in c]; tris += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]; dec += [1, 1]
    for loop in [outer] + list(holes):
        for i in range(len(loop)):
            a, b = loop[i], loop[(i + 1) % len(loop)]
            p1, p2, p3, p4 = P(*a, 0), P(*b, 0), P(*b, -t), P(*a, -t); tris += [[p1, p2, p3], [p1, p3, p4]]; dec += [0, 0]
    decal = dict(inside=inside, eng=engm, score=scm, o=O + ex * lo[0] + ey * lo[1], ex=ex, ey=ey, res=res, n=en)
    cc = outer.mean(0); sh = [[P(*cc, -t / 2), P(*outer[i], -t / 2), P(*outer[(i + 1) % len(outer)], -t / 2)] for i in range(len(outer))]
    decal["shadow"] = np.array(sh)
    return np.array(tris), np.array(dec), decal

def cylinder(p0, p1, r, n=28):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float); ax = p1 - p0; ax /= np.linalg.norm(ax)
    a = np.cross(ax, [0, 0, 1.0]) if abs(ax[2]) < .9 else np.cross(ax, [1.0, 0, 0]); a /= np.linalg.norm(a); b = np.cross(ax, a)
    ring = lambda p: [p + r * (math.cos(t) * a + math.sin(t) * b) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]
    A, B = ring(p0), ring(p1); T_ = []
    for i in range(n):
        j = (i + 1) % n; T_ += [[A[i], A[j], B[j]], [A[i], B[j], B[i]], [p0, A[j], A[i]], [p1, B[i], B[j]]]
    return np.array(T_)

def torus_y(cx, y, cz, R, r, n=96, m=12):
    T_ = []
    def P(i, j):
        a = 2 * math.pi * i / n; b = 2 * math.pi * j / m; rr = R + r * math.cos(b)
        return np.array([cx + rr * math.cos(a), y + r * math.sin(b), cz + rr * math.sin(a)])
    for i in range(n):
        for j in range(m):
            a, b, c, d = P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1); T_ += [[a, b, c], [a, c, d]]
    return np.array(T_)

def assembly():
    import scene
    items = []   # (tris, tone, decal or None)
    def add_plate(outer, holes, O, ex, ey, en, eng=(), score=(), tone="wood"):
        tr, dec, d = plate3d(np.asarray(outer), [np.asarray(h) for h in holes], T, O, ex, ey, en, eng, score); items.append((tr, tone, dec, d))
    # 側板：左（+y）用鏡像那片，右（−y）用正的那片。SVG (sx, sy) → 車座標
    o, hs, e, sc = side_plate(True)
    add_plate(o, hs, (L, W_IN / 2 + T, H), (-1, 0, 0), (0, 0, -1), (0, 1, 0), e, sc)
    o, hs, e, sc = side_plate(False)
    add_plate(o, hs, (0, -(W_IN / 2 + T), H), (1, 0, 0), (0, 0, -1), (0, -1, 0), e, sc)
    ue = W_IN / 2 + T + TAB_E
    for (xc, tabs, lab) in BULK:
        o, hs, e = bulkhead(xc, tabs, lab)
        add_plate(o, hs, (xc - T / 2, ue, H), (0, -1, 0), (0, 0, -1), (-1, 0, 0), e)
        for (z0, z1) in tabs:   # 竹籤
            zc = (z0 + z1) / 2; yc = W_IN / 2 + (T - DELTA) + PIN / 2
            for sgn in (1, -1): items.append((cylinder((xc - 7, sgn * yc, zc), (xc + 7, sgn * yc, zc), 1.5), "stick", None, None))
    # 黑鮪魚：在 y=0 的平面，雕刻面朝 −y（相機那一側）
    o, hs, e, sc = tuna()
    add_plate(o, hs, (58 - 26, T / 2, 64), (1, 0, 0), (0, 0, -1), (0, 1, 0), e, sc)
    # 輪軸、墊片、輪子、橡皮筋
    for xa in AX:
        items.append((cylinder((xa, -41, AZ), (xa, 41, AZ), AXLE / 2), "stick", None, None))
        for sgn in (1, -1):
            yb = W_IN / 2 + T
            o, hs = spacer()
            add_plate(o, hs, (xa, sgn * (yb + 3), AZ), (1, 0, 0), (0, 0, -1), (0, sgn, 0))
            for li, kind in enumerate(("in", "mid", "out")):
                o, hs, e, sc = wheel_disc(kind)
                yface = sgn * (yb + 3 + 3 * (li + 1))
                ex_ = (1, 0, 0) if sgn < 0 else (-1, 0, 0)
                add_plate(o, hs, (xa, yface, AZ), ex_, (0, 0, -1), (0, sgn, 0), e if kind == "out" else (), sc if kind == "out" else ())
            items.append((torus_y(xa, sgn * (yb + 3 + 4.5), AZ, WR - .9, .95), "rubber", None, None))
    tris = []; tones = []; part = []; tri_dec = []; decs = []
    for pid, (tr, tone, dec, d) in enumerate(items):
        tris.append(tr); tones += [tone] * len(tr); part += [pid] * len(tr)
        if d is None: tri_dec += [-1] * len(tr)
        else:
            decs.append(d); di = len(decs) - 1; tri_dec += [di if f else -1 for f in dec]
    tris = np.concatenate(tris); cols = np.array([scene.hexc(scene.COL.get(t, "#cccccc")) for t in tones])
    sh = np.concatenate([d["shadow"] for d in decs] + [tr for (tr, tone, dec, d) in items if d is None])
    z0 = tris[..., 2].min(); tris[..., 2] -= z0; sh[..., 2] -= z0
    for d in decs: d["o"] = d["o"] - np.array([0, 0, z0])
    return tris, cols, tones, np.array(part), decs, np.array(tri_dec), sh

def render(out, az=math.pi - .5, el=.75, rot=math.pi + .35):
    import scene
    tris, cols, tones, part, decs, tri_dec, sh = assembly()
    c = tris.reshape(-1, 3).mean(0); ca, sa = math.cos(rot), math.sin(rot)
    def R(A):
        A = A.copy(); X = A[..., 0] - c[0]; Y = A[..., 1] - c[1]; A[..., 0] = c[0] + X * ca - Y * sa; A[..., 1] = c[1] + X * sa + Y * ca; return A
    tris = R(tris); sh = R(sh)
    for d in decs:
        for k in ("o",):
            v = d[k] - c; d[k] = np.array([c[0] + v[0] * ca - v[1] * sa, c[1] + v[0] * sa + v[1] * ca, d[k][2]])
        for k in ("ex", "ey", "n"):
            v = d[k]; d[k] = np.array([v[0] * ca - v[1] * sa, v[0] * sa + v[1] * ca, v[2]])
    return scene.render_scene(tris, cols, tones, part, ground="kraft", az=az, el=el, decals=decs, tri_decal=tri_dec, shadow_tris=sh, out=out)

if __name__ == "__main__":
    b, w = sheets(); b.save(f"{OUT}/全木小車_車身_3mm.svg"); w.save(f"{OUT}/全木小車_輪子_3mm.svg")
    b.preview(f"{OUT}/全木小車_車身_預覽.png", 4); w.preview(f"{OUT}/全木小車_輪子_預覽.png", 4)
    s = standoff_sides(); s.save(f"{OUT}/全木小車_螺絲柱版側板_3mm.svg"); s.preview(f"{OUT}/全木小車_螺絲柱版側板_預覽.png", 4)
    print("sheets", (b.w, b.h), (w.w, w.h), (s.w, s.h))
    if "render" in sys.argv: render(f"{OUT}/全木小車_成品.png"); print("render ok")
