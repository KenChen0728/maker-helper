"""旋轉迷宮（雷切 3 mm 合板＋3 mm 透明壓克力）。
圓盤迷宮立在往後斜 12° 的架子上。用手轉圓盤，鋼珠靠重力在通道裡滾，從中間走到外圈的出口，掉進下面的小盤子。
- 圓盤：底板＋三層一樣的迷宮層（白膠黏起來）＋壓克力蓋＋前框；外圈 6 根竹籤對齊（位置不對稱，翻面就插不進去）。
- 軸：6 mm 木棒壓在圓盤後面（底板＋兩片輪轂），跟著圓盤轉；穿過立板和兩片軸承環，後面套擋圈。
- 架子：底座、兩片斜撐、立板，全部用梯形榫壓緊（不用插銷：插銷會凸出來刮到圓盤）；前面有接珠子的小盤子。
- 迷宮用程式產生，再用重力模擬確認「只靠轉圓盤一定走得出去」。
"""
import math, os, sys, numpy as np
from PIL import Image, ImageDraw
from kitlib import *
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)

T = 3.0; T_ACR = 3.0; SLOT_C = .3; TAPER = .3
SK_TIGHT, SK_ACR = 3.1, 3.5           # 竹籤孔：木頭卡緊、壓克力放鬆（壓克力硬，太緊會裂）
BALL = 8.0                            # 鋼珠直徑
CH = BALL + 3.0; WALL = 4.0; LAYERS = 3   # 通道寬、牆厚、迷宮幾層（3 層 × 3 mm＝深 9 mm）
R0 = 12.0                             # 中間起點（放珠子的地方）
RINGS = (4, 8, 16)                    # 每一圈分幾格（外圈要是內圈的整數倍）
RIM = 11.0                            # 外框寬（插竹籤）
ROD = 6.0; ROD_TIGHT = -.1; ROD_LOOSE = .3   # 6 mm 木棒：圓盤這邊壓緊、架子這邊會轉
TILT = math.radians(12)               # 架子往後斜
SEED = 59                             # 迷宮的樣子（search_seeds() 挑的：路長 25 格、5 個死路、換方向 6 次）

def ring_a(k): return R0 + WALL + (k - 1) * (CH + WALL)    # 第 k 圈通道內半徑（k＝1、2、3）
N = len(RINGS); R_CH = ring_a(N) + CH; R_DISC = R_CH + RIM
R_PIN = R_CH + RIM / 2; R_HUBPIN = R0 + WALL / 2; R_HUB = 20.0
PIN_ANG = (25, 62, 128, 170, 215, 320)   # 外圈竹籤角度（不對稱：每一層只有一種疊法；正上方留給名字、正下方是出口）

# ---------------- 迷宮 ----------------
def nbrs(c):
    k, j = c; n = RINGS[k - 1]; out = [(k, (j + 1) % n), (k, (j - 1) % n)]
    if k < N: m = RINGS[k] // n; out += [(k + 1, j * m + i) for i in range(m)]
    if k > 1: m = n // RINGS[k - 2]; out.append((k - 1, j // m))
    return out

def gen_maze(seed):
    """隨機深度優先：每一格都連得到、沒有繞圈（完美迷宮），牆就一定連到外框，切出來是一整片。"""
    rng = np.random.default_rng(seed)
    start = (1, int(rng.integers(RINGS[0]))); seen = {start}; stack = [start]; edges = set()
    while stack:
        c = stack[-1]; opts = [d for d in nbrs(c) if d not in seen]
        if not opts: stack.pop(); continue
        d = opts[int(rng.integers(len(opts)))]; seen.add(d); edges.add(frozenset((c, d))); stack.append(d)
    # 起點接第一圈哪一格、出口開在最外圈哪一格：挑路最長的一對
    adj = {}
    for e in edges:
        a, b = tuple(e); adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    def bfs(s):
        d = {s: 0}; q = [s]
        for c in q:
            for x in adj.get(c, []):
                if x not in d: d[x] = d[c] + 1; q.append(x)
        return d
    best = None
    for j in range(RINGS[0]):
        d = bfs((1, j))
        for jj in range(RINGS[-1]):
            L = d[(N, jj)]
            if best is None or L > best[0]: best = (L, j, jj)
    return dict(edges=edges, entry=best[1], exit=best[2], length=best[0] + 1)

ROT0 = [0.0]                          # 整個迷宮轉多少，讓出口在正下方（gen 之後設定）
def ang(k, j): return 2 * math.pi * j / RINGS[k - 1] + ROT0[0]          # 第 k 圈第 j 格的起始角（y 往上、逆時針）
def mid(k, j): return 2 * math.pi * (j + .5) / RINGS[k - 1] + ROT0[0]
def set_home(M): ROT0[0] = -math.pi / 2 - 2 * math.pi * (M["exit"] + .5) / RINGS[-1]

def channel_mask(M, res=.25, ext=None):
    set_home(M)
    """珠子走得到的地方（通道）：在以圓心為原點的格子上算。回傳（mask, xs）。"""
    ext = ext or R_DISC + 2
    xs = np.arange(-ext, ext + res / 2, res); X, Y = np.meshgrid(xs, xs); R = np.hypot(X, Y); A = np.arctan2(Y, X) % (2 * math.pi)
    ch = R <= R0
    def bar(theta, r0, r1, w):    # 沿半徑方向、寬 w 的長條（r0～r1）
        c, s = math.cos(theta), math.sin(theta); u = X * c + Y * s; v = -X * s + Y * c
        return (u >= r0) & (u <= r1) & (np.abs(v) <= w / 2)
    for k in range(1, N + 1):
        a = ring_a(k); ring = (R >= a) & (R <= a + CH)
        n = RINGS[k - 1]
        for j in range(n):                      # 同一圈相鄰兩格之間：沒打通就留一道徑向牆
            if frozenset(((k, j), (k, (j + 1) % n))) not in M["edges"]:
                ring &= ~bar(ang(k, j + 1), a - .5, a + CH + .5, WALL)
        ch |= ring
        for j in range(n):                      # 往內打通的開口（開在這一格的中間）
            if k == 1: op = (j == M["entry"])
            else: op = frozenset(((k, j), (k - 1, j // (n // RINGS[k - 2])))) in M["edges"]
            if op: ch |= bar(mid(k, j), a - WALL - 1, a + 1, CH)
    ch |= bar(mid(N, M["exit"]), R_CH - 1, ext + 1, CH)    # 出口
    return ch, xs

# ---------------- 重力模擬：只靠轉圓盤，珠子走不走得出去 ----------------
def simulate(M, res=.5, step_deg=3):
    from scipy.ndimage import binary_erosion
    ch, xs = channel_mask(M, res)
    yy, xx = np.mgrid[-8:9, -8:9]; rb = (BALL / 2) / res
    free = binary_erosion(ch, structure=(xx ** 2 + yy ** 2) <= rb ** 2)   # 珠子中心走得到的地方
    H, W = free.shape; idx = -np.ones(free.shape, int); P = np.argwhere(free); idx[P[:, 0], P[:, 1]] = np.arange(len(P))
    X = xs[P[:, 1]]; Y = xs[P[:, 0]]; Rr = np.hypot(X, Y)
    nb = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0: continue
            q = P + [dy, dx]; ok = (q[:, 0] >= 0) & (q[:, 0] < H) & (q[:, 1] >= 0) & (q[:, 1] < W)
            t = -np.ones(len(P), int); t[ok] = idx[q[ok, 0], q[ok, 1]]; nb.append(t)
    nb = np.stack(nb, 1)
    nth = 360 // step_deg; sink = []
    for i in range(nth):        # 每個角度：每一點往最低的鄰居滾，最後停在哪
        th = math.radians(i * step_deg); h = X * math.sin(th) + Y * math.cos(th)   # 圓盤轉 th 之後的高度（y 往上）
        hn = np.where(nb >= 0, h[np.clip(nb, 0, None)], np.inf); j = np.argmin(hn, 1)
        nxt = np.where(hn[np.arange(len(P)), j] < h - 1e-9, nb[np.arange(len(P)), j], np.arange(len(P)))
        for _ in range(12): nxt = nxt[nxt]
        sink.append(nxt)
    exitp = Rr > R_CH - .5
    s0 = sink[0][idx[tuple(np.argwhere(free & (np.hypot(*np.meshgrid(xs, xs)) < 2))[0])]]
    # 0-1 BFS：狀態（角度, 珠子停的點, 上一次轉的方向），換方向算一次
    from collections import deque
    start = (0, int(s0), 0); dist = {start: (0, 0)}; prev = {start: None}; dq = deque([start]); goal = None
    while dq:
        st = dq.popleft(); i, p, d = st; rev, tot = dist[st]
        if exitp[p]: goal = st; break
        for nd in (1, -1):
            ni = (i + nd) % nth; np_ = int(sink[ni][p]); ns = (ni, np_, nd)
            cost = (rev + (1 if d not in (0, nd) else 0), tot + 1)
            if ns not in dist or cost < dist[ns]:
                dist[ns] = cost; prev[ns] = st
                (dq.appendleft if cost[0] == rev else dq.append)(ns)
    if goal is None: return dict(ok=False)
    path = []; st = goal
    while st: path.append(st); st = prev[st]
    path = path[::-1]
    return dict(ok=True, turns=dist[goal][0], steps=len(path) - 1, path=[(i * step_deg, (X[p], Y[p])) for i, p, d in path])

def search_seeds(n=60):
    out = []
    for s in range(n):
        M = gen_maze(s); out.append((M["length"], s))
    out.sort(reverse=True); res = []
    for L, s in out[:14]:
        r = simulate(gen_maze(s)); res.append((s, L, r.get("ok"), r.get("turns"), r.get("steps")))
        print(s, L, r.get("ok"), r.get("turns"), r.get("steps"), flush=True)
    return res


def preview2d(M, out, sol=None, px=4):
    """平面預覽：木頭色是牆，白色是通道，紅點是起點，藍線是模擬出來珠子走的路。"""
    ch, xs = channel_mask(M, 1 / px)
    disc = np.hypot(*np.meshgrid(xs, xs)) <= R_DISC
    img = np.full(ch.shape + (3,), 250, np.uint8); img[disc & ~ch] = (222, 196, 152); img[disc & ch] = (255, 255, 255)
    im = Image.fromarray(img[::-1]); dr = ImageDraw.Draw(im); c = len(xs) / 2
    P = lambda x, y: (c + x * px, c - y * px)
    if sol and sol.get("ok"):
        pts = [P(*p) for a, p in sol["path"]]; dr.line(pts, fill=(40, 90, 200), width=2)
    dr.ellipse([P(-2, 2), P(2, -2)], fill=(220, 40, 40))
    im.save(out); return out

# ---------------- 零件（平面座標：圓心在原點，SVG 的 y 往下＝圓盤的「上」是 −y） ----------------
def pol(r, deg): a = math.radians(deg); return (r * math.cos(a), -r * math.sin(a))
LABEL_DEG = 292.0
def rim_pins(d): return [circle(*pol(R_PIN, a), d / 2) for a in PIN_ANG]
def hub_angles(M):
    """輪轂的 3 根竹籤：三等分（翻面也插得進去），整組轉到避開起點的開口。"""
    e = math.degrees(mid(1, M["entry"])) % 360
    for dlt in range(0, 120, 5):
        A = [(a + dlt) % 360 for a in (30, 150, 270)]
        if all(abs((a - e + 180) % 360 - 180) >= 30 for a in A): return A
    return [30, 150, 270]
def hub_pins(M): return [circle(*pol(R_HUBPIN, a), SK_TIGHT / 2) for a in hub_angles(M)]

def rim_label(s, size=3.6):
    x, y = pol(R_PIN, LABEL_DEG); return text(s, size, x, y + size / 2.6)

def maze_layer(M, label="迷宮", res=.1):
    ch, xs = channel_mask(M, res)
    wall = (np.hypot(*np.meshgrid(xs, xs)) <= R_DISC) & ~ch
    img = (np.pad(wall[::-1], 2) * 255).astype(np.uint8)      # 列翻過來：SVG 的 y 往下
    d = Drawing(xs[0], xs[0], xs[-1], xs[-1], res); d.im = Image.fromarray(img)
    L = d.loops(); outer, holes = L[0], L[1:]
    return outer, holes + rim_pins(SK_TIGHT), rim_label(label), []

def floor_disc(M):
    return circle(0, 0, R_DISC), [circle(0, 0, (ROD + ROD_TIGHT) / 2)] + rim_pins(SK_TIGHT) + hub_pins(M), rim_label("底板"), []

def cover():
    eng = text("起點", 3.4, 0, -(R0 + 3.4))
    return circle(0, 0, R_DISC), [circle(0, 0, (BALL + 2) / 2)] + rim_pins(SK_ACR), eng, []

def front_ring():
    d = Drawing(-R_DISC - 1, -R_DISC - 1, R_DISC + 1, R_DISC + 1, .05)
    d.ellipse(0, 0, R_DISC, R_DISC); d.ellipse(0, 0, R_CH + .5, R_CH + .5, fill=0)
    d.rect(-CH / 2, R_CH - 1, CH, RIM + 3, fill=0)              # 出口的缺口（正下方：SVG 的 +y）
    L = d.loops()
    eng = text("我的旋轉迷宮", 5.0, 0, -(R_PIN) + 2.0) + text("出口", 3.6, 15, R_PIN + 1.4)
    return L[0], rim_pins(SK_TIGHT), eng, []

def hub(M): return circle(0, 0, R_HUB), [circle(0, 0, (ROD + ROD_TIGHT) / 2)] + hub_pins(M), [], []

# ---------------- 架子 ----------------
S_AX = 84.0; S_TOP = S_AX + 18; W_U = 60.0; Z_U0 = .65         # 軸在立板上的高度（沿著立板量）
U_TABS = ((12.0, 24.0), (44.0, 56.0)); S_BR = 64.0; Y_BACK = 50.0; E_BR = 3.0
BASE = (-66.0, -36.0, 66.0, 56.0)
R_BOSS = 15.0; R_BOSSPIN = 10.0; BOSS_ANG = (90, 210, 330)
def U_pt(s, d=0.0):
    """立板正面那條斜線往上 s、往後 d 的點（y, z）。"""
    c, sn = math.cos(TILT), math.sin(TILT); return (s * sn + d * c, Z_U0 + s * c - d * sn)

def upright():
    """立板：u＝x+30，v＝S_TOP−s。兩側梯形榫插進斜撐。"""
    tabs = [(e, S_TOP - b, S_TOP - a, "taper") for (a, b) in U_TABS for e in ("L", "R")]
    outer, _ = tabbed_rect_t(0, 0, W_U, S_TOP, tabs)
    holes = [circle(W_U / 2, S_TOP - S_AX, (ROD + ROD_LOOSE) / 2)] + [circle(W_U / 2 + R_BOSSPIN * math.cos(math.radians(a)), S_TOP - S_AX - R_BOSSPIN * math.sin(math.radians(a)), SK_TIGHT / 2) for a in BOSS_ANG]
    return outer, holes, [], []

def tabbed_rect_t(x0, y0, x1, y1, tabs):
    """長方形加梯形榫（只往外凸 T）。tabs：[(邊, a, b, 'taper')]。"""
    out = [(x0, y0)]
    def tp(e, a, b):
        ra, rb, ta, tb = a - TAPER / 2, b + TAPER / 2, a + .2, b - .2; c = .5; d = T
        if e == "T": return [(ra, y0), (ta, y0 - d + c), (ta + c, y0 - d), (tb - c, y0 - d), (tb, y0 - d + c), (rb, y0)]
        if e == "B": return [(rb, y1), (tb, y1 + d - c), (tb - c, y1 + d), (ta + c, y1 + d), (ta, y1 + d - c), (ra, y1)]
        if e == "R": return [(x1, ra), (x1 + d - c, ta), (x1 + d, ta + c), (x1 + d, tb - c), (x1 + d - c, tb), (x1, rb)]
        return [(x0, rb), (x0 - d + c, tb), (x0 - d, tb - c), (x0 - d, ta + c), (x0 - d + c, ta), (x0, ra)]
    for e, a, b, st in sorted([t for t in tabs if t[0] == "T"], key=lambda t: t[1]): out += tp(e, a, b)
    out.append((x1, y0))
    for e, a, b, st in sorted([t for t in tabs if t[0] == "R"], key=lambda t: t[1]): out += tp(e, a, b)
    out.append((x1, y1))
    for e, a, b, st in sorted([t for t in tabs if t[0] == "B"], key=lambda t: -t[1]): out += tp(e, a, b)
    out.append((x0, y1))
    for e, a, b, st in sorted([t for t in tabs if t[0] == "L"], key=lambda t: -t[1]): out += tp(e, a, b)
    return np.array(out, float), []

def boss():   return circle(0, 0, R_BOSS), [circle(0, 0, (ROD + ROD_LOOSE) / 2)] + [circle(R_BOSSPIN * math.cos(math.radians(a)), -R_BOSSPIN * math.sin(math.radians(a)), SK_TIGHT / 2) for a in BOSS_ANG], [], []
def stopper(): return circle(0, 0, 9.0), [circle(0, 0, (ROD + ROD_TIGHT) / 2)], [], []

def brace(side):
    """斜撐（直立，在 y-z 平面）：u＝y+10，v＝70−z。前緣跟著立板斜，兩個梯形槽給立板的榫；底下兩個梯形榫插進底座。"""
    U = lambda y: y + 10; V = lambda z: 70 - z
    yf, zf = U_pt(0, -E_BR); yt, zt = U_pt(S_BR, -E_BR)
    yf = yf + (zf - 0) * math.tan(TILT); zf = 0.0          # 前緣延長到 z＝0
    pts = [(yf, 0), (yt, zt), (yt + 12, zt), (Y_BACK, 6), (Y_BACK, 0)]
    poly_ = [(U(y), V(z)) for (y, z) in pts]
    d = Drawing(-2, -2, 75, 80, .04); d.poly(poly_)
    for (a, b) in ((8.0, 20.0), (32.0, 44.0)):              # 底下的梯形榫（往下凸 T）
        d.poly([(U(a - TAPER / 2), V(0) - .05), (U(a + .2), V(-T)), (U(b - .2), V(-T)), (U(b + TAPER / 2), V(0) - .05)])
    outer = d.loops()[0]
    holes = []
    for (a, b) in U_TABS:      # 立板榫的槽：沿著斜線、寬 T+SLOT_C
        q = [U_pt(a, -SLOT_C / 2), U_pt(b, -SLOT_C / 2), U_pt(b, T + SLOT_C / 2), U_pt(a, T + SLOT_C / 2)]
        holes.append(np.array([(U(y), V(z)) for (y, z) in q]))
    return outer, holes, [], []

def base_plate():
    x0, y0, x1, y1 = BASE; U = lambda x: x - x0; V = lambda y: y1 - y      # 從上面看：u＝x，v＝往前
    outer = rrect(0, 0, x1 - x0, y1 - y0, 6)
    holes = []
    for sx in (1, -1):
        xc = sx * (W_U / 2 + T / 2)
        for (a, b) in ((8.0, 20.0), (32.0, 44.0)): holes.append(rect(U(xc) - (T + SLOT_C) / 2, V(b), T + SLOT_C, b - a))
    for (a, b) in ((-17.0, -7.0), (7.0, 17.0)): holes.append(rect(U(a), V(TRAY_Y0 + T / 2) - (T + SLOT_C) / 2, b - a, T + SLOT_C))
    for sx in (1, -1):
        xc = sx * (TRAY_HX + T / 2)
        for (a, b) in ((-27.0, -19.0), (-13.0, -5.0)): holes.append(rect(U(xc) - (T + SLOT_C) / 2, V(b), T + SLOT_C, b - a))
    eng = text("轉一轉", 4.0, U(-46), V(-27)) + text("轉一轉", 4.0, U(46), V(-27))
    return outer, holes, eng, []

TRAY_Y0 = -34.5; TRAY_HX = 22.0; TRAY_H = 10.0
F_LEN, S_LEN = 2 * (TRAY_HX + T), -2.0 - (TRAY_Y0 + T)
def fence(kind):
    """接珠子的小盤子：前板（u＝x+25）和側板（u＝y+31.5），下面的梯形榫插進底座。"""
    if kind == "front": L, tabs = F_LEN, ((8.0, 18.0), (32.0, 42.0))
    else: L, tabs = S_LEN, ((4.5, 12.5), (18.5, 26.5))
    outer, _ = tabbed_rect_t(0, 0, L, TRAY_H, [("B", a, b, "taper") for (a, b) in tabs])
    return outer, [], [], []

# ---------------- 排版 ----------------
NEST = (("輪轂1", 31, 0), ("輪轂2", -31, 0), ("軸承環1", 0, 37), ("軸承環2", 0, -37), ("擋圈", 0, 0))   # 放在前框中間的洞裡
def all_parts(M):
    P = [("底板", *floor_disc(M))] + [(f"迷宮層{i+1}", *maze_layer(M)) for i in range(LAYERS)]
    P += [("前框", *front_ring()), ("立板", *upright()), ("斜撐1", *brace(1)), ("斜撐2", *brace(-1)), ("底座", *base_plate()),
          ("盤子前板", *fence("front")), ("盤子側板1", *fence("side")), ("盤子側板2", *fence("side"))]
    return P

def nested(M):
    parts = {"輪轂1": hub(M), "輪轂2": hub(M), "軸承環1": boss(), "軸承環2": boss(), "擋圈": stopper()}
    return [(n, parts[n], x, y) for (n, x, y) in NEST]

def place(sh, name, part, x, y, rot=0):
    """把零件轉 rot 度，外框左上角放到 (x, y)。"""
    o, hs, e, sc = part
    R = (lambda P: rot_pts(np.asarray(P, float), rot)) if rot else (lambda P: np.asarray(P, float))
    o2 = R(o); lo = o2.min(0)
    sh.part(name, o2, [R(h) for h in hs], [R(q) for q in e], [R(q) for q in sc], dx=x - lo[0], dy=y - lo[1])
    return o2.max(0) - lo

def sheets(M):
    """合板三張：①底板＋迷宮層 ②迷宮層×2 ③前框（中間放輪轂、軸承環、擋圈）＋架子。壓克力一片。"""
    out = []; D = 2 * R_DISC
    discs = [("底板", floor_disc(M))] + [(f"迷宮層{i+1}", maze_layer(M)) for i in range(LAYERS)]
    for i in range(0, len(discs), 2):
        sh = Sheet(300, 210, f"旋轉迷宮 合板 3 mm（{i // 2 + 1}）")
        for k, (name, part) in enumerate(discs[i:i + 2]): place(sh, name, part, 4 + k * (D + 3), 4)
        out.append((f"合板{i // 2 + 1}", sh.fit()))
    sh = Sheet(300, 210, f"旋轉迷宮 合板 3 mm（{len(out) + 1}）")
    place(sh, "前框", front_ring(), 4, 4)
    for n, part, x, y in nested(M): place(sh, n, part, 4 + R_DISC + x - part[0][:, 0].max() + (part[0][:, 0].max() - part[0][:, 0].min()) / 2 - (part[0][:, 0].max() - part[0][:, 0].min()) / 2, 4 + R_DISC + y - (part[0][:, 1].max() - part[0][:, 1].min()) / 2)
    bw = place(sh, "底座", base_plate(), D + 8, 4)
    uw = place(sh, "立板", upright(), D + 8, 4 + bw[1] + 4)
    place(sh, "斜撐1", brace(1), D + 8 + uw[0] + 4, 4 + bw[1] + 4)
    b2 = place(sh, "斜撐2", brace(-1), 4, D + 8, rot=90)
    fw = place(sh, "盤子前板", fence("front"), 4 + b2[0] + 4, D + 8)
    sw = place(sh, "盤子側板1", fence("side"), 4 + b2[0] + 4, D + 8 + fw[1] + 3)
    place(sh, "盤子側板2", fence("side"), 4 + b2[0] + 4 + sw[0] + 3, D + 8 + fw[1] + 3)
    out.append((f"合板{len(out) + 1}", sh.fit()))
    sh = Sheet(300, 210, "旋轉迷宮 透明壓克力 3 mm"); place(sh, "透明蓋", cover(), 4, 4)
    out.append(("壓克力", sh.fit()))
    return out

# ---------------- 3D ----------------
def sphere(c, r, n=18):
    c = np.asarray(c, float); T_ = []
    th = np.linspace(0, math.pi, n // 2 + 1); ph = np.linspace(0, 2 * math.pi, n + 1)
    P = lambda a, b: c + r * np.array([math.sin(a) * math.cos(b), math.sin(a) * math.sin(b), math.cos(a)])
    for i in range(len(th) - 1):
        for j in range(n):
            a, b, cc, d = P(th[i], ph[j]), P(th[i + 1], ph[j]), P(th[i + 1], ph[j + 1]), P(th[i], ph[j + 1])
            T_ += [[a, b, cc], [a, cc, d]]
    return np.array(T_)

def assembly(M, phi=0.0, ball=None):
    """phi：圓盤從正面看逆時針轉幾度（弧度）。ball：珠子在圓盤座標的位置（x, y 往上），None＝放在起點。"""
    sys.path.insert(0, os.path.join(HERE, "..", "card-images"))
    from wood_car import plate3d, cylinder
    import scene
    c, sn = math.cos(TILT), math.sin(TILT)
    exw = np.array([1.0, 0, 0]); upw = np.array([0, sn, c]); n = np.array([0, -c, sn])
    yA, zA = U_pt(S_AX); A = np.array([0, yA, zA])
    exr = math.cos(phi) * exw + math.sin(phi) * upw; upr = -math.sin(phi) * exw + math.cos(phi) * upw
    items = []
    def add(part, O, ex, ey, en, tone="wood", t=T, edges_only=False):
        o, hs, e, sc = part
        tr, dec, d = plate3d(np.asarray(o), [np.asarray(h) for h in hs], t, O, ex, ey, en, e, sc)
        if edges_only: items.append((tr[dec == 0], tone, None, None))
        else: items.append((tr, tone, dec, d))
    # 圓盤（跟著轉）
    for f, part in ((3.2, hub(M)), (6.2, hub(M)), (9.2, floor_disc(M))): add(part, A + n * f, exr, -upr, n)
    ml = maze_layer(M)
    for i in range(LAYERS): add(ml, A + n * (12.2 + 3 * i), exr, -upr, n)
    add(cover(), A + n * 21.2, exr, -upr, n, tone="acrylic", edges_only=True)
    add(front_ring(), A + n * 24.2, exr, -upr, n)
    for a in PIN_ANG:
        q = R_PIN * (math.cos(math.radians(a)) * exr + math.sin(math.radians(a)) * upr)
        items.append((cylinder(A + q + n * 6.4, A + q + n * 24.0, 1.5, 12), "stick", None, None))
    if ball is None: ball = (0.0, -(R0 - BALL / 2))
    bp = A + n * (9.2 + BALL / 2 + .05) + ball[0] * exr + ball[1] * upr
    items.append((sphere(bp, BALL / 2), "metal", None, None))
    # 軸、立板、軸承環、擋圈
    items.append((cylinder(A - n * 13.0, A + n * 9.2, ROD / 2, 20), "stick", None, None))
    add(upright(), np.array([-W_U / 2, *U_pt(S_TOP)]), exw, -upw, n)
    add(boss(), A - n * 3.0, exw, -upw, n); add(boss(), A - n * 6.0, exw, -upw, n); add(stopper(), A - n * 9.3, exw, -upw, n)
    for a in BOSS_ANG:
        q = R_BOSSPIN * (math.cos(math.radians(a)) * exw + math.sin(math.radians(a)) * upw)
        items.append((cylinder(A + q - n * 8.8, A + q - n * .2, 1.5, 12), "stick", None, None))
    for x0 in (W_U / 2 + T, -W_U / 2): add(brace(1), np.array([x0, -10.0, 70.0]), (0, 1, 0), (0, 0, -1), (1, 0, 0))
    add(base_plate(), np.array([BASE[0], BASE[3], 0.0]), (1, 0, 0), (0, -1, 0), (0, 0, 1))
    add(fence("front"), np.array([-F_LEN / 2, TRAY_Y0, TRAY_H]), (1, 0, 0), (0, 0, -1), (0, -1, 0))
    for x0 in (TRAY_HX + T, -TRAY_HX): add(fence("side"), np.array([x0, TRAY_Y0 + T, TRAY_H]), (0, 1, 0), (0, 0, -1), (1, 0, 0))
    tris = []; tones = []; part = []; tri_dec = []; decs = []
    for pid, (tr, tone, dec, d) in enumerate(items):
        tris.append(tr); tones += [tone] * len(tr); part += [pid] * len(tr)
        if d is None: tri_dec += [-1] * len(tr)
        else: decs.append(d); tri_dec += [len(decs) - 1 if f else -1 for f in dec]
    scene.COL.setdefault("acrylic", "#cfe2ea")
    tris = np.concatenate(tris); cols = np.array([scene.hexc(scene.COL.get(t, "#cccccc")) for t in tones])
    sh = np.concatenate([d["shadow"] for d in decs] + [tr for (tr, tone, dec, d) in items if d is None and tone != "acrylic"])
    return tris, cols, tones, np.array(part), decs, np.array(tri_dec), sh

def render(M, out, phi=0.0, ball=None, az=math.pi - .38, el=.42, W=960, H=660, cam_fix=None):
    sys.path.insert(0, os.path.join(HERE, "..", "card-images"))
    import scene
    tris, cols, tones, part, decs, tri_dec, sh = assembly(M, phi, ball)
    return scene.render_scene(tris, cols, tones, part, ground="kraft", az=az, el=el, decals=decs, tri_decal=tri_dec, shadow_tris=sh, W=W, H=H, out=out, cam_fix=cam_fix)

def check(M):
    """組裝間隙：斜撐前緣到底板背面、盤子到圓盤最低點、底座夠不夠寬。"""
    c, sn = math.cos(TILT), math.sin(TILT); yA, zA = U_pt(S_AX)
    low = zA + 9.2 * sn - R_DISC * c - 6.2 * 0           # 圓盤（底板背面）最低點的高度
    ring_low = zA + 24.2 * sn - R_DISC * c
    ball_exit = (yA - 13.2 * c - R_DISC * sn, zA + 13.2 * sn - R_DISC * c)
    return {"圓盤最低離底座": round(min(low, ring_low), 1), "盤子高": TRAY_H, "珠子出口（y, z）": tuple(round(v, 1) for v in ball_exit),
            "盤子範圍 y": (TRAY_Y0 + T, -2.0), "斜撐到底板背面": round(6.2 - E_BR, 1)}

if __name__ == "__main__":
    M = gen_maze(SEED); r = simulate(M)
    print("迷宮", SEED, "路長", M["length"], "走得出去", r["ok"], "換方向", r["turns"], "轉幾圈", round(r["steps"] * 3 / 360, 1))
    print(check(M))
    for name, sh in sheets(M):
        sh.save(f"{OUT}/旋轉迷宮_{name}.svg"); sh.preview(f"{OUT}/旋轉迷宮_{name}_預覽.png", 3); print("sheet", name, sh.w, sh.h, len(sh.parts))
    preview2d(M, f"{OUT}/旋轉迷宮_走法.png", r, px=4)

def solve_gif(M, out, sol=None, px=3, every=5):
    """正面看圓盤怎麼轉、珠子怎麼走（照重力模擬的結果）。"""
    sol = sol or simulate(M)
    ch, xs = channel_mask(M, 1 / px); R = np.hypot(*np.meshgrid(xs, xs))
    img = np.zeros(ch.shape + (4,), np.uint8)
    img[(R <= R_DISC) & ~ch] = (214, 178, 122, 255); img[(R <= R_DISC) & ch] = (247, 241, 230, 255)
    img[(R <= R_CH + .5) & (R > R_CH - .5) & ~ch] = (196, 158, 104, 255)
    disc = Image.fromarray(img[::-1]); c = disc.width / 2
    frames = []; path = sol["path"]; idx = list(range(0, len(path), every)) + [len(path) - 1]
    idx = [0] * 6 + idx + [len(path) - 1] * 10
    for i in idx:
        a, (x, y) = path[i]; t = math.radians(a)
        bg = Image.new("RGB", disc.size, (250, 246, 238)); rd = disc.rotate(a, resample=Image.BICUBIC)
        bg.paste(rd, (0, 0), rd); dr = ImageDraw.Draw(bg)
        X, Y = x * math.cos(t) - y * math.sin(t), x * math.sin(t) + y * math.cos(t)
        bx, by = c + X * px, c - Y * px; r = BALL / 2 * px
        dr.ellipse([bx - r, by - r, bx + r, by + r], fill=(120, 126, 136), outline=(70, 74, 80), width=2)
        dr.ellipse([bx - r * .45, by - r * .55, bx - r * .05, by - r * .15], fill=(235, 238, 242))
        frames.append(bg.convert("P", palette=Image.ADAPTIVE, colors=64))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=[90] * len(frames), loop=0, optimize=True)
    return out
