"""黑鮪魚骨架杯墊（雷切 3 mm）。
魚的側面是一片「龍骨」（頭、背、尾鰭，刻上眼睛、脊椎骨和肋骨），四片杯墊是魚身的橫切面，從上面插下去就變成一條立體的黑鮪魚；
要用杯墊時往上抽出來。前後兩片「鰭腳」讓杯墊全部拿走時，魚骨還站得住。
接頭全部是十字插接（槽寬＝板厚＋0.3，槽口倒角），杯墊的缺口只在下緣 14.5 mm，當杯墊用也不礙事。
第 2 片杯墊刻黑鮪魚的橫切面（赤身、中腹、大腹、血合），其他三片留給小朋友畫。
"""
import math, os, sys, numpy as np
from PIL import Image
from kitlib import *
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)

T = 3.0; SLOT = T + .3; FLARE = .8          # 槽寬、槽口每邊多開 0.8（倒角）
SPINE = 14.0                                 # 龍骨下面那條「脊椎」留多高（杯墊的缺口就這麼深＋0.5）
STATIONS = (70.0, 102.0, 134.0, 166.0)       # 四片杯墊在魚身上的位置（從嘴巴量）
GIRTH = .82                                  # 杯墊寬／高（黑鮪魚很圓）
STANDS = (40.0, 188.0)                       # 前後鰭腳的位置
STAND_DEPTH = 10.0                           # 龍骨插進鰭腳多深

def crspline(P, n=12, closed=True):
    P = np.asarray(P, float); m = len(P); out = []
    rng = range(m) if closed else range(m - 1)
    for i in rng:
        p0, p1, p2, p3 = P[(i - 1) % m], P[i], P[(i + 1) % m], P[(i + 2) % m]
        if not closed: p0 = P[max(i - 1, 0)]; p3 = P[min(i + 2, m - 1)]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed: out.append(P[-1])
    return np.array(out)

# 魚的側面（x 從嘴巴往尾巴、z 往上；肚子在 60～180 是平的，放在桌上）
DORSAL = [(0, 40), (8, 53), (22, 67), (42, 81), (66, 91), (92, 97), (120, 99), (148, 97), (172, 90), (192, 78), (210, 64), (224, 56)]
VENTRAL = [(224, 44), (208, 31), (194, 16), (184, 5), (178, 0)]
VENTRAL2 = [(62, 0), (50, 5), (36, 14), (20, 25), (8, 33)]
TAIL = [(230, 58), (244, 74), (256, 92), (266, 106), (262, 92), (254, 70), (250, 50), (254, 30), (262, 8), (266, -0), (256, 8), (244, 26), (230, 42)]

def body_outline():
    top = crspline(DORSAL, 10, closed=False); bot1 = crspline(VENTRAL, 10, closed=False); bot2 = crspline(VENTRAL2, 10, closed=False)
    tail = crspline(TAIL, 8, closed=False)
    return np.vstack([top, tail, bot1, [(178, 0), (62, 0)], bot2])

def dorsal_z(x):
    top = crspline(DORSAL, 40, closed=False); i = np.searchsorted(top[:, 0], x); i = int(np.clip(i, 1, len(top) - 1))
    a, b = top[i - 1], top[i]; return float(a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]))

def ventral_z(x):
    L = np.vstack([crspline(VENTRAL2, 40, closed=False)[::-1], [(62, 0), (178, 0)], crspline(VENTRAL, 40, closed=False)[::-1]])
    i = np.searchsorted(L[:, 0], x); i = int(np.clip(i, 1, len(L) - 1)); a, b = L[i - 1], L[i]
    return float(a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]))

H_ST = [round(dorsal_z(x), 1) for x in STATIONS]

def keel():
    """龍骨：u＝x，v＝110−z。"""
    V = lambda z: 110 - z
    d = Drawing(-2, -2, 270, 116, .04)
    d.poly([(x, V(z)) for (x, z) in body_outline()])
    for x0, h in ((112, 8), (150, 16)):                 # 背鰭（第一背鰭小、第二背鰭像鐮刀）
        z0 = dorsal_z(x0); z1 = dorsal_z(x0 + 14)
        d.poly([(x0 - 2, V(z0 - 3)), (x0 + 4, V(z0 + h)), (x0 + 9, V(z0 + h - 2)), (x0 + 16, V(z1 - 3))])
    for i in range(6):                                  # 背上的小離鰭
        x = 178 + i * 7.5; z = dorsal_z(x)
        d.poly([(x - 2.6, V(z - 2)), (x + 1.2, V(z + 4)), (x + 3.2, V(dorsal_z(x + 3.2) - 2))])
    for i in range(4):                                  # 肚子的小離鰭
        x = 201 + i * 6.5; z = ventral_z(x)
        d.poly([(x - 2.4, V(z + 2)), (x + 1.2, V(z - 3.5)), (x + 3, V(ventral_z(x + 3) + 2))])
    for x in STATIONS:                                  # 杯墊的槽：從背上往下到脊椎上緣
        top = dorsal_z(x) + 6
        d.poly([(x - SLOT / 2, V(SPINE)), (x + SLOT / 2, V(SPINE)), (x + SLOT / 2, V(top - 2.5)), (x + SLOT / 2 + FLARE, V(dorsal_z(x) + .01)),
                (x + SLOT / 2 + FLARE, V(top)), (x - SLOT / 2 - FLARE, V(top)), (x - SLOT / 2 - FLARE, V(dorsal_z(x) + .01)), (x - SLOT / 2, V(top - 2.5))], fill=0)
    for x in STANDS:                                    # 鰭腳的槽：從肚子往上
        zb = ventral_z(x); zt = zb + STAND_DEPTH
        d.poly([(x - SLOT / 2, V(zt)), (x + SLOT / 2, V(zt)), (x + SLOT / 2, V(zb + 2)), (x + SLOT / 2 + FLARE, V(zb - .5)),
                (x + SLOT / 2 + FLARE, V(zb - 4)), (x - SLOT / 2 - FLARE, V(zb - 4)), (x - SLOT / 2 - FLARE, V(zb - .5)), (x - SLOT / 2, V(zb + 2))], fill=0)
    L = d.loops(); outer, holes = L[0], L[1:]
    eng = [circle(26, V(60), 3.6)]                       # 眼睛
    score = [crspline([(44, V(30)), (49, V(48)), (46, V(66)), (40, V(76))], 8, closed=False),          # 鰓蓋
             crspline([(1, V(41)), (12, V(42)), (20, V(45))], 6, closed=False),                          # 嘴
             crspline([(52, V(52)), (70, V(56)), (92, V(52)), (72, V(50)), (52, V(52))], 8, closed=False)]  # 胸鰭
    for i in range(24):                                  # 脊椎骨（一節一節）
        x = 52 + i * 7.0
        if all(abs(x - s) > 4.5 for s in STATIONS) and all(abs(x - s) > 4.5 for s in STANDS) and x < 214:
            zc = max(SPINE / 2, ventral_z(x) + 5); eng.append(rrect(x - 2.4, V(zc + 2.6), 4.8, 5.2, 1.2))
    for i in range(22):                                  # 肋骨（往後斜）
        x = 58 + i * 7.0
        if all(abs(x - s) > 5 for s in STATIONS) and x < 200:
            z0 = SPINE + 2; z1 = min(dorsal_z(x + 6) - 7, 88)
            if z1 > z0 + 8: score.append(np.array([(x, V(z0)), (x + 6, V(z1))]))
    eng += text("東港黑鮪魚", 4.2, 236, V(48))
    return outer, holes, eng, score

def coaster(i):
    """杯墊（魚身橫切面）：u＝y+W/2，v＝H−z。下緣中間有插龍骨的缺口。"""
    H = H_ST[i]; W = GIRTH * H; a, b, p = W / 2, H / 2 + 3, 2.3
    th = np.linspace(0, 2 * math.pi, 240, endpoint=False)
    P = np.stack([a * np.sign(np.cos(th)) * np.abs(np.cos(th)) ** (2 / p), (H / 2 - 3) + b * np.sign(np.sin(th)) * np.abs(np.sin(th)) ** (2 / p)], 1)
    d = Drawing(-2, -2, W + 2, H + 4, .04)
    d.poly([(y + a, H - z) for (y, z) in P])
    d.rect(-2, H, W + 4, 10, fill=0)                     # 下面切平
    sd = SPINE + .5
    d.poly([(a - SLOT / 2, H - sd), (a + SLOT / 2, H - sd), (a + SLOT / 2, H - 2.0), (a + SLOT / 2 + FLARE, H + .01), (a + SLOT / 2 + FLARE, H + 3),
            (a - SLOT / 2 - FLARE, H + 3), (a - SLOT / 2 - FLARE, H + .01), (a - SLOT / 2, H - 2.0)], fill=0)
    L = d.loops(); eng = text(str(i + 1), 3.0, a + 7, H - 3.2); score = []
    if i == 1:   # 黑鮪魚的橫切面
        zc = H * .52; c = (a, H - zc)
        inner = np.stack([.86 * a * np.sign(np.cos(th)) * np.abs(np.cos(th)) ** (2 / p) + a, H - ((H / 2 - 3) + .88 * b * np.sign(np.sin(th)) * np.abs(np.sin(th)) ** (2 / p) + 1.5)], 1)
        score += [np.vstack([inner, inner[:1]]), np.array([(a, H - (H - 7)), (a, H - sd - 3)]), np.array([(a - .86 * a, c[1]), (a + .86 * a, c[1])])]
        eng += [circle(*c, 4.2)]
        for s in (-1, 1):
            x0 = a + s * .80 * a; eng.append(np.array([(x0, c[1] - 4.5), (x0 - s * 9, c[1]), (x0, c[1] + 4.5), (x0 + s * 1.5, c[1])]))
        eng += text("赤身", 4.0, a - 17, c[1] - 14) + text("赤身", 4.0, a + 17, c[1] - 14)
        eng += text("中腹", 4.0, a - 18, c[1] + 12) + text("中腹", 4.0, a + 18, c[1] + 12)
        eng += text("大腹", 4.4, a - 14, H - 17.5) + text("大腹", 4.4, a + 14, H - 17.5)
        eng += text("血合", 2.8, a - .62 * a, c[1] - 6.5) + text("血合", 2.8, a + .62 * a, c[1] - 6.5)
    return L[0], L[1:], eng, score

def stand(x):
    """鰭腳（橫的，放在桌上）：龍骨從上面插進中間的槽。u＝y+50，v＝高−z。"""
    zb = ventral_z(x); zm = zb + STAND_DEPTH; Hs = zb + STAND_DEPTH + 16; w = 46 if x < 100 else 38
    V = lambda z: Hs - z
    d = Drawing(-52, -2, 52, Hs + 4, .04)
    pts = crspline([(-w, 0), (-w + 6, 9), (-14, Hs - 6), (0, Hs), (14, Hs - 6), (w - 6, 9), (w, 0)], 10, closed=False)
    d.poly([(y, V(z)) for (y, z) in pts] + [(w, V(0)), (-w, V(0))])
    d.ellipse(0, V(0), 16, 7, fill=0)                    # 下面挖個拱，變成兩隻腳
    d.poly([(-SLOT / 2, V(zm - .5)), (SLOT / 2, V(zm - .5)), (SLOT / 2, V(Hs - 2)), (SLOT / 2 + FLARE, V(Hs + .01)), (SLOT / 2 + FLARE, V(Hs + 3)),
            (-SLOT / 2 - FLARE, V(Hs + 3)), (-SLOT / 2 - FLARE, V(Hs + .01)), (-SLOT / 2, V(Hs - 2))], fill=0)
    L = d.loops()
    return L[0], L[1:], text("前" if x < 100 else "後", 3.4, w - 12, V(4)), []

def check():
    """每一個十字插接的重疊：兩個槽加起來要剛好等於重疊的高度。"""
    rows = []
    for x, h in zip(STATIONS, H_ST):
        rows.append(("杯墊", x, "龍骨槽底", SPINE, "杯墊槽頂", SPINE + .5, "高", h))
    for x in STANDS:
        zb = ventral_z(x); rows.append(("鰭腳", x, "龍骨槽頂", round(zb + STAND_DEPTH, 1), "鰭腳槽底", round(zb + STAND_DEPTH - .5, 1), "肚子離桌", round(zb, 1)))
    return rows

def all_parts():
    P = [("龍骨", *keel())] + [(f"杯墊{i+1}", *coaster(i)) for i in range(len(STATIONS))]
    P += [("前鰭腳", *stand(STANDS[0])), ("後鰭腳", *stand(STANDS[1]))]
    return P

def sheets():
    """兩張：①骨架（龍骨＋兩片鰭腳，課前切好）②四片杯墊（上課刻小朋友的圖）。"""
    sh = Sheet(300, 210, "黑鮪魚骨架杯墊 骨架 3 mm"); kw = place(sh, "龍骨", keel(), 4, 4)
    fw = place(sh, "前鰭腳", stand(STANDS[0]), 4, 4 + kw[1] + 4); place(sh, "後鰭腳", stand(STANDS[1]), 4 + fw[0] + 6, 4 + kw[1] + 4)
    out = [("骨架", sh.fit())]
    sh = Sheet(300, 210, "黑鮪魚骨架杯墊 杯墊 3 mm"); x = 4
    for i in range(len(STATIONS)):
        w = place(sh, f"杯墊{i+1}", coaster(i), x if i < 3 else 4, 4 if i < 3 else 4 + max(H_ST) + 6); x += w[0] + 4
    out.append(("杯墊", sh.fit()))
    return out

# ---------------- 3D ----------------
def assembly(lift=None):
    """lift：{杯墊編號: 往上抽多高}，拿來畫「抽出來」的樣子。"""
    sys.path.insert(0, os.path.join(HERE, "..", "card-images"))
    from wood_car import plate3d
    import scene
    lift = lift or {}; items = []
    def add(part, O, ex, ey, en):
        o, hs, e, sc = part; tr, dec, d = plate3d(np.asarray(o), [np.asarray(h) for h in hs], T, O, ex, ey, en, e, sc); items.append((tr, "wood", dec, d))
    add(keel(), (0, -T / 2, 110), (1, 0, 0), (0, 0, -1), (0, -1, 0))
    for i, x in enumerate(STATIONS):
        H = H_ST[i]; W = GIRTH * H; dz = lift.get(i, 0.0)
        add(coaster(i), (x - T / 2, W / 2, H + dz), (0, -1, 0), (0, 0, -1), (-1, 0, 0))
    for x in STANDS:
        zb = ventral_z(x); Hs = zb + STAND_DEPTH + 16
        add(stand(x), (x - T / 2, 0, Hs), (0, -1, 0), (0, 0, -1), (-1, 0, 0))
    tris = []; tones = []; part = []; tri_dec = []; decs = []
    for pid, (tr, tone, dec, d) in enumerate(items):
        tris.append(tr); tones += [tone] * len(tr); part += [pid] * len(tr); decs.append(d); tri_dec += [len(decs) - 1 if f else -1 for f in dec]
    tris = np.concatenate(tris); cols = np.array([scene.hexc(scene.COL.get(t, "#cccccc")) for t in tones])
    sh = np.concatenate([d["shadow"] for d in decs])
    return tris, cols, tones, np.array(part), decs, np.array(tri_dec), sh

def render(out, lift=None, az=math.pi + .62, el=.50, W=960, H=660):
    sys.path.insert(0, os.path.join(HERE, "..", "card-images"))
    import scene
    tris, cols, tones, part, decs, tri_dec, sh = assembly(lift)
    return scene.render_scene(tris, cols, tones, part, ground="kraft", az=az, el=el, decals=decs, tri_decal=tri_dec, shadow_tris=sh, W=W, H=H, out=out)

if __name__ == "__main__":
    print("杯墊高", H_ST, "寬", [round(GIRTH * h, 1) for h in H_ST])
    for r in check(): print(r)
    for name, sh in sheets():
        sh.save(f"{OUT}/黑鮪魚骨架杯墊_{name}_3mm.svg"); sh.preview(f"{OUT}/黑鮪魚骨架杯墊_{name}_預覽.png", 3); print("sheet", name, sh.w, sh.h, len(sh.parts))
