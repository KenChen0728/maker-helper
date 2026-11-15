"""一印出來就會滾的車（3D 列印，免組裝）：車頂朝下平放列印，輪子和車身之間留間隙，輪軸末端的凸環卡在輪子裡面不會掉。
用法：python3 roll_car.py [名字]
"""
import math, os, sys, struct, numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)

RW, WW = 11.0, 7.0            # 輪子半徑、寬
BX, BY = 30.0, 13.0           # 車身半長、半寬
TOP = 2 * RW                  # 車頂高度（跟輪子頂端一樣高）
XA = 22.0                     # 輪軸位置 ±22
RA, RB, LB = 2.4, 3.6, 3.0    # 軸半徑、軸端凸環半徑、凸環長
GAP = .6                      # 車身側面和輪子內側的距離
WALL = 1.6                    # 輪子內側那圈擋住凸環的壁厚
CH = dict(front=7.0, back=3.0, side=2.0)   # 車頂四邊的 45° 斜角
RC = 4.0                      # 車身四個直角的圓角

def lathe(profile, center, n=120):
    """profile：(r, y) 封閉多邊形（r≥0），繞 y 軸旋轉；center＝(x, z)。"""
    cx, cz = center; T = []
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    def P(r, y, a): return np.array([cx + r * math.cos(a), y, cz + r * math.sin(a)])
    m = len(profile)
    for i in range(m):
        (r1, y1), (r2, y2) = profile[i], profile[(i + 1) % m]
        if r1 == 0 and r2 == 0: continue
        for k in range(n):
            a, b = ang[k], ang[(k + 1) % n]
            p1, p2, p3, p4 = P(r1, y1, a), P(r2, y2, a), P(r2, y2, b), P(r1, y1, b)
            if r1 == 0: T.append([p1, p2, p3])
            elif r2 == 0: T.append([p1, p2, p4])
            else: T += [[p1, p2, p3], [p1, p3, p4]]
    return np.array(T)

def wheel(side, c):
    """side＝+1（左）／−1（右）。輪子的剖面：外胎有兩條溝，外側有一圈凹的輪蓋線，裡面有放凸環的空腔。"""
    yi = BY + GAP; yo = yi + WW; yc0 = yi + WALL; yc1 = yc0 + LB + 2 * c; ch = .6
    g1, g2 = yi + 2.0, yi + WW - 2.0   # 胎紋溝的中心
    prof = [(RA + c, yi), (RW - ch, yi), (RW, yi + ch),
            (RW, g1 - .5), (RW - .5, g1 - .3), (RW - .5, g1 + .3), (RW, g1 + .5),
            (RW, g2 - .5), (RW - .5, g2 - .3), (RW - .5, g2 + .3), (RW, g2 + .5),
            (RW, yo - ch), (RW - ch, yo), (7.0, yo), (7.0, yo - .5), (5.6, yo - .5), (5.6, yo), (0, yo),
            (0, yc1), (RB + c, yc1), (RB + c, yc0), (RA + c, yc0)]
    if side < 0: prof = [(r, -y) for (r, y) in prof][::-1]
    return prof

def axle(c, lift):
    """整根輪軸（穿過車身），兩端各一個凸環；輪軸中心比輪子中心高 lift（列印時軸靠在孔上）。"""
    yb = BY + GAP + WALL + c; ye = yb + LB; ch = .3
    prof = [(0, -ye), (RB - ch, -ye), (RB, -ye + ch), (RB, -yb), (RA, -yb), (RA, yb), (RB, yb), (RB, ye - ch), (RB - ch, ye), (0, ye)]
    return prof

def body_mesh(zf):
    """車身：圓角長方體，車頂四邊有 45° 斜角（車頂在 TOP）。"""
    def ring(z):
        ins = {k: max(0.0, v - (TOP - z)) for k, v in CH.items()}
        x0, x1 = -BX + ins["back"], BX - ins["front"]; y0, y1 = -BY + ins["side"], BY - ins["side"]; r = RC; pts = []
        for (cx, cy, a0) in ((x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270)):
            for k in range(13):
                a = math.radians(a0 + 90 * k / 12); pts.append([cx + r * math.cos(a), cy + r * math.sin(a), z])
        return np.array(pts)
    zs = [zf] + sorted({TOP - v for v in CH.values()} | {TOP})
    rings = [ring(z) for z in zs]; T = []; n = len(rings[0])
    for A, B in zip(rings[:-1], rings[1:]):
        for i in range(n):
            j = (i + 1) % n; T += [[A[i], A[j], B[j]], [A[i], B[j], B[i]]]
    for R_, top in ((rings[0], False), (rings[-1], True)):
        c = R_.mean(0)
        for i in range(n):
            j = (i + 1) % n; T.append([c, R_[j], R_[i]] if not top else [c, R_[i], R_[j]])
    return np.array(T)

def cyl_x(x0, x1, y, z, r, n=48):
    T = []; a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    A = [np.array([x0, y + r * math.cos(t), z + r * math.sin(t)]) for t in a]; B = [p + [x1 - x0, 0, 0] for p in A]
    c0, c1 = np.array([x0, y, z]), np.array([x1, y, z])
    for i in range(n):
        j = (i + 1) % n; T += [[A[i], A[j], B[j]], [A[i], B[j], B[i]], [c0, A[j], A[i]], [c1, B[i], B[j]]]
    return np.array(T)

def text_mesh(name, side, zc, h=5.0, emb=.5, sink=.4, res=.07):
    """車身側面的凸字（從側面往外凸 emb，往裡埋 sink）。用距離場加 marching cubes 做成封閉的網格。"""
    path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    from fontTools.ttLib import TTCollection
    idx = next(i for i, f in enumerate(TTCollection(path).fonts) if "CJK TC" in (f["name"].getDebugName(1) or ""))
    px = int(h / res * 1.15); font = ImageFont.truetype(path, px, index=idx)
    bb = font.getbbox(name); W, H = bb[2] - bb[0] + 40, bb[3] - bb[1] + 40
    im = Image.new("L", (W, H), 0); ImageDraw.Draw(im).text((20 - bb[0], 20 - bb[1]), name, font=font, fill=255)
    m = np.asarray(im) > 127
    sdf2 = (distance_transform_edt(~m) - distance_transform_edt(m)) * res     # 外面正、裡面負（mm）
    sc = h / ((bb[3] - bb[1]) * res); sdf2 = sdf2 * sc                         # 把字高縮放成 h
    nw = int((emb + sink) / res * .5) + 6; ws = np.linspace(-sink - 2 * res * 2, emb + 2 * res * 2, nw)
    half = (emb + sink) / 2; mid = (emb - sink) / 2
    vol = np.maximum(sdf2[:, :, None], (np.abs(ws - mid) - half)[None, None, :]).astype(np.float32)
    vol = vol[::1, ::1, :]
    v, f, _, _ = marching_cubes(vol, 0.0, spacing=(res * sc, res * sc, ws[1] - ws[0]))
    rows, cols_, w = v[:, 0], v[:, 1], v[:, 2] + ws[0]
    u = cols_ - (W * res * sc) / 2; vv = -(rows - (H * res * sc) / 2)            # 字的 u 往右、v 往上
    if side > 0:   # 左側（+y）從外面看，字往 −x 走
        P = np.stack([-u, np.full_like(u, BY) + w, zc + vv], 1)
    else:          # 右側（−y）從外面看，字往 +x 走
        P = np.stack([u, np.full_like(u, -BY) - w, zc + vv], 1)
    T = P[f]
    if side > 0: T = T[:, ::-1]
    return T

def orient(T):
    """讓三角形的法向量朝外（用體積的正負判斷）。"""
    v = np.einsum("ij,ij->i", T[:, 0], np.cross(T[:, 1], T[:, 2])).sum() / 6
    return T if v > 0 else T[:, ::-1]

def build(c=.4, name="小創客", lift_gap=.05):
    lift = c - lift_gap                          # 用的時候軸比輪子中心高一點；列印時（車頂朝下）軸剛好靠在孔上
    zf = (RW + lift) - RA - 2.2                  # 車底（離地）
    shells = {}
    shells["body"] = orient(body_mesh(zf))
    for xa in (XA, -XA):
        shells[f"axle{xa:+.0f}"] = orient(lathe(axle(c, lift), (xa, RW + lift)))
        for s in (1, -1): shells[f"wheel{xa:+.0f}{s:+d}"] = orient(lathe(wheel(s, c), (xa, RW)))
    for y in (7.0, -7.0): shells[f"light{y:+.0f}"] = orient(cyl_x(BX - .5, BX + .6, y, zf + 4.0, 2.2))
    if name:
        zc = (zf + TOP - CH["side"]) / 2
        shells["nameL"] = orient(text_mesh(name, 1, zc)); shells["nameR"] = orient(text_mesh(name, -1, zc))
    return shells, zf

def to_print(T):
    """車頂朝下：繞 x 軸轉 180°，車頂在 z=0。"""
    P = T.copy(); P[..., 1] = -T[..., 1]; P[..., 2] = TOP - T[..., 2]; return P

def write_stl(path, T):
    T = np.asarray(T, np.float32); n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
    with open(path, "wb") as f:
        f.write(b"maker-helper roll car".ljust(80, b" ")); f.write(struct.pack("<I", len(T)))
        rec = np.zeros(len(T), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]); rec["n"] = n; rec["v"] = T; f.write(rec.tobytes())

def check(shells, c):
    """檢查：輪子和車身（軸）之間最小距離；輪子自己是不是封閉。"""
    from scipy.spatial import cKDTree
    out = {}
    fixed = np.concatenate([v for k, v in shells.items() if not k.startswith("wheel")]).reshape(-1, 3)
    tree = cKDTree(fixed)
    for k, v in shells.items():
        if not k.startswith("wheel"): continue
        d, _ = tree.query(v.reshape(-1, 3)); out[k] = round(float(d.min()), 3)
    return out

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "小創客"
    for c in (.3, .4, .5):
        shells, zf = build(c, name)
        T = np.concatenate([to_print(v) for v in shells.values()])
        write_stl(f"{OUT}/滾滾車_間隙{c:.1f}.stl", T)
        print(c, "tris", len(T), "z", round(float(T[..., 2].min()), 3), round(float(T[..., 2].max()), 3), "floor", round(zf, 2), "gap", check(shells, c), round(os.path.getsize(f"{OUT}/滾滾車_間隙{c:.1f}.stl") / 1e6, 1), "MB")
