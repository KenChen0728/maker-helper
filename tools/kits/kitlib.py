"""雷切套件的小工具：形狀都用折線（mm），輸出 Beam Studio 用的 SVG（紅＝切割、黑填色＝雕刻、藍＝線刻），也能畫預覽圖。"""
import math
import numpy as np
from fontTools.ttLib import TTCollection
from fontTools.pens.basePen import BasePen
from PIL import Image, ImageDraw

SEG = 0.25  # 圓弧每段長度（mm）

def circle(cx, cy, r, n=None):
    n = n or max(24, int(2 * math.pi * r / SEG))
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)

def rect(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)

def rrect(x, y, w, h, r):
    r = min(r, w / 2, h / 2)
    if r <= 0: return rect(x, y, w, h)
    pts = []
    for cx, cy, a0 in ((x + w - r, y + r, -90), (x + w - r, y + h - r, 0), (x + r, y + h - r, 90), (x + r, y + r, 180)):
        n = max(6, int(math.pi / 2 * r / SEG))
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n); pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
    return np.array(pts)

def poly(*pts): return np.array(pts, float)

def move(P, dx, dy): return P + np.array([dx, dy])

def rot(P, deg, cx=0, cy=0):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a); Q = P - [cx, cy]
    return np.stack([cx + Q[:, 0] * c - Q[:, 1] * s, cy + Q[:, 0] * s + Q[:, 1] * c], 1)

def mirror_x(P, x0=0): Q = P.copy(); Q[:, 0] = 2 * x0 - Q[:, 0]; return Q[::-1]

# ---------- 文字轉成折線（思源黑體 繁中） ----------
_FONT = {}
def _font(bold=True):
    key = bold
    if key not in _FONT:
        path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        try: coll = TTCollection(path)
        except Exception: coll = TTCollection("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
        f = next((f for f in coll.fonts if "CJK TC" in (f["name"].getDebugName(1) or "")), coll.fonts[0])
        _FONT[key] = (f, f.getGlyphSet(), f.getBestCmap(), f["head"].unitsPerEm, f["hmtx"])
    return _FONT[key]

class _Flat(BasePen):
    def __init__(self, gs, tf):
        super().__init__(gs); self.loops = []; self.cur = None; self.tf = tf
    def _p(self, p): a, b, c, d, e, f = self.tf; return [a * p[0] + c * p[1] + e, b * p[0] + d * p[1] + f]
    def _moveTo(self, p): self.cur = [self._p(p)]; self.last = p
    def _lineTo(self, p): self.cur.append(self._p(p)); self.last = p
    def _curveToOne(self, p1, p2, p3):
        p0 = self.last
        for t in np.linspace(0, 1, 9)[1:]:
            u = 1 - t; q = [u ** 3 * p0[i] + 3 * u * u * t * p1[i] + 3 * u * t * t * p2[i] + t ** 3 * p3[i] for i in (0, 1)]
            self.cur.append(self._p(q))
        self.last = p3
    def _qCurveToOne(self, p1, p2):
        p0 = self.last
        for t in np.linspace(0, 1, 7)[1:]:
            u = 1 - t; q = [u * u * p0[i] + 2 * u * t * p1[i] + t * t * p2[i] for i in (0, 1)]
            self.cur.append(self._p(q))
        self.last = p2
    def _closePath(self):
        if self.cur and len(self.cur) > 2: self.loops.append(np.array(self.cur))
        self.cur = None
    _endPath = _closePath

def text(s, size, x, y, anchor="middle", bold=True):
    """size＝字高（大約是 em 的 0.88），y＝基線。回傳折線（雕刻用，evenodd 填色）。"""
    f, gs, cmap, upm, hmtx = _font(bold)
    sc = size / (upm * .88)
    w = sum(hmtx[cmap.get(ord(ch), cmap.get(32))][0] for ch in s) * sc
    x0 = x - w / 2 if anchor == "middle" else (x - w if anchor == "end" else x)
    loops = []; cur = x0
    for ch in s:
        g = cmap.get(ord(ch))
        if g is None: cur += size * .5; continue
        pen = _Flat(gs, (sc, 0, 0, -sc, cur, y)); gs[g].draw(pen); loops += pen.loops; cur += hmtx[g][0] * sc
    return loops

def text_width(s, size, bold=True):
    f, gs, cmap, upm, hmtx = _font(bold); sc = size / (upm * .88)
    return sum(hmtx[cmap.get(ord(ch), cmap.get(32))][0] for ch in s) * sc

# ---------- 文件 ----------
class Sheet:
    """一張雷切排版。cut＝切割（紅）、eng＝雕刻（黑填色）、score＝線刻（藍）。"""
    def __init__(self, w, h, title=""):
        self.w, self.h, self.title = w, h, title; self.cut = []; self.eng = []; self.score = []; self.parts = []
    def add_cut(self, *loops): self.cut += [np.asarray(l, float) for l in loops]
    def add_eng(self, *loops): self.eng += [np.asarray(l, float) for l in loops]
    def add_score(self, *lines): self.score += [np.asarray(l, float) for l in lines]
    def part(self, name, outer, holes=(), eng=(), score=(), dx=0, dy=0, rot_deg=0, cx=0, cy=0):
        """加一個零件（外框＋孔），可以平移、旋轉；也記下來給 3D 預覽用。"""
        def T(P):
            P = np.asarray(P, float)
            if rot_deg: P = rot(P, rot_deg, cx, cy)
            return P + [dx, dy]
        o = T(outer); hs = [T(hh) for hh in holes]; es = [T(e) for e in eng]; ss = [T(s) for s in score]
        self.add_cut(o, *hs); self.add_eng(*es); self.add_score(*ss)
        self.parts.append(dict(name=name, outer=o, holes=hs, eng=es))
        return o
    def fit(self, margin=4):
        """把畫布縮到零件的範圍（左上留 margin），在 Beam Studio 比較好擺。"""
        lo, hi = self.bounds(); sh = np.array([margin, margin]) - lo
        for L in (self.cut, self.eng, self.score):
            for i in range(len(L)): L[i] = L[i] + sh
        for p in self.parts:
            p["outer"] = p["outer"] + sh; p["holes"] = [h + sh for h in p["holes"]]; p["eng"] = [e + sh for e in p["eng"]]
        self.w, self.h = float(np.ceil(hi[0] - lo[0] + 2 * margin)), float(np.ceil(hi[1] - lo[1] + 2 * margin)); return self
    def bounds(self):
        P = np.concatenate([l for l in self.cut + self.eng + self.score]) if (self.cut or self.eng) else np.zeros((1, 2))
        return P.min(0), P.max(0)
    def svg(self):
        f = lambda v: f"{v:.3f}".rstrip("0").rstrip(".")
        def d(loops, close=True):
            return "".join("M" + " L".join(f"{f(p[0])} {f(p[1])}" for p in l) + ("Z" if close else "") for l in loops)
        s = f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="{f(self.w)}mm" height="{f(self.h)}mm" viewBox="0 0 {f(self.w)} {f(self.h)}">\n'
        if self.title: s += f"<title>{self.title}</title>\n"
        if self.eng: s += f'<g id="engrave" data-layer="雕刻"><path fill="#000000" fill-rule="evenodd" d="{d(self.eng)}"/></g>\n'
        if self.score: s += f'<g id="score" data-layer="線刻"><path fill="none" stroke="#0000FF" stroke-width="0.1" d="{d(self.score, False)}"/></g>\n'
        if self.cut: s += f'<g id="cut" data-layer="切割"><path fill="none" stroke="#FF0000" stroke-width="0.1" d="{d(self.cut)}"/></g>\n'
        return s + "</svg>\n"
    def save(self, path):
        lo, hi = self.bounds(); assert lo.min() >= -1e-6 and hi[0] <= self.w + 1e-6 and hi[1] <= self.h + 1e-6, (self.title, lo, hi)
        open(path, "w").write(self.svg()); return path
    def preview(self, path, px_per_mm=4, wood=(222, 196, 152), bg=(250, 248, 244)):
        k = px_per_mm; W, H = int(self.w * k), int(self.h * k)
        im = Image.new("RGB", (W, H), bg); dr = ImageDraw.Draw(im)
        # 零件填木頭色（外框），孔挖掉
        for p in self.parts:
            dr.polygon([tuple(q * k) for q in p["outer"]], fill=wood)
            for hh in p["holes"]: dr.polygon([tuple(q * k) for q in hh], fill=bg)
        # evenodd：重疊兩次的區域（例如字的內圈）要挖掉 → 用 XOR 疊
        acc = np.zeros((H, W), np.uint8)
        for l in self.eng:
            m = Image.new("1", (W, H), 0); ImageDraw.Draw(m).polygon([tuple(q * k) for q in l], fill=1); acc ^= np.asarray(m, np.uint8)
        im.paste((70, 40, 20), mask=Image.fromarray((acc * 255).astype(np.uint8)))
        for l in self.score: dr.line([tuple(q * k) for q in l], fill=(40, 70, 200), width=max(1, k // 3))
        for l in self.cut: dr.line([tuple(q * k) for q in np.vstack([l, l[:1]])], fill=(220, 30, 30), width=max(1, k // 3))
        im.save(path); return path

# ---------- 角色造型：用畫的（圓、多邊形、粗線）組起來，再描出外框 ----------
def cross2(a, b):
    a = np.asarray(a); b = np.asarray(b); return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

def area(P):
    P = np.asarray(P); return cross2(P, np.roll(P, -1, 0)).sum() / 2

def rdp(P, eps=.04):
    """Douglas–Peucker 簡化折線（封閉）。"""
    P = np.asarray(P, float)
    if len(P) < 5: return P
    def _r(a, b):
        if b <= a + 1: return [a]
        seg = P[b] - P[a]; L = np.hypot(*seg) or 1e-9
        d = np.abs(cross2(seg, P[a + 1:b] - P[a])) / L; k = int(np.argmax(d))
        if d[k] > eps: m = a + 1 + k; return _r(a, m) + _r(m, b)
        return [a]
    i0 = 0; i1 = int(np.argmax(np.hypot(*(P - P[0]).T)))
    idx = _r(i0, i1) + _r(i1, len(P) - 1) + [len(P) - 1]
    return P[sorted(set(idx))]

class Drawing:
    """在 res mm/格 的畫布上畫形狀（座標 mm，y 往下），最後描出外框和孔。"""
    def __init__(self, x0, y0, x1, y1, res=.04):
        self.x0, self.y0, self.res = x0, y0, res
        self.W, self.H = int((x1 - x0) / res) + 4, int((y1 - y0) / res) + 4
        self.im = Image.new("L", (self.W, self.H), 0); self.dr = ImageDraw.Draw(self.im)
    def _p(self, x, y): return ((x - self.x0) / self.res + 2, (y - self.y0) / self.res + 2)
    def poly(self, pts, fill=255): self.dr.polygon([self._p(*p) for p in pts], fill=fill)
    def ellipse(self, cx, cy, rx, ry, fill=255): self.dr.ellipse([self._p(cx - rx, cy - ry), self._p(cx + rx, cy + ry)], fill=fill)
    def line(self, pts, w, fill=255):
        q = [self._p(*p) for p in pts]; wpx = max(1, int(w / self.res))
        self.dr.line(q, fill=fill, width=wpx, joint="curve")
        for p in q: self.dr.ellipse([p[0] - wpx / 2, p[1] - wpx / 2, p[0] + wpx / 2, p[1] + wpx / 2], fill=fill)
    def rect(self, x, y, w, h, fill=255): self.poly([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], fill)
    def loops(self, eps=.03):
        from skimage.measure import find_contours
        A = np.asarray(self.im, float) / 255
        cs = find_contours(np.pad(A, 1), .5)
        out = []
        for c in cs:
            P = np.stack([(c[:, 1] - 3) * self.res + self.x0, (c[:, 0] - 3) * self.res + self.y0], 1)
            if len(P) > 8 and abs(area(P[:-1])) > .5: out.append(rdp(P[:-1], eps))
        out.sort(key=lambda P: -abs(area(P)))
        return out   # 第一個是外框，其他是孔

def plus_hole(cx, cy, aw, at, rot=0):
    """十字孔（套在十字軸上的輪轂）：兩條 aw×at 的長方形交叉。"""
    a, t = aw / 2, at / 2
    P = np.array([(t, a), (-t, a), (-t, t), (-a, t), (-a, -t), (-t, -t), (-t, -a), (t, -a), (t, -t), (a, -t), (a, t), (t, t)], float)
    return rot_pts(P, rot) + [cx, cy]

def rot_pts(P, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a); return np.stack([P[:, 0] * c - P[:, 1] * s, P[:, 0] * s + P[:, 1] * c], 1)

def pack(items, W=290, H=200, gap=3, margin=4):
    """items：[(name, outer, holes, eng, score), ...]（各自的座標）。架子式排版，放不下就換下一張。回傳 [[(item, dx, dy), ...], ...]"""
    boxes = []
    for it in items:
        o = it[1]; lo, hi = o.min(0), o.max(0); boxes.append((it, lo, hi - lo))
    boxes.sort(key=lambda b: -b[2][1])
    sheets = []; cur = []; x = y = margin; rowh = 0
    for it, lo, sz in boxes:
        if x + sz[0] > W - margin:
            x = margin; y += rowh + gap; rowh = 0
        if y + sz[1] > H - margin:
            sheets.append(cur); cur = []; x = y = margin; rowh = 0
        cur.append((it, x - lo[0], y - lo[1])); x += sz[0] + gap; rowh = max(rowh, sz[1])
    if cur: sheets.append(cur)
    return sheets
