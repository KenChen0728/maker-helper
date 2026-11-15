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
