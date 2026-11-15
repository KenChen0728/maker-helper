"""桌面場景算圖：作品放在牛皮紙或切割墊上，暖色燈光、統一角度、接觸陰影、雷切焦邊、3D 列印層紋。"""
import json, math, numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter

def hexc(h): h = h.lstrip('#'); return np.array([int(h[i:i+2], 16) / 255 for i in (0, 2, 4)])
def norm(v): return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-9)
COL = {"base": "#f4f0e8", "top": "#e07a36", "wood": "#e6c893", "burn": "#4a2a12", "metal": "#c3c7cc", "stick": "#d8bb88",
       "paper": "#fbf8f1", "ink": "#b8423a", "clay": "#d08f6d", "rubber": "#2e2c2a"}
WOODISH = {"wood", "engr"}
LAST_CAM = None
PLA = {"base", "top", "clay"}

# ---------- 雜訊 ----------
def _h(a, b, seed):
    s = np.sin(a * 127.1 + b * 311.7 + seed * 74.7) * 43758.5453
    return s - np.floor(s)
def vnoise(x, y, seed=0.0):
    xi, yi = np.floor(x), np.floor(y); fx, fy = x - xi, y - yi
    u, v = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b, c, d = _h(xi, yi, seed), _h(xi + 1, yi, seed), _h(xi, yi + 1, seed), _h(xi + 1, yi + 1, seed)
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v
def fbm(x, y, octs=4, seed=0.0):
    s, amp, tot = 0.0, .5, 0.0
    for o in range(octs):
        s = s + amp * vnoise(x, y, seed + o * 13.1); tot += amp; x = x * 2.03 + 17.1; y = y * 2.03 + 5.3; amp *= .5
    return s / tot
def rot2(X, Y, th): return X * math.cos(th) + Y * math.sin(th), -X * math.sin(th) + Y * math.cos(th)

# ---------- 桌面 ----------
def kraft(X, Y, pf):
    base = np.array([.735, .595, .425])
    m = fbm(X / 26, Y / 26, 4, 1.0) - .5; m2 = fbm(X / 6, Y / 6, 3, 2.0) - .5
    fade = np.clip(1 - pf / .35, 0, 1)
    u, v = rot2(X, Y, .5); f1 = vnoise(u / .18, v / 1.6, 7.0) - .5
    u, v = rot2(X, Y, 2.1); f2 = vnoise(u / .2, v / 1.2, 11.0) - .5
    sp = vnoise(X / .4, Y / .4, 23.0); sp2 = vnoise(X / .9 + 3, Y / .9, 29.0)
    k = 1 + .12 * m + .05 * m2 + fade * (.03 * f1 + .025 * f2) - .10 * fade * np.clip((sp - .94) / .06, 0, 1) + .06 * fade * np.clip((sp2 - .95) / .05, 0, 1)
    return base[None, :] * k[:, None]
def mat(X, Y, pf):
    base = np.array([.19, .40, .335])
    n = fbm(X / 30, Y / 30, 3, 3.0) - .5
    col = base[None, :] * (1 + .07 * n)[:, None]
    def lines(c, period, width, off):
        d = np.abs(((c + off) / period + .5) % 1.0 - .5) * period
        p = np.maximum(pf, 1e-3); return np.clip((width / 2 + p / 2 - d) / p, 0, 1) * np.clip(width / p, 0, 1) ** .5
    minor = np.maximum(lines(X, 10, .26, 3.7), lines(Y, 10, .26, -6.1))
    major = np.maximum(lines(X, 50, .55, 23.7), lines(Y, 50, .55, 13.9))
    d45 = np.abs(X - Y + 31.0) / math.sqrt(2); p = np.maximum(pf, 1e-3); diag = np.clip((.15 + p / 2 - d45) / p, 0, 1)
    a = np.maximum(np.maximum(.42 * minor, .72 * major), .3 * diag)
    lc = np.array([.80, .91, .85])
    return col * (1 - a[:, None]) + lc[None, :] * a[:, None]
GROUNDS = {"kraft": kraft, "mat": mat}

# ---------- 每個零件的板面方向（面積最大的那個法向量） ----------
def part_axes(tris, part):
    nb = np.zeros((len(tris), 3))
    cr = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]); area = np.linalg.norm(cr, axis=1); n = cr / area.clip(1e-9)[:, None]
    for pid in np.unique(part):
        m = part == pid; M = (n[m, :, None] * n[m, None, :] * area[m, None, None]).sum(0)
        w, V = np.linalg.eigh(M); ax = V[:, -1]
        if ax[2] < 0: ax = -ax
        nb[m] = ax
    return nb

def render_scene(tris, cols, tones, part=None, ground="kraft", az=math.pi - .5, el=.75, W=960, H=660, SS=2, fov=24,
                 fillw=.74, fillh=.70, warm=True, engr=None, glow=None, out=None, center_y=.50, local=None, decals=None, tri_decal=None, shadow_tris=None, layer_axis=None, cam_fix=None):
    w, h = W * SS, H * SS
    if part is None: part = np.zeros(len(tris), int)
    lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0); zmin = lo[2]
    fwd = np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])
    f = 1 / math.tan(math.radians(fov) / 2)
    tgt = (lo + hi) / 2; dist = np.linalg.norm(hi - lo) / 2 / math.sin(math.radians(fov) / 2)
    def cam(tgt, dist):
        eye = tgt + fwd * dist; zc = norm(eye - tgt); xc = norm(np.cross(np.array([0, 0, 1.0]), zc)); yc = np.cross(zc, xc)
        return eye, xc, yc, zc
    def proj(P, eye, xc, yc, zc):
        d = P - eye; x = d @ xc; y = d @ yc; z = d @ zc
        s = f / (-z); return np.stack([w / 2 + x * s * (h / 2), h / 2 - y * s * (h / 2), -z], -1)
    # 讓作品在畫面裡的大小、位置一致（迭代幾次）；動畫時用第一張的相機（cam_fix）
    global LAST_CAM
    for _ in range(0 if cam_fix else 4):
        eye, xc, yc, zc = cam(tgt, dist); V = proj(tris.reshape(-1, 3), eye, xc, yc, zc)
        bx0, by0 = V[:, 0].min(), V[:, 1].min(); bx1, by1 = V[:, 0].max(), V[:, 1].max()
        k = max((bx1 - bx0) / (fillw * w), (by1 - by0) / (fillh * h)); dist *= k ** .9
        eye, xc, yc, zc = cam(tgt, dist); V = proj(tris.reshape(-1, 3), eye, xc, yc, zc)
        cx, cy = (V[:, 0].min() + V[:, 0].max()) / 2, (V[:, 1].min() + V[:, 1].max()) / 2
        depth = np.median(V[:, 2]); px = depth / (f * h / 2)
        tgt = tgt + xc * (cx - w / 2) * px - yc * (cy - h * center_y) * px
    if cam_fix: tgt, dist = np.asarray(cam_fix[0]), cam_fix[1]
    LAST_CAM = (tgt.copy(), dist)
    eye, xc, yc, zc = cam(tgt, dist)
    V = proj(tris, eye, xc, yc, zc)
    n = norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]))
    view = norm(eye - tris.mean(1)); flip = (n * view).sum(-1) < 0; n[flip] *= -1
    nb = part_axes(tris, part)
    if layer_axis is not None: nb[:] = np.asarray(layer_axis, float)   # 3D 列印：層紋跟著列印方向
    nd = np.abs((n * nb).sum(-1))
    # 燈光：左前上方的主光（暖）、右邊的補光（冷）、環境光
    L1 = norm(np.array([-.45, -.55, .9])); L2 = norm(np.array([.75, -.15, .35]))
    c1 = np.array([1.0, .93, .82]) if warm else np.ones(3); c2 = np.array([.80, .87, 1.0]) if warm else np.ones(3)
    camb = np.array([1.0, .97, .93]) if warm else np.ones(3)
    def light(nn, vv, spec_k=.10, spec_p=24):
        d1 = np.clip(nn @ L1, 0, 1)[:, None] * c1 * .64; d2 = np.clip(nn @ L2, 0, 1)[:, None] * c2 * .20
        am = (.30 + .14 * (nn[:, 2] * .5 + .5))[:, None] * camb
        hv = norm(L1 + vv); sp = spec_k * np.clip((nn * hv).sum(-1), 0, 1) ** spec_p
        return am + d1 + d2, sp[:, None] * c1
    img = np.zeros((h, w, 3)); zb = np.full((h, w), np.inf); cov = np.zeros((h, w), bool); emis = np.zeros((h, w), bool)
    order = np.argsort(-V[:, :, 2].mean(1))
    for i in order:
        a, b, c = V[i]
        x0 = int(max(0, math.floor(min(a[0], b[0], c[0])))); x1 = int(min(w - 1, math.ceil(max(a[0], b[0], c[0]))))
        y0 = int(max(0, math.floor(min(a[1], b[1], c[1])))); y1 = int(min(h - 1, math.ceil(max(a[1], b[1], c[1]))))
        if x1 < x0 or y1 < y0: continue
        X, Y = np.meshgrid(np.arange(x0, x1 + 1) + .5, np.arange(y0, y1 + 1) + .5)
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-12: continue
        l1 = ((b[1] - c[1]) * (X - c[0]) + (c[0] - b[0]) * (Y - c[1])) / den; l2 = ((c[1] - a[1]) * (X - c[0]) + (a[0] - c[0]) * (Y - c[1])) / den; l3 = 1 - l1 - l2
        m = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        if not m.any(): continue
        z = l1 * a[2] + l2 * b[2] + l3 * c[2]; sub = zb[y0:y1 + 1, x0:x1 + 1]
        dc = decals[tri_decal[i]] if (tri_decal is not None and tri_decal[i] >= 0) else None
        if dc is not None:   # 板子的正反面：用平面圖決定哪裡是板子、哪裡是孔
            Pc = l1[m, None] * tris[i, 0] + l2[m, None] * tris[i, 1] + l3[m, None] * tris[i, 2]; q = Pc - dc["o"]
            ui = np.clip((q @ dc["ex"] / dc["res"]).astype(int), 0, dc["inside"].shape[1] - 1); vi = np.clip((q @ dc["ey"] / dc["res"]).astype(int), 0, dc["inside"].shape[0] - 1)
            keep = dc["inside"][vi, ui]; idx = np.flatnonzero(m); m = np.zeros_like(m); m.flat[idx[keep]] = True
        m &= z < sub
        if not m.any(): continue
        P = l1[m, None] * tris[i, 0] + l2[m, None] * tris[i, 1] + l3[m, None] * tris[i, 2]; N = len(P)
        t = tones[i]; base = cols[i].copy(); spk, spp = .10, 24
        e1 = norm(np.cross(nb[i], np.array([0.0, 0.31, .95]) if abs(nb[i][2]) < .9 else np.array([1.0, 0, 0]))); e2 = np.cross(nb[i], e1)
        U, Vv = P @ e1, P @ e2
        col = np.broadcast_to(base, (N, 3)).copy()
        if t in WOODISH or t == "burn":
            if nd[i] < .35:      # 雷切的切口：焦黑色
                g = vnoise(U / .8, (P @ nb[i]) / .3, 5.0) - .5
                col = np.broadcast_to(hexc("#6e4524"), (N, 3)) * (1 + .12 * g)[:, None]; spk = .03
            else:                # 板面：木紋
                g = .045 * np.sin(U * .30 + 1.8 * np.sin(Vv * .05) + 3 * vnoise(U / 9, Vv / 30, 2.0)) + .02 * (vnoise(U / .6, Vv / 6, 3.0) - .5)
                col *= (1 + g)[:, None]
                if t == "burn": col = np.broadcast_to(hexc("#4a2a12"), (N, 3)) * (1 + .5 * g + .08 * (vnoise(U / .25, Vv / .25, 9.0) - .5))[:, None]; spk = .02
                if dc is not None and dc.get("eng") is not None and (n[i] @ dc["n"]) > .5:   # 雕刻：燒成深褐色
                    q = P - dc["o"]; ui = np.clip((q @ dc["ex"] / dc["res"]).astype(int), 0, dc["eng"].shape[1] - 1); vi = np.clip((q @ dc["ey"] / dc["res"]).astype(int), 0, dc["eng"].shape[0] - 1)
                    bn = dc["eng"][vi, ui]; col[bn] = hexc("#4a2a12") * (1 + .5 * g[bn, None])
                    if dc.get("score") is not None:
                        sc_ = dc["score"][vi, ui]; col[sc_] = hexc("#6b4223") * (1 + .3 * g[sc_, None])
                if t == "engr" and engr is not None:
                    E, mm, (hx, hy, hr) = engr; iw, ih = E.shape[1], E.shape[0]; Q = local(P) if local else P
                    uu = np.clip(((Q[:, 0] + mm / 2) / mm * iw).astype(int), 0, iw - 1); vv = np.clip(((mm / 2 - Q[:, 1]) / mm * ih).astype(int), 0, ih - 1)
                    bn = E[vv, uu] < 128; col[bn] = hexc("#4a2a12") * (1 + .5 * g[bn, None])
                    hole = (Q[:, 0] - hx) ** 2 + (Q[:, 1] - hy) ** 2 < hr * hr; col[hole] = np.array([.16, .10, .05])
        elif t in PLA:
            if nd[i] < .35:      # 側面：一層一層的紋路（0.2 mm）
                col *= (1 + .045 * np.sin((P @ nb[i]) / .2 * 2 * math.pi))[:, None]
            col *= (1 + .025 * (vnoise(U / .4, Vv / .4, 4.0) - .5))[:, None]; spk, spp = .07, 18
        elif t == "stick":
            u2 = P @ norm(np.array([0, 0, 1.0])); col *= (1 + .05 * np.sin(np.arctan2(P[:, 1], P[:, 0]) * 9 + u2 * .2) + .03 * (vnoise(P[:, 0] * 3, u2 / 4, 6.0) - .5))[:, None]
        elif t == "paper":
            col *= (1 + .025 * (fbm(P[:, 0] / .8, P[:, 1] / .8, 2, 8.0) - .5))[:, None]; spk = .02
        elif t == "ink":
            col *= (1 + .18 * (fbm(P[:, 0] / .7, P[:, 1] / .7, 3, 12.0) - .5))[:, None]; spk = .02
        elif t == "metal":
            spk, spp = .55, 40
        if t == "litho":
            Q = local(P) if local else P
            dd = np.clip(-Q[:, 1], .6, 3.2); T = np.exp(-1.1 * (dd - .8)) ** .5
            col = np.array([1.0, .90, .70])[None, :] * (.10 + 1.05 * T[:, None])
            img[y0:y1 + 1, x0:x1 + 1][m] = col; sub[m] = z[m]; cov[y0:y1 + 1, x0:x1 + 1][m] = True; emis[y0:y1 + 1, x0:x1 + 1][m] = True; continue
        sh, sp = light(np.broadcast_to(n[i], (N, 3)), np.broadcast_to(view[i], (N, 3)), spk, spp)
        col = col * sh + sp
        sub[m] = z[m]; img[y0:y1 + 1, x0:x1 + 1][m] = col; cov[y0:y1 + 1, x0:x1 + 1][m] = True; emis[y0:y1 + 1, x0:x1 + 1][m] = False
    # 作品上的凹處變暗（深度的反銳化）
    D = np.where(cov, zb, 0.0).astype(np.float32); cf = cov.astype(np.float32)
    ao = np.zeros((h, w))
    for s_, k_ in ((6 * SS, .5), (18 * SS, .5)):
        avg = gaussian_filter(D, s_) / np.maximum(gaussian_filter(cf, s_), 1e-4)
        ao += k_ * np.clip((np.where(cov, zb, 0) - avg) / 1.2, 0, 1)
    am_ = cov & ~emis; img[am_] *= (1 - .55 * np.clip(ao[am_], 0, 1))[:, None]
    # 桌面
    ys, xs = np.mgrid[0:h, 0:w]; xcam = (xs + .5 - w / 2) / (h / 2) / f; ycam = -(ys + .5 - h / 2) / (h / 2) / f
    dirs = xcam[..., None] * xc + ycam[..., None] * yc - zc
    tg = (zmin - eye[2]) / np.minimum(dirs[..., 2], -1e-6)
    G = eye + dirs * tg[..., None]
    dl2 = (dirs ** 2).sum(-1); pf = tg * dl2 * (2 / (h * f)) / np.maximum(np.abs(dirs[..., 2]), .05)
    gm = ~cov
    gcol = np.zeros((h, w, 3))
    gcol[gm] = GROUNDS[ground](G[gm][:, 0], G[gm][:, 1], pf[gm])
    up = np.array([[0, 0, 1.0]]); gsh, _ = light(up, up, 0, 1); gcol *= gsh[0]
    # 陰影：主光投影（柔）＋正下方的接觸陰影（緊）
    ST = tris if shadow_tris is None else shadow_tris
    def shadow_mask(Ld, blur):
        Pt = ST.reshape(-1, 3).copy(); tt = (Pt[:, 2] - zmin) / Ld[2]; Pt = Pt - Ld * tt[:, None]; Pt[:, 2] = zmin
        S2 = proj(Pt.reshape(-1, 3, 3), eye, xc, yc, zc); im = Image.new("L", (w, h), 0); dr = ImageDraw.Draw(im)
        for tri in S2: dr.polygon([(float(p[0]), float(p[1])) for p in tri], fill=255)
        return np.asarray(im.filter(ImageFilter.GaussianBlur(blur))) / 255
    mk = shadow_mask(L1, 9 * SS); mc = shadow_mask(np.array([0, 0, 1.0]), 2.2 * SS)
    gcol *= (1 - .34 * mk - .38 * mc)[..., None]
    if glow is not None:   # 透光片後面的燈照在桌上的暖光
        gx, gy, gr, gs = glow; dd = np.hypot(G[..., 0] - gx, G[..., 1] - gy); gcol += (np.array([1.0, .78, .45]) * gs)[None, None, :] * np.exp(-(dd / gr) ** 2)[..., None] * gm[..., None]
    # 景深：離焦點越遠越模糊
    tf = np.median(zb[cov]) if cov.any() else np.median(tg)
    gb = np.dstack([gaussian_filter(gcol[..., k].astype(np.float32), 5 * SS) for k in range(3)])
    wdf = np.clip(np.abs(tg - tf) / (.55 * tf), 0, 1)[..., None]
    gcol = gcol * (1 - wdf) + gb * wdf
    out_img = np.where(cov[..., None], img, gcol)
    if emis.any():   # 透光片發亮：亮的地方往外暈開一點
        em = np.where(emis[..., None], img, 0).astype(np.float32)
        out_img = out_img + .30 * np.dstack([gaussian_filter(em[..., k], 9 * SS) for k in range(3)])
    # 暗角
    rr = ((xs - w / 2) / (w / 2)) ** 2 + ((ys - h * center_y) / (h / 2)) ** 2
    out_img *= (1 - .16 * np.clip(rr, 0, 1.6))[..., None]
    res = Image.fromarray((np.clip(out_img, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
    if out: res.save(out)
    return res

def load(path, key, col=None):
    d = json.load(open(path)); C = d["colors"]; tris = []; cols = []; tones = []; part = []
    for pid, m in enumerate(d[key]):
        P = np.array(m["pos"], float).reshape(-1, 3, 3); tris.append(P); tones += m["tone"]; part += [pid] * len(m["tone"])
        cols += [hexc((col or COL).get(t, C.get(t, "#cccccc"))) for t in m["tone"]]
    return np.concatenate(tris), np.array(cols), tones, np.array(part)
