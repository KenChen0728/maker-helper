"""印章十字握把（雷切 3 mm）：給 3D 列印印章面用。
兩片握把十字插接（A 從上面開槽、B 從下面開槽），下面的十字穿過底座的十字孔、跟底座底面切齊；3D 列印的印章面用白膠黏在底座下面。
底座的十字孔長邊比握把窄 0.1（壓緊），短邊多 0.3（吃板厚誤差）。一張板排 8 組，全班 12 組切兩張。
"""
import math, os, sys, numpy as np
from kitlib import *
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)

T = 3.0; SLOT = T + .3; FLARE = .6
NECK = 20.0; GRIP = 26.0; H = 56.0          # 脖子寬、握的地方寬、高
BASE = 36.0; HOLE_FIT = -.1                  # 底座邊長、十字孔長邊比握把少多少
LANYARD = 4.0                                # A 片頂端的繩孔

def outline():
    """握把側面（u＝x+GRIP/2，v＝H−z）：下面 0～12 是脖子（穿進底座），往上變寬，頂端圓的。"""
    pts = [(-NECK / 2, 0), (NECK / 2, 0), (NECK / 2, 10)]
    for t in np.linspace(0, 1, 10)[1:]:          # 脖子到握把：外凸的弧
        pts.append((NECK / 2 + (GRIP - NECK) / 2 * math.sin(t * math.pi / 2), 10 + 12 * t))
    pts.append((GRIP / 2, H - GRIP / 2))
    for a in np.linspace(0, math.pi, 24)[1:-1]: pts.append((GRIP / 2 * math.cos(a), H - GRIP / 2 + GRIP / 2 * math.sin(a)))
    pts.append((-GRIP / 2, H - GRIP / 2))
    for t in np.linspace(1, 0, 10)[:-1]: pts.append((-NECK / 2 - (GRIP - NECK) / 2 * math.sin(t * math.pi / 2), 10 + 12 * t))
    pts.append((-NECK / 2, 10))
    return np.array([(x + GRIP / 2, H - z) for (x, z) in pts])

def plate(kind):
    o = outline(); c = GRIP / 2; mid = H / 2
    d = Drawing(-2, -2, GRIP + 2, H + 2, .03); d.poly([tuple(p) for p in o])
    if kind == "A":   # 從上面往下開槽到一半
        d.poly([(c - SLOT / 2, H - mid + .3), (c + SLOT / 2, H - mid + .3), (c + SLOT / 2, 2.0), (c + SLOT / 2 + FLARE, -.1), (c + SLOT / 2 + FLARE, -3),
                (c - SLOT / 2 - FLARE, -3), (c - SLOT / 2 - FLARE, -.1), (c - SLOT / 2, 2.0)], fill=0)
    else:             # 從下面往上開槽到一半
        d.poly([(c - SLOT / 2, H - mid - .3), (c + SLOT / 2, H - mid - .3), (c + SLOT / 2, H - 2.0), (c + SLOT / 2 + FLARE, H + .1), (c + SLOT / 2 + FLARE, H + 3),
                (c - SLOT / 2 - FLARE, H + 3), (c - SLOT / 2 - FLARE, H + .1), (c - SLOT / 2, H - 2.0)], fill=0)
    L = d.loops(); holes = L[1:]
    eng = []
    if kind == "A":
        holes.append(circle(c - 7.5, 8.5, LANYARD / 2))
        eng += text("A", 3.2, c + 7.0, H - 15.5) + text("印", 6.0, c - 7.0, H - 20.0)
    else:
        eng += text("B", 3.2, c + 7.0, 13.5) + text("章", 6.0, c - 7.0, H - 20.0)
    return L[0], holes, eng, []

def base():
    """底座：中間十字孔給握把的脖子（長邊 NECK+HOLE_FIT、短邊 T+0.3）。"""
    o = rrect(0, 0, BASE, BASE, 4)
    return o, [plus_hole(BASE / 2, BASE / 2, NECK + HOLE_FIT, T + .3)], text("底座", 3.0, BASE / 2, BASE - 4.0), []

def sheets():
    out = []
    sh = Sheet(300, 210, "印章十字握把 一組 3 mm"); w = place(sh, "握把A", plate("A"), 4, 4); w2 = place(sh, "握把B", plate("B"), 4 + w[0] + 4, 4)
    place(sh, "底座", base(), 4 + w[0] + 4 + w2[0] + 4, 4); out.append(("一組", sh.fit()))
    items = []
    for k in range(8): items += [(f"握把A{k+1}", *plate("A")), (f"握把B{k+1}", *plate("B")), (f"底座{k+1}", *base())]
    packs = pack(items, W=292, H=202)
    for n, pk in enumerate(packs):
        sh = Sheet(300, 210, f"印章十字握把 8 組 3 mm")
        for (name, o, hs, e, sc), dx, dy in pk: sh.part(name, o, hs, e, sc, dx=dx, dy=dy)
        out.append(("8組" + ("" if len(packs) == 1 else str(n + 1)), sh.fit()))
    return out

if __name__ == "__main__":
    for name, sh in sheets():
        sh.save(f"{OUT}/印章十字握把_{name}_3mm.svg"); sh.preview(f"{OUT}/印章十字握把_{name}_預覽.png", 5); print("sheet", name, sh.w, sh.h, len(sh.parts))

def render(out, W=960, H_=660):
    sys.path.insert(0, os.path.join(HERE, "..", "card-images"))
    from wood_car import plate3d
    import scene
    items = []
    def add(part, O, ex, ey, en, tone="wood", t=T):
        o, hs, e, sc = part; tr, dec, d = plate3d(np.asarray(o), [np.asarray(h) for h in hs], t, O, ex, ey, en, e, sc); items.append((tr, tone, dec, d))
    z0 = 4.0                                     # 3D 印章面 4 mm，握把從它上面開始
    add(plate("A"), (-GRIP / 2, -T / 2, z0 + H), (1, 0, 0), (0, 0, -1), (0, -1, 0))
    add(plate("B"), (-T / 2, GRIP / 2, z0 + H), (0, -1, 0), (0, 0, -1), (-1, 0, 0))
    add(base(), (-BASE / 2, BASE / 2, z0 + T), (1, 0, 0), (0, -1, 0), (0, 0, 1))
    face = (rrect(0, 0, BASE, BASE, 4), [], [], [])
    add(face, (-BASE / 2, BASE / 2, z0), (1, 0, 0), (0, -1, 0), (0, 0, 1), tone="top", t=z0)
    tris = []; tones = []; part = []; tri_dec = []; decs = []
    for pid, (tr, tone, dec, d) in enumerate(items):
        tris.append(tr); tones += [tone] * len(tr); part += [pid] * len(tr); decs.append(d); tri_dec += [len(decs) - 1 if f else -1 for f in dec]
    tris = np.concatenate(tris); cols = np.array([scene.hexc(scene.COL.get(t, "#cccccc")) for t in tones])
    sh = np.concatenate([d["shadow"] for d in decs])
    return scene.render_scene(tris, cols, tones, np.array(part), ground="kraft", az=math.pi - .55, el=.55, decals=decs, tri_decal=np.array(tri_dec), shadow_tris=sh, W=W, H=H_, out=out)
