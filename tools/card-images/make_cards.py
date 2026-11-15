"""重做作品卡片圖（works/*.webp）：打開創客小幫手，取出每個作品的真實幾何，放在牛皮紙桌面上算成圖。

用法（在 maker-helper 專案根目錄）：
  pip install playwright numpy pillow scipy && python3 -m playwright install chromium
  python3 tools/card-images/make_cards.py                 # 9 個作品全部重做
  python3 tools/card-images/make_cards.py charm spin      # 只做其中幾個
新作品：在 POSE 加一行（平放或組裝、轉幾度），範例圖用 64-kid.js 的手繪圖。
"""
import sys, os, json, math, struct, base64, subprocess, time, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.dirname(__file__))
import scene

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))   # maker-helper 專案根目錄（就是網站）
OUT = os.path.join(ROOT, "works")
AZ, EL = math.pi - .5, .75          # 所有卡片同一個角度
# 每個作品怎麼擺：flat=平放的零件（buildMeshes），否則用組裝好的樣子（ASM）；rot=繞垂直軸轉幾度（弧度），讓圖案比較正
POSE = {"stand": dict(rot=.3), "engrave": dict(rot=.35), "charm": dict(flat=True, rot=.35, ring=True), "stampW": dict(),
        "mold": dict(rot=.2), "thin": dict(flat=True, rot=.35), "spin": dict(), "photoL": dict(rot=.5), "photo3d": dict(rot=.4)}
ALL = list(POSE)

def serve(port):
    return subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

EXPORT_HOW = """(()=>{const c=APP.card, sc=ASM.build(APP.res, APP.params, APP.out, c.use||[]), col=colorsFor(); if(!sc||!sc.objs||!sc.objs.length) return null;
  const out={colors:col, objs:[]};
  for (const o of sc.objs) { const ps=(o.keys&&o.keys.length)? o.keys[o.keys.length-1].pose : (o.built||o.flat||{pos:[0,0,0],q:Q.id()}); const pos=[], tone=[];
    for (let i=0;i<o.tris.length;i++){ for (const v of o.tris[i]) { const r=Q.rot(ps.q, v); pos.push(+(r[0]+ps.pos[0]).toFixed(3), +(r[1]+ps.pos[1]).toFixed(3), +(r[2]+ps.pos[2]).toFixed(3)); } tone.push(o.tones? o.tones[i] : 'base'); }
    out.objs.push({pos, tone}); }
  return out; })()"""
EXPORT_FLAT = """(()=>{const ms=buildMeshes(), col=colorsFor(), out={colors:col, meshes:[]};
  for (const m of ms) { const pos=[], tone=[]; for (let i=0;i<m.tris.length;i++){ const t=m.tris[i]; for (const q of t) pos.push(+(q[0]+(m.dx||0)).toFixed(3), +(q[1]+(m.dy||0)).toFixed(3), +q[2].toFixed(3)); tone.push(m.tones? m.tones[i] : 'base'); }
    out.meshes.push({name:m.name||'', pos, tone}); }
  return out; })()"""

def export_geometry(ids, work, port=8811):
    """在瀏覽器裡打開每個作品（用內建的範例圖），存下組裝和平放兩種幾何；照片作品另外做雕刻圖和透光片 STL。"""
    from playwright.sync_api import sync_playwright
    srv = serve(port); time.sleep(.8); base = f"http://127.0.0.1:{port}/"
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            for cid in [i for i in ids if not i.startswith("photo")]:
                ctx = b.new_context(viewport={"width": 1280, "height": 860}); pg = ctx.new_page()
                pg.route("**/*", lambda r: r.continue_() if r.request.url.startswith(base) else r.abort())
                pg.goto(base); pg.wait_for_timeout(500); pg.evaluate(f"go('{cid}')"); pg.wait_for_timeout(900); pg.evaluate("setStep(2)")
                for _ in range(50):
                    pg.wait_for_timeout(200)
                    if pg.evaluate("!!(APP.res && !APP.res.empty)"): break
                how = pg.evaluate(EXPORT_HOW); flat = pg.evaluate(EXPORT_FLAT)
                if how: json.dump(how, open(f"{work}/{cid}_how.json", "w"))
                json.dump(flat, open(f"{work}/{cid}.json", "w")); ctx.close(); print("geometry", cid, flush=True)
            if any(i.startswith("photo") for i in ids):
                landscape(f"{work}/photo_src.png")
                ctx = b.new_context(viewport={"width": 1280, "height": 860}); pg = ctx.new_page()
                pg.route("**/*", lambda r: r.continue_() if r.request.url.startswith(base) else r.abort())
                pg.goto(base + "photo/?out=laser"); pg.wait_for_timeout(500)
                pg.set_input_files('input[type=file]', f"{work}/photo_src.png"); pg.wait_for_timeout(800)
                pg.evaluate("Object.assign(PHOTO.O,{kind:'plain',shape:'circle',size:50})")
                import re
                svg = pg.evaluate("PHOTO.buildSVG()"); m = re.search(r'href="data:image/png;base64,([^"]+)"', svg)
                open(f"{work}/engrave_src.png", "wb").write(base64.b64decode(m.group(1)))
                pg.evaluate("PHOTO.setOut('litho')"); pg.wait_for_timeout(300)
                b64 = pg.evaluate("""(()=>{Object.assign(PHOTO.O.L,{kind:'stand',shape:'rect',orient:'landscape',size:60,frame:true}); const u=new Uint8Array(PHOTO.stl()); let s=''; for(let i=0;i<u.length;i+=32768) s+=String.fromCharCode.apply(null,u.subarray(i,i+32768)); return btoa(s);})()""")
                open(f"{work}/litho.stl", "wb").write(base64.b64decode(b64)); ctx.close(); print("geometry photo", flush=True)
            b.close()
    finally:
        srv.terminate()

def landscape(path):
    """範例照片：程式畫的虛構海邊夕陽（漁船、島、岬角、海鳥），不是真的照片。"""
    W, H, HY, SX, SY = 1200, 900, 540, 800, 470; rng = np.random.default_rng(7)
    Y, X = np.mgrid[0:H, 0:W].astype(float)
    def fbm2(gw, gh, octs, seed):
        r = np.random.default_rng(seed); out = np.zeros((H, W)); tot = 0
        for o in range(octs):
            a = r.uniform(0, 1, (gh * 2 ** o, gw * 2 ** o)); out += np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC), float) / 255 * .5 ** o; tot += .5 ** o
        return out / tot
    def fbm1(n0, octs, seed):
        r = np.random.default_rng(seed); out = np.zeros(W); x = np.arange(W)
        for o in range(octs):
            n = n0 * 2 ** o; p = r.uniform(-1, 1, n + 2); xs = x / W * n; i = np.floor(xs).astype(int); f = xs - i; f = f * f * (3 - 2 * f); out += (p[i] * (1 - f) + p[i + 1] * f) * .55 ** o
        return out
    def mix(a, b, t): t = np.clip(t, 0, 1)[..., None]; return np.array(a, float) * (1 - t) + np.array(b, float) * t
    t = Y / HY
    sky = np.where((t < .62)[..., None], mix((92, 132, 178), (205, 176, 168), t / .62), mix((205, 176, 168), (253, 204, 146), (t - .62) / .38))
    d = np.hypot(X - SX, Y - SY); glow = np.exp(-(d / 300) ** 2) * .55 + np.exp(-(d / 95) ** 2) * .6
    sky = sky * (1 - np.clip(glow, 0, 1)[..., None] * .6) + np.array([255, 226, 180.]) * np.clip(glow, 0, 1)[..., None] * .6
    cn = fbm2(3, 10, 4, 3); cm = np.clip((cn - .52) / .14, 0, 1) * np.clip((HY * .9 - Y) / 120, 0, 1) * np.clip((Y - 40) / 120, 0, 1)
    ccol = mix((150, 128, 140), (255, 214, 170), np.exp(-(d / 380) ** 2)); sky = sky * (1 - cm[..., None] * .55) + ccol * cm[..., None] * .55
    img = np.where((d < 34)[..., None], np.array([255, 250, 236.]), sky)
    ts = np.clip((Y - HY) / (H - HY), 0, 1); sea = mix((182, 150, 146), (30, 44, 66), ts ** .65); sea *= (.9 + .2 * fbm2(6, 60, 3, 11))[..., None]
    col = np.clip(fbm2(14, 90, 3, 13) * 1.9 - .95, 0, 1) * np.exp(-((X - SX) / (14 + 190 * ts)) ** 2) * (1 - .45 * ts)
    sea = sea * (1 - col[..., None]) + np.array([255, 232, 190.]) * col[..., None]; img = np.where((Y >= HY)[..., None], sea, img)
    env = np.clip(np.sin(np.clip((np.arange(W) - 60) / 520, 0, 1) * np.pi), 0, 1) ** .8; h1 = env * (26 + 30 * (fbm1(3, 6, 21) * .5 + .5))
    img[(Y > HY - h1[None, :]) & (Y < HY)] = np.array([178, 143, 146.])
    x = np.arange(W); r = np.clip((x - 850) / 350, 0, 1); h2 = np.where(x > 850, (r ** .6) * (150 + 30 * fbm1(4, 6, 23)) + np.clip(fbm1(9, 4, 29), 0, 1) * 8, 0)
    m2 = (Y > HY - h2[None, :]) & (Y < HY + 18 * r[None, :] ** 2); img = np.where(m2[..., None], mix((52, 46, 66), (34, 32, 48), np.clip((Y - HY + 150) / 170, 0, 1)), img)
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)); dr = ImageDraw.Draw(pil); bx, by = 470, 646; dk = (24, 26, 38)
    boat = [[(bx - 74, by), (bx + 84, by), (bx + 62, by + 20), (bx - 56, by + 20)], [(bx - 8, by - 30), (bx + 38, by - 30), (bx + 38, by), (bx - 8, by)], [(bx - 2, by - 42), (bx + 26, by - 42), (bx + 26, by - 30), (bx - 2, by - 30)]]
    ref = Image.new("L", (W, H), 0); rd = ImageDraw.Draw(ref)
    for p in boat: dr.polygon(p, fill=dk); rd.polygon([(a, 2 * (by + 20) - b) for a, b in p], fill=150)
    dr.line([(bx - 44, by), (bx - 44, by - 78)], fill=dk, width=4); dr.line([(bx - 44, by - 78), (bx - 74, by - 2)], fill=dk, width=2); dr.line([(bx - 44, by - 78), (bx + 30, by - 42)], fill=dk, width=2)
    rd.line([(bx - 44, by + 40), (bx - 44, by + 100)], fill=110, width=4); ref = ref.filter(ImageFilter.GaussianBlur(3))
    ra = np.asarray(ref, float) / 255 * (.6 + .4 * fbm2(8, 70, 2, 31)); a = np.asarray(pil, float); a = a * (1 - ra[..., None] * .75) + np.array(dk, float) * ra[..., None] * .75
    pil = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)); dr = ImageDraw.Draw(pil)
    for (cx, cy, s) in [(330, 250, 16), (372, 232, 11), (398, 262, 9)]:
        dr.line([(cx - s, cy - s * .45), (cx - s * .45, cy - s * .1), (cx, cy + s * .15), (cx + s * .45, cy - s * .1), (cx + s, cy - s * .45)], fill=(52, 48, 62), width=3, joint="curve")
    pil = pil.filter(ImageFilter.GaussianBlur(.9)); a = np.asarray(pil, float); vg = 1 - .18 * ((X - W / 2) ** 2 / (W / 2) ** 2 + (Y - H / 2) ** 2 / (H / 2) ** 2)
    Image.fromarray(np.clip(a * vg[..., None] + rng.normal(0, 3.2, (H, W, 1)), 0, 255).astype(np.uint8)).save(path)

# ---------- 擺放 ----------
def rotz(T, a, c=None):
    if not a: return T
    c = T.reshape(-1, 3).mean(0) if c is None else c; ca, sa = math.cos(a), math.sin(a); X = T[..., 0] - c[0]; Y = T[..., 1] - c[1]
    T = T.copy(); T[..., 0] = c[0] + X * ca - Y * sa; T[..., 1] = c[1] + X * sa + Y * ca; return T
def invz(a, c):
    ca, sa = math.cos(-a), math.sin(-a)
    def f(P):
        Q = P.copy(); X = P[:, 0] - c[0]; Y = P[:, 1] - c[1]; Q[:, 0] = c[0] + X * ca - Y * sa; Q[:, 1] = c[1] + X * sa + Y * ca; return Q
    return f
def torus(C, u, R=9.0, r=.9, tilt=0.0, n=72, m=12):
    u = np.array([u[0], u[1], 0.0]); u /= np.linalg.norm(u); v = np.array([-u[1], u[0], 0.0]); T = []
    def P(i, j):
        a = 2 * np.pi * i / n; b = 2 * np.pi * j / m; rr = R + r * np.cos(b)
        loc = u * (rr * np.cos(a)) + v * (rr * np.sin(a)) + np.array([0, 0, r * np.sin(b)]); k = loc @ u
        return np.array(C) + loc + np.array([0, 0, -k * math.tan(tilt)])
    for i in range(n):
        for j in range(m):
            a, b, c, d = P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1); T += [[a, b, c], [a, c, d]]
    return np.array(T)
def find_hole(tris, res=.12):
    P = tris.reshape(-1, 3); lo, hi = P.min(0), P.max(0); Wg, Hg = int((hi[0] - lo[0]) / res) + 5, int((hi[1] - lo[1]) / res) + 5
    im = Image.new("L", (Wg, Hg), 0); dr = ImageDraw.Draw(im)
    for t in tris: dr.polygon([((p[0] - lo[0]) / res + 2, (p[1] - lo[1]) / res + 2) for p in t], fill=255)
    ImageDraw.floodfill(im, (0, 0), 128); ys, xs = np.nonzero(np.asarray(im) == 0)
    if not len(xs): return None
    return np.array([xs.mean() * res + lo[0] - 2 * res, ys.mean() * res + lo[1] - 2 * res]), (lo + hi) / 2, hi[2]
def add_ring(T, C, tn, pt):
    hz = find_hole(T)
    if hz is None: return T, C, tn, pt
    H, ctr, ztop = hz; u = H - ctr[:2]; u /= np.linalg.norm(u); Rr = 9.0
    ring = torus(np.array([H[0] + u[0] * Rr, H[1] + u[1] * Rr, .9]), u, Rr, .9, tilt=math.atan((ztop - .9) / (2 * Rr)))
    return (np.concatenate([T, ring]), np.concatenate([C, np.repeat(scene.hexc(scene.COL["metal"])[None], len(ring), 0)]),
            tn + ["metal"] * len(ring), np.concatenate([pt, np.full(len(ring), pt.max() + 1)]))

def render(cid, work, out):
    ps = POSE[cid]
    if cid == "photoL":
        E = np.asarray(Image.open(f"{work}/engrave_src.png").convert("L")); Rd, t, n = 25.0, 3.0, 144
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False); rim = np.stack([Rd * np.cos(ang), Rd * np.sin(ang)], 1); T = []; tn = []
        for i in range(n):
            a, b = rim[i], rim[(i + 1) % n]
            T += [[[0, 0, t], [a[0], a[1], t], [b[0], b[1], t]], [[0, 0, 0], [b[0], b[1], 0], [a[0], a[1], 0]], [[a[0], a[1], 0], [b[0], b[1], 0], [b[0], b[1], t]], [[a[0], a[1], 0], [b[0], b[1], t], [a[0], a[1], t]]]
            tn += ["engr", "wood", "wood", "wood"]
        T = np.array(T, float); C = np.repeat(scene.hexc(scene.COL["wood"])[None], len(T), 0); pt = np.zeros(len(T), int)
        ring = torus(np.array([0.0, Rd - 4.6 + 9.0, .9]), np.array([0.0, 1.0]), 9.0, .9, tilt=math.atan((t - .9) / 18.0))
        T = np.concatenate([T, ring]); C = np.concatenate([C, np.repeat(scene.hexc(scene.COL["metal"])[None], len(ring), 0)]); tn += ["metal"] * len(ring); pt = np.concatenate([pt, np.ones(len(ring), int)])
        a = ps["rot"]; T = rotz(T, a, np.zeros(3))
        return scene.render_scene(T, C, tn, pt, ground="kraft", az=AZ, el=EL, engr=(E, 50.0, (0.0, Rd - 4.6, 2.0)), local=invz(a, np.zeros(3)), out=out)
    if cid == "photo3d":
        b = open(f"{work}/litho.stl", "rb").read(); n = struct.unpack_from("<I", b, 80)[0]
        A = np.frombuffer(b, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84); T = A["v"].astype(float); nr = A["n"]
        front = (nr[:, 1] < .5) & (T[:, :, 1].mean(1) < -.05) & (T[:, :, 2].mean(1) > 3.5)
        tn = ["litho" if f else "base" for f in front]; C = np.repeat(scene.hexc(scene.COL["base"])[None], len(T), 0)
        lo, hi = T.reshape(-1, 3).min(0), T.reshape(-1, 3).max(0); c = np.array([(lo[0] + hi[0]) / 2, 0.0, 0.0]); a = ps["rot"]; T = rotz(T, a, c)
        return scene.render_scene(T, C, tn, np.zeros(len(T), int), ground="kraft", az=AZ, el=EL, glow=(c[0] + 14 * math.sin(a), -14 * math.cos(a), 32.0, .30), local=invz(a, c), out=out)
    flat = ps.get("flat")
    T, C, tn, pt = scene.load(f"{work}/{cid}.json" if flat else f"{work}/{cid}_how.json", "meshes" if flat else "objs")
    if flat: T = T.copy(); T[..., 2] -= T[..., 2].min()
    if ps.get("ring"): T, C, tn, pt = add_ring(T, C, tn, pt)
    T = rotz(T, ps.get("rot", 0))
    return scene.render_scene(T, C, tn, pt, ground="kraft", az=AZ, el=EL, out=out)

if __name__ == "__main__":
    ids = sys.argv[1:] or ALL
    bad = [i for i in ids if i not in POSE]
    if bad: sys.exit(f"不認識的作品：{bad}；可以用的有 {ALL}")
    work = tempfile.mkdtemp(prefix="cards-"); export_geometry(ids, work); os.makedirs(OUT, exist_ok=True)
    for cid in ids:
        png = f"{work}/{cid}.png"; render(cid, work, png)
        Image.open(png).convert("RGB").resize((640, 440), Image.LANCZOS).save(f"{OUT}/{cid}.webp", "WEBP", quality=84, method=6); print("card", cid, flush=True)
