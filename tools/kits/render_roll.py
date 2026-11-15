import sys, os, math, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "card-images")); import scene
from roll_car import build
shells, zf = build(.4, "小創客")
T = np.concatenate(list(shells.values())); T[..., 2] -= T[..., 2].min()
a = math.pi + .45; c = T.reshape(-1, 3).mean(0); ca, sa = math.cos(a), math.sin(a)
X = T[..., 0] - c[0]; Y = T[..., 1] - c[1]; T = T.copy(); T[..., 0] = c[0] + X * ca - Y * sa; T[..., 1] = c[1] + X * sa + Y * ca
tones = ["top"] * len(T); cols = np.repeat(scene.hexc(scene.COL["top"])[None], len(T), 0)
scene.render_scene(T, cols, tones, np.zeros(len(T), int), ground="kraft", az=math.pi - .5, el=.62, layer_axis=(0, 0, 1), out=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "滾滾車_成品.png"))
print("ok")
