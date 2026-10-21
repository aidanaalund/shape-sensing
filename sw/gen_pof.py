"""
Synthetic images of a hanging plastic optical fiber (POF), shaped by Kirchhoff/Euler elastica.

Model (planar inextensible elastica, clamped-free, self-weight + intrinsic curvature):
    M(s)   = EI * (theta'(s) - k0)                      bending moment, k0 = spool "memory"
    M'(s)  = -q * (L - s) * cos(theta(s))               moment balance, gravity = image down
    theta(0) = theta0      (clamp angle, measured below horizontal; 90 deg = straight down)
    M(L)   = 0  ->  theta'(L) = k0                      free, moment-free tip
    x' = cos(theta),  y' = -sin(theta)

Non-dimensional (s -> s/L):  theta'' = -alpha (1 - s) cos(theta),   theta'(1) = k0 L
    alpha = q L^3 / EI         gravity-to-stiffness ratio, from real material data

q (weight per length) and EI (bending stiffness) come from real POF data:
    bare PMMA fiber  : E 2.4-3.3 GPa, rho 1190 kg/m^3, d 0.5/0.75/1.0 mm
    jacketed 1 mm POF: PMMA core + PE jacket (OD 2.2 mm, E 0.2-0.5 GPa, rho 930 kg/m^3)
Spool memory: fiber stored on a reel keeps part of the reel curvature (k0 = -f / R_reel).

The renderer is true to scale: the fiber's pixel width is its real diameter at the
image's mm/px, and gravity is always straight down in the image (no camera roll).
Each sample ships with an exact mask, the centerline + theta(s) + curvature, and the
physical parameters.

Usage:
  python gen_pof.py --selftest                 # check the solver against beam theory
  python gen_pof.py --n 9 --out sample         # vertical hanging fibers (default)
  python gen_pof.py --n 500 --pose any         # also sideways clamps
"""
import argparse, json, os
import numpy as np
from scipy.integrate import solve_bvp
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from PIL import Image

G = 9.81


# ----------------------------------------------------------------------------- physics
def fiber_properties(rng):
    """Sample a realistic POF and return weight/length q [N/m], EI [N m^2], diameter [m]."""
    if rng.random() < 0.6:
        d = rng.choice([0.5e-3, 0.75e-3, 1.0e-3])
        E, rho = rng.uniform(2.4e9, 3.3e9), 1190.0
        A, I = np.pi * d**2 / 4, np.pi * d**4 / 64
        q, EI, kind = rho * A * G, E * I, "bare_PMMA"
    else:
        dc, do = 1.0e-3, 2.2e-3
        Ec, Ej = rng.uniform(2.4e9, 3.3e9), rng.uniform(0.2e9, 0.5e9)
        Ac, Ao = np.pi * dc**2 / 4, np.pi * do**2 / 4
        Ic, Io = np.pi * dc**4 / 64, np.pi * do**4 / 64
        q = G * (1190 * Ac + 930 * (Ao - Ac))
        EI = Ec * Ic + Ej * (Io - Ic)
        d, kind = do, "jacketed_POF"
    return dict(kind=kind, d=float(d), q=float(q), EI=float(EI))


def elastica(alpha, theta0, k0L=0.0, n=600):
    """Solve the non-dimensional elastica. Returns x, y (units of L), theta, kappa*L, ok."""
    s = np.linspace(0, 1, n)

    def f(s, y):
        return np.vstack([y[1], -alpha * (1 - s) * np.cos(y[0])])

    def bc(ya, yb):
        return np.array([ya[0] - theta0, yb[1] - k0L])

    # continuation in alpha: start from the gravity-free solution (exact: theta = theta0 + k0L s)
    sol = None
    for a in np.linspace(0, alpha, max(2, int(np.ceil(alpha / 2)) + 1)):
        if sol is None:
            guess = np.vstack([theta0 + k0L * s, k0L * np.ones_like(s)])
            mesh = s
        else:
            mesh, guess = sol.x, sol.y
        alpha_c = a
        sol = solve_bvp(lambda s, y: np.vstack([y[1], -alpha_c * (1 - s) * np.cos(y[0])]),
                        bc, mesh, guess, tol=1e-8, max_nodes=50000)
    th, kap = sol.sol(s)
    ds = s[1] - s[0]
    x = np.concatenate([[0], np.cumsum(0.5 * (np.cos(th[1:]) + np.cos(th[:-1])) * ds)])
    y = -np.concatenate([[0], np.cumsum(0.5 * (np.sin(th[1:]) + np.sin(th[:-1])) * ds)])
    return x, y, th, kap, bool(sol.status == 0)


def selftest():
    print("1) small load, horizontal clamp -> Euler-Bernoulli tip deflection alpha/8")
    for a in [0.01, 0.1, 0.5]:
        x, y, *_ , ok = elastica(a, 0.0)
        print(f"   alpha={a:<5} elastica={-y[-1]:.6f}  linear={a/8:.6f}  ok={ok}")
    print("2) no gravity -> exact circular arc of curvature k0")
    x, y, th, kap, ok = elastica(0.0, np.pi / 2, k0L=-1.0)
    r = 1.0
    cx = r  # centre is to the right of the clamp
    print(f"   max |dist-to-centre - R| = {np.abs(np.hypot(x-cx, y) - r).max():.2e}")
    print("3) large load: tip angle must stay below 90 deg (fiber cannot pass vertical)")
    for a in [5, 20, 80]:
        x, y, th, *_ , ok = elastica(a, 0.0)
        print(f"   alpha={a:<3} tip angle={np.degrees(th[-1]):6.2f} deg  "
              f"tip drop={-y[-1]:.4f} L  ok={ok}")


# ----------------------------------------------------------------------------- render
def render(pts, radius, H, W, rng, bg_level, fiber_rgb, light_dir):
    """Anti-aliased tube render from a dense centerline (pixel coords, y down)."""
    # resample to ~0.2 px spacing so the nearest-point distance is accurate
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], max(int(cum[-1] / 0.2), 2))
    dense = np.column_stack([np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])])
    tang = np.gradient(dense, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12

    pad = int(radius * 3 + 6)
    x0, y0 = np.floor(dense.min(0)).astype(int) - pad
    x1, y1 = np.ceil(dense.max(0)).astype(int) + pad
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    P = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])
    d, k = cKDTree(dense).query(P)
    dv = P - dense[k]
    side = tang[k, 0] * dv[:, 1] - tang[k, 1] * dv[:, 0]

    dmin = np.full((H, W), np.inf, np.float32)
    nside = np.zeros((H, W), np.float32)
    dmin[y0:y1, x0:x1] = d.reshape(xx.shape)
    nside[y0:y1, x0:x1] = side.reshape(xx.shape)

    cover = np.clip(radius + 0.5 - dmin, 0, 1)
    u = np.clip(nside / max(radius, 1e-6), -1, 1)
    nz = np.sqrt(1 - u**2)
    diffuse = np.clip(light_dir[0] * u + light_dir[1] * nz, 0, 1)
    spec = np.clip(1 - np.abs(u - light_dir[0] * 0.6) / 0.18, 0, 1) ** 3
    shade = 0.35 + 0.65 * diffuse

    gx = np.linspace(0, 1, W)[None, :]
    gy = np.linspace(0, 1, H)[:, None]
    bg = bg_level * (0.85 + 0.15 * gy + 0.05 * gx) * (1 - 0.25 * ((gx - .5)**2 + (gy - .5)**2))
    img = np.repeat(bg[..., None], 3, -1).astype(np.float32)

    sh = gaussian_filter((dmin < radius).astype(np.float32), max(radius * 1.5, 1.5))
    sh = np.roll(np.roll(sh, int(round(radius * 1.2)) + 1, 0), int(round(radius * 0.8)) + 1, 1)
    img *= 1 - 0.25 * sh[..., None]

    fiber = shade[..., None] * np.asarray(fiber_rgb)[None, None] + 0.9 * spec[..., None]
    img = img * (1 - cover[..., None]) + fiber * cover[..., None]
    return img, cover


def render_ss(pts, radius, H, W, rng, *args):
    """Supersample thin fibers so sub-2px PMMA doesn't alias, then add optics + sensor."""
    ss = int(np.clip(np.ceil(2.5 / max(radius, 0.1)), 1, 3))
    img, cover = render(pts * ss, radius * ss, H * ss, W * ss, rng, *args)
    if ss > 1:
        img = img.reshape(H, ss, W, ss, 3).mean((1, 3))
        cover = cover.reshape(H, ss, W, ss).mean((1, 3))
    img = gaussian_filter(img, sigma=(0.6, 0.6, 0))
    img += rng.normal(0, 0.012, img.shape)
    mask = (cover >= 0.3).astype(np.uint8) * 255
    return np.clip(img, 0, 1), mask


# ----------------------------------------------------------------------------- sample
def make_sample(rng, H=1024, W=1024, pose="vertical"):
    fib = fiber_properties(rng)
    L = rng.uniform(0.08, 0.40)                              # free length [m]
    alpha = fib["q"] * L**3 / fib["EI"]

    R_reel = rng.uniform(0.08, 0.20)                         # reel radius [m]
    f_mem = rng.uniform(0.05, 0.6)                           # fraction of reel curvature kept
    if pose == "vertical":
        theta0 = np.deg2rad(90 + rng.uniform(-3, 3))         # clamp pointing straight down
        k0 = -f_mem / R_reel                                 # memory curls the fiber to the right
    else:
        theta0 = np.deg2rad(rng.uniform(-30, 45))
        k0 = rng.choice([-1, 1]) * f_mem / R_reel
    x, y, th, kapL, ok = elastica(alpha, theta0, k0 * L)

    # camera: framing varies, gravity always image-down; true physical scale
    xy = np.stack([x, y], 1) * L                             # metres, y up
    span = np.ptp(xy, 0)
    margin = 0.1
    fov = max(span.max() / (1 - 2 * margin), 0.05) * rng.uniform(1.0, 1.4)   # metres across
    m_per_px = fov / min(H, W)
    px = np.empty_like(xy)
    px[:, 0] = (xy[:, 0] - xy[:, 0].min()) / m_per_px
    px[:, 1] = (xy[:, 1].max() - xy[:, 1]) / m_per_px
    off = np.array([rng.uniform(margin * W, max(W - margin * W - np.ptp(px[:, 0]), margin * W + 1)),
                    rng.uniform(margin * H, max(H - margin * H - np.ptp(px[:, 1]), margin * H + 1))])
    px += off

    radius = max(fib["d"] / 2 / m_per_px, 0.75)              # true width (>=1.5 px so it's visible)
    bg = rng.uniform(0.15, 0.9)
    tint = np.array([[0.95, 0.95, 0.97], [0.15, 0.15, 0.17],
                     [0.9, 0.2, 0.15], [0.2, 0.4, 0.9]][rng.integers(4)]) * rng.uniform(0.8, 1.0)
    ang = rng.uniform(-0.8, 0.8)
    img, mask = render_ss(px, radius, H, W, rng, bg, tint, np.array([np.sin(ang), np.cos(ang)]))

    # clamp block, oriented with the clamp, contrasting with the background
    c = px[0].astype(int)
    r = int(max(radius * 3, 8))
    shade_clamp = 0.2 if bg > 0.45 else 0.6
    if pose == "vertical":
        img[max(c[1] - 2 * r, 0):c[1] + 1, max(c[0] - r, 0):c[0] + r] = shade_clamp
    else:
        img[max(c[1] - r, 0):c[1] + r, max(c[0] - 2 * r, 0):c[0] + 1] = shade_clamp

    meta = dict(fiber=fib["kind"], diameter_mm=fib["d"] * 1e3, length_m=float(L),
                q_N_per_m=fib["q"], EI_Nm2=fib["EI"], alpha=float(alpha),
                k0_per_m=float(k0), reel_radius_m=float(R_reel), memory_fraction=float(f_mem),
                theta0_deg=float(np.degrees(theta0)), mm_per_px=float(m_per_px * 1e3),
                radius_px=float(radius), solver_ok=ok)
    return (img * 255).astype(np.uint8), mask, px, th, kapL / L, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--pose", choices=["any", "vertical"], default="vertical")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    rng = np.random.default_rng(a.seed)
    for sub in ("images", "masks", "centerlines"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    metas = []
    for i in range(a.n):
        img, mask, px, th, kap, meta = make_sample(rng, a.size, a.size, a.pose)
        name = f"pof_{i:05d}"
        Image.fromarray(img).save(f"{a.out}/images/{name}.png")
        Image.fromarray(mask).save(f"{a.out}/masks/{name}.png")
        np.savetxt(f"{a.out}/centerlines/{name}.csv",
                   np.column_stack([np.linspace(0, 1, len(px)), px, th, kap]),
                   delimiter=",", header="s_norm,x_px,y_px,theta_rad,curvature_per_m",
                   comments="")
        meta["name"] = name
        metas.append(meta)
    with open(f"{a.out}/meta.json", "w") as fh:
        json.dump(metas, fh, indent=1)


if __name__ == "__main__":
    main()
