"""
Synthetic two-camera images of a plastic optical fiber (POF) in 3D, for depth / 3D
shape reconstruction. The fiber hangs from a clamp and a cable gantry pulls its tip.

Model (3D Kirchhoff rod, inextensible + unshearable, isotropic section):
    r'  = d3                              centerline, d3 = tangent
    di' = u x di                          material frame (d1, d2, d3), u = Darboux vector
    m   = EI (k1 - k1^0) d1 + EI (k2 - k2^0) d2 + GJ k3 d3,    u = k1 d1 + k2 d2 + k3 d3
    n'  = q e_z                           self-weight (world z up, gravity -z)
    m'  = -d3 x n
    clamp:  r(0) = 0, frame(0) given           tip:  n(L) = F (cable), m(L) = 0
  => n(s) = F - q (L - s) e_z is explicit, so the BVP is in (r, d1, d2, d3, m): 15 states.

Non-dimensional (s -> s/L, r -> r/L, m -> m L/EI, n -> n L^2/EI):
    alpha = q L^3 / EI        gravity-to-stiffness ratio (same as gen_pof.py)
    beta  = F L^2 / EI        cable pull (vector), cantilever tip deflection ~ |beta|/3
    k0 L                      spool memory (intrinsic curvature along d1)
    gamma = GJ / EI = 1/(1+nu)

Material data is shared with gen_pof.py (bare PMMA / jacketed POF, reel memory).

Cameras: two pinhole cameras (OpenCV convention: X_cam = R X_world + t, x right, y down,
z forward), either a rectified stereo pair or two views at an angle around the fiber.
Rendering is a z-buffered sphere-swept tube under perspective, so the fiber's pixel width
is its true diameter at its depth, self-occlusion is correct, and each view ships with an
exact mask and GT depth map. The clamp and the gantry cable are rendered but not masked.

Outputs (per sample <name>, per camera c in {0, 1}):
  images/<name>_c<c>.png      RGB
  masks/<name>_c<c>.png       fiber mask (0/255)
  depth/<name>_c<c>.png       uint16 camera-z depth of the fiber surface, 0.1 mm units, 0 = none
  centerlines/<name>.csv      s_norm, X/Y/Z [m], tangent, curvature/twist [1/m],
                              and the projection u, v [px] + depth z [m] in each camera
  meta.json                   physics + cameras (K, R, t) per sample
  Stereo disparity, when rig == stereo: disp = fx * baseline / depth.

Usage:
  python gen_pof3d.py --selftest
  python gen_pof3d.py --n 9 --out sample3d                 # mixed stereo / multi-view rigs
  python gen_pof3d.py --n 500 --rig stereo --baseline 0.1  # match a real rig
"""
import argparse, json, os
import numpy as np
from scipy.integrate import solve_bvp
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from PIL import Image

from gen_pof import fiber_properties, elastica

EZ = np.array([0.0, 0.0, 1.0])


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def random_dir(rng):
    return unit(rng.normal(size=3))


def perp_frame(d3, roll):
    """d1, d2 perpendicular to d3, rotated by `roll` about it (d1 x d2 = d3)."""
    a = EZ if abs(d3 @ EZ) < 0.9 else np.array([1.0, 0, 0])
    e1 = unit(np.cross(a, d3))
    e2 = np.cross(d3, e1)
    d1 = np.cos(roll) * e1 + np.sin(roll) * e2
    return d1, np.cross(d3, d1)


# ----------------------------------------------------------------------------- physics
def rod3d(alpha, beta, d3_0, roll=0.0, k0L=0.0, gamma=0.73, n=600):
    """Solve the non-dimensional 3D rod. Returns r (n,3), d3 (n,3), u_mat (n,3) = (k1,k2,k3)
    in units of 1/L, ok. Continuation in a load factor from the straight unloaded rod."""
    d3_0 = unit(d3_0)
    d1_0, d2_0 = perp_frame(d3_0, roll)
    beta = np.asarray(beta, float)
    frame0 = np.concatenate([d1_0, d2_0, d3_0])

    def make(lam):
        a, b, k = lam * alpha, lam * beta, lam * k0L

        def f(s, y):
            d1, d2, d3, m = y[3:6], y[6:9], y[9:12], y[12:15]
            nvec = b[:, None] - a * (1 - s)[None] * EZ[:, None]
            k1 = (m * d1).sum(0) + k
            k2 = (m * d2).sum(0)
            k3 = (m * d3).sum(0) / gamma
            u = k1 * d1 + k2 * d2 + k3 * d3
            return np.vstack([d3, np.cross(u, d1, axis=0), np.cross(u, d2, axis=0),
                              np.cross(u, d3, axis=0), -np.cross(d3, nvec, axis=0)])
        return f

    def bc(ya, yb):
        return np.concatenate([ya[0:3], ya[3:12] - frame0, yb[12:15]])

    # lambda = 0 is exact: straight rod along d3_0, frame constant, no moment
    xs = np.linspace(0, 1, 101)
    ys = np.vstack([np.outer(d3_0, xs), np.repeat(frame0[:, None], len(xs), 1),
                    np.zeros((3, len(xs)))])
    lam, step, sol = 0.0, 0.25, None
    while lam < 1:
        nxt = min(1.0, lam + step)
        trial = solve_bvp(make(nxt), bc, xs, ys, tol=1e-6, max_nodes=20000)
        if trial.status == 0:
            lam, sol, xs, ys = nxt, trial, trial.x, trial.y
            step = min(step * 1.5, 0.5)
        else:
            step /= 2
            if step < 1e-3:
                break
    if sol is None:
        return None, None, None, False
    y = sol.sol(np.linspace(0, 1, n))
    d1, d2, d3, m = y[3:6], y[6:9], y[9:12], y[12:15]
    u_mat = np.stack([(m * d1).sum(0) + k0L, (m * d2).sum(0), (m * d3).sum(0) / gamma], 1)
    return y[0:3].T, d3.T, u_mat, bool(lam == 1)


def self_intersects(r, diameter):
    ds = np.linalg.norm(r[1] - r[0])
    gap = int(np.ceil(2 * diameter / ds)) + 2
    pairs = cKDTree(r).query_pairs(diameter, output_type="ndarray")
    return bool(len(pairs) and (np.abs(pairs[:, 0] - pairs[:, 1]) > gap).any())


# ----------------------------------------------------------------------------- cameras
def look_at(C, target, roll=0.0):
    z = unit(target - C)
    x = unit(np.cross(z, EZ))
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    c, s = np.cos(roll), np.sin(roll)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R
    return R, -R @ C


def project(cam, X):
    Xc = X @ cam["R"].T + cam["t"]
    uv = Xc @ cam["K"].T
    return uv[:, :2] / uv[:, 2:3], Xc[:, 2]


def projection_matrix(cam):
    return cam["K"] @ np.column_stack([cam["R"], cam["t"]])


def triangulate(P0, P1, uv0, uv1):
    """Linear (DLT) triangulation of matched pixel tracks, (N,2) each -> (N,3)."""
    A = np.stack([uv0[:, :1] * P0[2] - P0[0], uv0[:, 1:] * P0[2] - P0[1],
                  uv1[:, :1] * P1[2] - P1[0], uv1[:, 1:] * P1[2] - P1[1]], 1)
    X = np.linalg.svd(A)[2][:, -1]
    return X[:, :3] / X[:, 3:]


def make_rig(rng, pts, W, H, rig, hfov=None, baseline=None, angle=None):
    """Place two cameras so `pts` fits in both views. Returns cameras + rig description."""
    hfov = np.deg2rad(hfov if hfov else rng.uniform(50, 70))
    f = W / 2 / np.tan(hfov / 2)
    K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]])
    half = np.arctan(min(W, H) / 2 / f)
    cen = (pts.min(0) + pts.max(0)) / 2
    rb = np.linalg.norm(pts - cen, axis=1).max()
    az, el = rng.uniform(0, 2 * np.pi), np.deg2rad(rng.uniform(-15, 25))
    D = rb / np.sin(half) * rng.uniform(1.05, 1.4)
    b = baseline if baseline else rng.uniform(0.06, 0.15)
    phi = np.deg2rad(angle if angle else rng.uniform(60, 120))

    def view_dir(a):
        return np.array([np.cos(el) * np.cos(a), np.cos(el) * np.sin(a), np.sin(el)])

    for _ in range(30):
        if rig == "stereo":
            Rc, _ = look_at(cen + D * view_dir(az), cen)
            Cs = [cen + D * view_dir(az) + s * b / 2 * Rc[0] for s in (-1, 1)]
            poses = [(Rc, -Rc @ C) for C in Cs]
        else:
            poses = [look_at(cen + D * view_dir(a), cen, np.deg2rad(rng.uniform(-2, 2)))
                     for a in (az, az + phi)]
        cams = [dict(K=K, R=R, t=t, W=W, H=H) for R, t in poses]
        fits = True
        for cam in cams:
            uv, z = project(cam, pts)
            fits &= bool((z > 0).all() and (uv[:, 0] > 0.05 * W).all() and (uv[:, 0] < 0.95 * W).all()
                         and (uv[:, 1] > 0.05 * H).all() and (uv[:, 1] < 0.95 * H).all())
        if fits:
            break
        D *= 1.1
    desc = dict(rig=rig, hfov_deg=float(np.degrees(hfov)), distance_m=float(D))
    if rig == "stereo":
        desc["baseline_m"] = float(b)
    else:
        desc["angle_deg"] = float(np.degrees(phi))
    return cams, desc


# ----------------------------------------------------------------------------- render
def splat(tubes, cam, ss, light_c):
    """z-buffered sphere-swept tubes at ss x supersampling. tubes: list of (pts3d, radius_m).
    Returns per-pixel tube id (-1 = none), cover, camera-space normal, surface depth."""
    W, H = cam["W"] * ss, cam["H"] * ss
    K = cam["K"].copy()
    K[:2] *= ss
    K[:2, 2] = ss * (cam["K"][:2, 2] + 0.5) - 0.5
    camS = dict(cam, K=K)
    f = K[0, 0]
    zbuf = np.full(H * W, np.inf)
    dxb, dyb, rb = np.zeros(H * W), np.zeros(H * W), np.ones(H * W)
    tid = np.full(H * W, -1)

    for i, (pts, rad) in enumerate(tubes):
        uv, z = project(camS, pts)
        ok = z > 1e-3
        if ok.sum() < 2:
            continue
        uv, P = uv[ok], pts[ok]
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(uv, axis=0), axis=1))])
        tt = np.linspace(0, cum[-1], max(int(cum[-1] / 0.25), 2))
        P = np.column_stack([np.interp(tt, cum, P[:, j]) for j in range(3)])
        uv, z = project(camS, P)
        r = np.maximum(f * rad / z, 0.75 * ss)
        keep = (uv[:, 0] > -r - 2) & (uv[:, 0] < W + r + 2) & (uv[:, 1] > -r - 2) & (uv[:, 1] < H + r + 2)
        uv, z, r = uv[keep], z[keep], r[keep]
        if not len(z):
            continue
        Ro = int(np.ceil(r.max() + 1))
        oy, ox = [o.ravel() for o in np.mgrid[-Ro:Ro + 1, -Ro:Ro + 1]]
        chunk = max(1, int(4e6 // len(ox)))
        for c0 in range(0, len(z), chunk):
            cu, cv, cz, cr = uv[c0:c0 + chunk, 0], uv[c0:c0 + chunk, 1], z[c0:c0 + chunk], r[c0:c0 + chunk]
            ix = np.round(cu).astype(int)[:, None] + ox[None]
            iy = np.round(cv).astype(int)[:, None] + oy[None]
            dx, dy = ix - cu[:, None], iy - cv[:, None]
            d = np.hypot(dx, dy)
            rr = np.broadcast_to(cr[:, None], d.shape)
            sel = (d < rr + 0.5) & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            dx, dy, d, rr = dx[sel], dy[sel], d[sel], rr[sel]
            zk = np.broadcast_to(cz[:, None], sel.shape)[sel]
            zs = zk - np.sqrt(np.clip(rr**2 - d**2, 0, None)) * zk / f
            pix = (iy * W + ix)[sel]
            order = np.lexsort((zs, pix))
            up, first = np.unique(pix[order], return_index=True)
            win = order[first]
            better = zs[win] < zbuf[up]
            up, win = up[better], win[better]
            zbuf[up], dxb[up], dyb[up], rb[up], tid[up] = zs[win], dx[win], dy[win], rr[win], i

    d = np.hypot(dxb, dyb)
    cover = np.where(tid >= 0, np.clip(rb + 0.5 - d, 0, 1), 0)
    nx, ny = dxb / rb, dyb / rb
    nz = -np.sqrt(np.clip(1 - nx**2 - ny**2, 0, None))
    nrm = np.stack([nx, ny, nz], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    shp = (H, W)
    return tid.reshape(shp), cover.reshape(shp), nrm.reshape(H, W, 3), zbuf.reshape(shp)


def render_view(rng, cam, fiber_pts, fiber_rad, extras, colors, light_w, bg_level):
    """extras: list of (pts3d, radius_m) drawn but not masked (clamp, cable)."""
    W, H = cam["W"], cam["H"]
    _, z = project(cam, fiber_pts)
    rmin = (cam["K"][0, 0] * fiber_rad / z).min()
    ss = int(np.clip(np.ceil(2.5 / max(rmin, 0.1)), 1, 3))
    light_c = cam["R"] @ light_w
    tid, cover, nrm, zs = splat([(fiber_pts, fiber_rad)] + extras, cam, ss, light_c)

    hv = unit(light_c + np.array([0, 0, -1.0]))
    diffuse = np.clip(nrm @ light_c, 0, 1)
    spec = np.clip(nrm @ hv, 0, 1) ** 40
    rgb = np.asarray(colors)[np.maximum(tid, 0)]
    obj = rgb * (0.3 + 0.7 * diffuse)[..., None] + (0.7 * spec)[..., None]

    Hs, Ws = tid.shape
    gx = np.linspace(0, 1, Ws)[None, :]
    gy = np.linspace(0, 1, Hs)[:, None]
    bg = bg_level * (0.85 + 0.15 * gy + 0.05 * gx) * (1 - 0.25 * ((gx - .5)**2 + (gy - .5)**2))
    img = np.repeat(bg[..., None], 3, -1)
    rpx = np.median(cam["K"][0, 0] * ss * fiber_rad / z)
    sh = gaussian_filter((cover > 0.5).astype(np.float32), max(rpx * 1.5, 1.5))
    sh = np.roll(np.roll(sh, int(round(rpx * 1.2)) + 1, 0), int(round(rpx * 0.8)) + 1, 1)
    img *= 1 - 0.25 * sh[..., None]
    img = img * (1 - cover[..., None]) + obj * cover[..., None]

    fcov = np.where(tid == 0, cover, 0)
    fz = np.where(tid == 0, zs, np.inf)
    if ss > 1:
        img = img.reshape(H, ss, W, ss, 3).mean((1, 3))
        fcov = fcov.reshape(H, ss, W, ss).mean((1, 3))
        fz = fz.reshape(H, ss, W, ss).min((1, 3))
    img = gaussian_filter(img, sigma=(0.6, 0.6, 0))
    img += rng.normal(0, 0.012, img.shape)
    mask = fcov >= 0.3
    depth = np.where(mask & np.isfinite(fz), np.round(fz * 1e4), 0).clip(0, 65535).astype(np.uint16)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8), mask.astype(np.uint8) * 255, depth


# ----------------------------------------------------------------------------- sample
def sample_physics(rng, pose, p_cable, p_memory):
    """Sample material, clamp, memory and cable pull until the rod solves cleanly."""
    for _ in range(50):
        fib = fiber_properties(rng)
        nu = rng.uniform(0.35, 0.40)
        gamma = 1 / (1 + nu)
        L = rng.uniform(0.08, 0.40)
        alpha = fib["q"] * L**3 / fib["EI"]
        if pose == "vertical":                               # clamp pointing down, small tilt
            d3_0 = unit(-EZ + np.tan(np.deg2rad(rng.uniform(0, 3))) * perp_frame(EZ, rng.uniform(0, 2 * np.pi))[0])
        else:                                                # anything from 30 deg above horizontal down
            el = np.arcsin(rng.uniform(-1, np.sin(np.deg2rad(30))))
            a = rng.uniform(0, 2 * np.pi)
            d3_0 = np.array([np.cos(el) * np.cos(a), np.cos(el) * np.sin(a), np.sin(el)])
        roll = rng.uniform(0, 2 * np.pi)
        R_reel, f_mem = rng.uniform(0.08, 0.20), rng.uniform(0.05, 0.6)
        k0 = f_mem / R_reel if rng.random() < p_memory else 0.0
        beta = (np.exp(rng.uniform(np.log(0.2), np.log(30))) * random_dir(rng)
                if rng.random() < p_cable else np.zeros(3))
        r, t, u, ok = rod3d(alpha, beta, d3_0, roll, k0 * L, gamma)
        if ok and not self_intersects(r * L, fib["d"]):
            break
    else:
        raise RuntimeError("no solvable rod configuration in 50 tries")
    meta = dict(fiber=fib["kind"], diameter_mm=fib["d"] * 1e3, length_m=float(L),
                q_N_per_m=fib["q"], EI_Nm2=fib["EI"], GJ_Nm2=fib["EI"] * gamma,
                alpha=float(alpha), beta=beta.tolist(),
                tip_force_N=(beta * fib["EI"] / L**2).tolist(),
                clamp_dir=d3_0.tolist(), clamp_roll_rad=float(roll),
                k0_per_m=float(k0), reel_radius_m=float(R_reel), memory_fraction=float(f_mem))
    return r * L, t, u / L, fib["d"], meta


def make_sample(rng, W=1024, H=768, pose="vertical", rig="mixed", hfov=None, baseline=None,
                angle=None, cable=True, p_cable=0.85, p_memory=0.7):
    X, T, U, dia, meta = sample_physics(rng, pose, p_cable, p_memory)

    d3_0 = np.asarray(meta["clamp_dir"])
    clamp = X[0] - np.outer(np.linspace(0, 0.025, 20), d3_0)          # 25 mm clamp body
    rig = rig if rig != "mixed" else ("stereo" if rng.random() < 0.5 else "multiview")
    cams, rig_meta = make_rig(rng, np.vstack([X, clamp]), W, H, rig, hfov, baseline, angle)

    extras = [(clamp, 0.005)]
    colors = [np.array([[0.95, 0.95, 0.97], [0.15, 0.15, 0.17],
                        [0.9, 0.2, 0.15], [0.2, 0.4, 0.9]][rng.integers(4)]) * rng.uniform(0.8, 1.0)]
    colors.append(np.full(3, rng.uniform(0.15, 0.35)))
    F = np.asarray(meta["tip_force_N"])
    if cable and np.linalg.norm(F) > 0:
        extras.append((X[-1] + np.outer(np.linspace(0, 0.6, 300), unit(F)), 0.00015))
        colors.append(np.full(3, rng.uniform(0.05, 0.3)))
    light_w = unit(np.array([*rng.normal(size=2), rng.uniform(0.5, 2.0)]))

    views = []
    for cam in cams:
        img, mask, depth = render_view(rng, cam, X, dia / 2, extras, colors, light_w,
                                       rng.uniform(0.15, 0.9))
        views.append((img, mask, depth))

    cols = [np.linspace(0, 1, len(X)), X, T, np.linalg.norm(U[:, :2], axis=1), U[:, 2]]
    for cam in cams:
        uv, z = project(cam, X)
        cols += [uv, z]
    table = np.column_stack(cols)
    meta.update(rig_meta)
    meta["cameras"] = [dict(K=c["K"].tolist(), R=c["R"].tolist(), t=c["t"].tolist(),
                            width=W, height=H) for c in cams]
    return views, table, meta


CSV_HEADER = ("s_norm,X_m,Y_m,Z_m,tx,ty,tz,curvature_per_m,twist_per_m,"
              "u0_px,v0_px,z0_m,u1_px,v1_px,z1_m")


# ----------------------------------------------------------------------------- checks
def selftest():
    print("1) planar gravity, horizontal clamp -> matches 2D elastica in gen_pof.py")
    for a in [0.1, 2.0, 20.0]:
        r, *_, ok = rod3d(a, np.zeros(3), [1, 0, 0])
        x, y, *_ = elastica(a, 0.0)
        print(f"   alpha={a:<5} tip drop 3D={-r[-1, 2]:.6f}  2D={-y[-1]:.6f}  "
              f"out-of-plane={abs(r[-1, 1]):.1e}  ok={ok}")
    print("2) small sideways cable pull, no gravity -> Euler-Bernoulli tip deflection beta/3")
    for b in [0.01, 0.1]:
        r, *_, ok = rod3d(0.0, [0, b, 0], [0, 0, -1])
        print(f"   beta={b:<5} rod={r[-1, 1]:.6f}  linear={b / 3:.6f}  ok={ok}")
    print("3) memory only -> circular arc of radius 1/k0")
    r, t, u, ok = rod3d(0.0, np.zeros(3), [0, 0, -1], roll=0.7, k0L=2.0)
    d1_0, d2_0 = perp_frame(unit([0, 0, -1]), 0.7)
    c = r[0] - 0.5 * d2_0                                    # d3' = k0 d1 x d3 = -k0 d2
    print(f"   max |dist-to-centre - 1/k0| = {np.abs(np.linalg.norm(r - c, axis=1) - 0.5).max():.2e}  "
          f"out of the d2-d3 plane: {np.abs(r @ d1_0).max():.1e}  ok={ok}")
    print("4) 3D load (gravity + oblique pull + memory + tilted clamp): global moment balance")
    alpha, beta = 8.0, np.array([3.0, -2.0, 1.0])
    d3_0 = unit([0.3, 0.2, -1])
    r, t, u, ok = rod3d(alpha, beta, d3_0, roll=1.1, k0L=1.5)
    # m(0) = r(1) x beta + int r x (-alpha e_z) ds, and m(0) in the material frame is
    # (k1 - k0, k2, gamma k3) evaluated at the clamp
    s = np.linspace(0, 1, len(r))
    m_loads = np.cross(r[-1], beta) + np.trapezoid(np.cross(r, -alpha * EZ), s, axis=0)
    d1_0, d2_0 = perp_frame(d3_0, 1.1)
    m_rod = (u[0, 0] - 1.5) * d1_0 + u[0, 1] * d2_0 + 0.73 * u[0, 2] * d3_0
    print(f"   |m(0) loads - m(0) rod| = {np.linalg.norm(m_loads - m_rod):.2e}  "
          f"|tangent|-1 max = {np.abs(np.linalg.norm(t, axis=1) - 1).max():.1e}  "
          f"out-of-plane extent = {np.ptp(r @ np.cross(d3_0, EZ) / np.linalg.norm(np.cross(d3_0, EZ))):.3f} L  ok={ok}")
    print("5) cameras: triangulate GT projections back to 3D")
    rng = np.random.default_rng(1)
    for rig in ["stereo", "multiview"]:
        cams, desc = make_rig(rng, r * 0.3, 1024, 768, rig)
        uv0, _ = project(cams[0], r * 0.3)
        uv1, _ = project(cams[1], r * 0.3)
        Xh = triangulate(projection_matrix(cams[0]), projection_matrix(cams[1]), uv0, uv1)
        print(f"   {rig:<9} max error = {np.abs(Xh - r * 0.3).max():.1e} m   {desc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out3d")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--pose", choices=["any", "vertical"], default="vertical")
    ap.add_argument("--rig", choices=["stereo", "multiview", "mixed"], default="mixed")
    ap.add_argument("--hfov", type=float, help="horizontal FOV [deg] (default: 50-70)")
    ap.add_argument("--baseline", type=float, help="stereo baseline [m] (default: 0.06-0.15)")
    ap.add_argument("--angle", type=float, help="multi-view angle between cameras [deg] (default: 60-120)")
    ap.add_argument("--no-cable", action="store_true", help="don't render the gantry cable")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    rng = np.random.default_rng(a.seed)
    for sub in ("images", "masks", "depth", "centerlines"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    metas = []
    for i in range(a.n):
        views, table, meta = make_sample(rng, a.width, a.height, a.pose, a.rig, a.hfov,
                                         a.baseline, a.angle, not a.no_cable)
        name = f"pof_{i:05d}"
        for c, (img, mask, depth) in enumerate(views):
            Image.fromarray(img).save(f"{a.out}/images/{name}_c{c}.png")
            Image.fromarray(mask).save(f"{a.out}/masks/{name}_c{c}.png")
            Image.fromarray(depth).save(f"{a.out}/depth/{name}_c{c}.png")
        np.savetxt(f"{a.out}/centerlines/{name}.csv", table, delimiter=",",
                   header=CSV_HEADER, comments="")
        meta["name"] = name
        metas.append(meta)
        print(f"{name}: {meta['rig']}, alpha={meta['alpha']:.2f}, "
              f"|beta|={np.linalg.norm(meta['beta']):.2f}")
    with open(f"{a.out}/meta.json", "w") as fh:
        json.dump(metas, fh, indent=1)


if __name__ == "__main__":
    main()
