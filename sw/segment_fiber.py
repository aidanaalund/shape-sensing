"""
Segment the fiber in real camera images (e.g. from grid_capture.py) and extract its centerline.

Method
  1. Backdrop: the largest red region (convex hull, eroded). Tip: saturated blue blobs (the tip LED cap).
  2. Ridge map: Sato filter on min(R,G,B). The clear/white fiber is bright in all channels while the
     red backdrop is dark in G and B, so the fiber is a bright thin ridge. Off the backdrop, Sato on B-R:
     the fiber carries blue light from the tip, which stands out from the clutter behind it.
  3. Tracking: start at the clamp point and step along the ridge. Each step picks the direction
     (within +-MAX_TURN of the heading over the last HEADING_LEN px) with the strongest ridge response
     over a short look-ahead. The response is capped near the fiber's own level, so brighter crossing
     lines (acrylic edges) can't pull the track off; they are passed straight through. Short gaps (e.g. a bead on the
     fiber) are bridged by coasting straight. At the backdrop edge, tracking carries on over the B-R
     ridge with the same heading. Stops at the blue tip, the image edge, or when the ridge fades.
  4. Smoothing spline through the tracked points, resampled to N points.
  5. Outline: the ridge follows the brightest streak inside the clear fiber, not its axis. Across each
     centerline sample, both fiber edges are found from the colour difference to the backdrop. The fiber
     is then modelled as a tube (smooth axis + width that varies only slowly through perspective), fitted
     robustly so cross-sections spoiled by the clamp, crossing acrylic edges or the bead are ignored. The
     centerline is moved onto the tube axis, and width / FIBER_DIAM_MM gives the image scale [px/mm].

The clamp point is clicked once per data folder and stored in <folder>/seg_config.json, since the
camera and clamp don't move between images of one run.

Outputs in <out> (default <folder>/seg):
  centerlines/<name>.csv  s_px, s_norm, x_px, y_px, theta_rad, curvature_per_px   (clamp -> end)
                          theta is measured below horizontal in image coordinates (90 deg = straight
                          down), same convention as gen_pof.py
  outlines/<name>.csv     s_px, xl_px, yl_px, xr_px, yr_px, width_px   left/right fiber edges per sample
  masks/<name>.png        fiber mask (filled outline)
  overlays/<name>.png     image with backdrop outline, fiber outline, centerline, and end marker, for checking
  summary.csv             one row per image

Usage
  python segment_fiber.py ../grid_data                  # all images in the folder (click clamp once)
  python segment_fiber.py ../grid_data/img_r00_c00.png --show
  python segment_fiber.py ../grid_data --pick           # re-click the clamp point
  python segment_fiber.py ../grid_data --clamp 428 117
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import make_smoothing_spline
from skimage.filters import sato

N_OUT = 600         # centerline samples, same as gen_pof.py
SIGMAS = (1.5, 2.5, 3.5)   # ridge filter scales [px], ~ fiber radius
STEP = 3.0          # tracking step [px]
LOOKAHEAD = 15      # [px] ridge response is averaged over this distance ahead
MAX_TURN = 12       # [deg] max deviation from the heading over the last HEADING_LEN px
HEADING_LEN = 30    # [px]
INIT_TURN = 45      # [deg] search cone around INIT_DIR for the first step
INIT_LOOK = 60      # [px] look-ahead for the first step
INIT_DIR = 90       # [deg] initial direction, below horizontal in image coords (90 = down)
STOP_FRAC = 0.3     # stop when response < STOP_FRAC * typical response of the fiber
MAX_GAP = 60        # [px] longest gap (bead, occlusion) bridged by going straight
FIBER_DIAM_MM = 2.5 # fiber diameter, gives the image scale from the measured width
EDGE_SEARCH = 16    # [px] half-length of the cross-section profiles used to find the edges
EDGE_FRAC = 0.2     # edge = where the colour difference to the backdrop drops below this fraction of its peak
RED_HUE = (6, 160)  # backdrop hue < [0] or > [1] (OpenCV, 0..180); glare through the acrylic shifts it to ~170,
                    # the low side stays tight to keep skin (hue ~5-15) out
RED_SAT = 60        # min backdrop saturation; glare washes the backdrop out to ~90
TIP_HUE = (95, 130) # blue tip hue range (OpenCV, 0..180)
TIP_MIN_AREA = 100  # [px] smaller blue blobs are ignored


# ----------------------------------------------------------------------------- image features
def backdrop_mask(img, erode=6):
    """Convex hull of the largest saturated red region."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    red = (((h < RED_HUE[0]) | (h > RED_HUE[1])) & (s > RED_SAT) & (v > 40)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(red)
    if n < 2:
        return np.ones(img.shape[:2], np.uint8)          # no backdrop found: search everywhere
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    pts = cv2.findNonZero((lab == big).astype(np.uint8))
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1)
    return cv2.erode(mask, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))


def ridge_map(img):
    b, g, r = cv2.split(img.astype(np.float32))
    return sato(np.minimum(np.minimum(r, g), b), sigmas=SIGMAS, black_ridges=False).astype(np.float32)


def blue_ridge_map(img):
    """Off the backdrop: the fiber carries blue light from the tip LED, so B-R is a clean ridge over
    the clutter (cables, desk, skin) where brightness alone is not."""
    b, g, r = cv2.split(img.astype(np.float32))
    return sato(b - r, sigmas=SIGMAS, black_ridges=False).astype(np.float32)


def tip_mask(img):
    """Saturated, bright blue blobs (thin dark cables are opened away). Not just the largest: other blue
    things in view (a tape roll) can be bigger, but the track only stops on a blob it runs into."""
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    blue = ((h > TIP_HUE[0]) & (h < TIP_HUE[1]) & (s > 120) & (v > 80)).astype(np.uint8)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(blue)
    return np.isin(lab, 1 + np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= TIP_MIN_AREA)).astype(np.uint8)


def sample(a, pts):
    """Bilinear sample of a at (N,2) xy points; 0 outside."""
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.remap(a, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR, borderValue=0).ravel()


# ----------------------------------------------------------------------------- tracking
def unit(deg):
    t = np.radians(deg)
    return np.array([np.cos(t), np.sin(t)])


def best_direction(R, p, theta, cone, look):
    """Direction (deg) within theta +- cone with the highest mean ridge response over `look` px."""
    cands = theta + np.arange(-cone, cone + 1e-9, 2.0)
    ks = np.arange(1, look + 1)
    scores = np.array([sample(R, p + ks[:, None] * unit(c)).mean() for c in cands])
    scores -= 0.002 * np.abs(cands - theta) * scores.max()   # slight preference for going straight
    i = np.argmax(scores)
    return cands[i], scores[i]


def recenter(R, q, theta, half=3):
    """Move q along the normal to the ridge maximum (sub-pixel)."""
    nrm = unit(theta + 90)
    offs = np.arange(-half, half + 1)
    v = sample(R, q + offs[:, None] * nrm)
    i = int(np.argmax(v))
    if 0 < i < len(v) - 1:
        d = v[i - 1] - 2 * v[i] + v[i + 1]
        o = offs[i] + (0.5 * (v[i - 1] - v[i + 1]) / d if d < 0 else 0)
    else:
        o = offs[i]
    return q + o * nrm


def track(R, inside, clamp, init_dir=INIT_DIR, tip=None):
    Rd = cv2.dilate(R, np.ones((5, 5), np.uint8))
    p = np.asarray(clamp, float)
    theta, _ = best_direction(Rd, p, init_dir, INIT_TURN, INIT_LOOK)
    # the clicked point is only approximate: start on the fiber a few px in, extrapolated back
    q = recenter(R, p + 3 * STEP * unit(theta), theta)
    return follow(R, inside, [q - 3 * STEP * unit(theta)], theta, tip, "backdrop edge")


def follow(R, region, pts, theta, tip=None, leave="image edge"):
    """Continue the track `pts` (list) along the ridge of R until it leaves `region` (end = `leave`),
    reaches the blue tip (end = "tip") or the ridge fades. theta = initial heading [deg]."""
    # directions are scored on a widened ridge map so a candidate a pixel or two off still counts
    Rd = cv2.dilate(R, np.ones((5, 5), np.uint8))
    p, pts, resp = pts[-1].copy(), list(pts), []
    ref = None            # typical fiber response, from the first steps
    Rc = Rd               # response capped at the fiber's level once known
    gap = 0.0
    end = "faded"
    h, w = R.shape
    k_head = max(2, int(HEADING_LEN / STEP))

    def stop(q):
        if not (0 <= q[0] < w and 0 <= q[1] < h) or not region[int(q[1]), int(q[0])]:
            return leave
        if tip is not None and tip[int(q[1]), int(q[0])]:
            return "tip"

    while len(pts) < 5000:
        if len(pts) > k_head:
            d = pts[-1] - pts[-1 - k_head]
            theta = np.degrees(np.arctan2(d[1], d[0]))
        cand, score = best_direction(Rc, p, theta, MAX_TURN, LOOKAHEAD)
        if ref is not None and score < STOP_FRAC * ref:
            # weak ridge ahead: coast straight, look for the fiber resuming
            p = p + STEP * unit(theta)
            gap += STEP
            if gap > MAX_GAP:
                end = "faded"
                break
            if stop(p):
                end = stop(p)
                break
            continue
        q = recenter(R, p + STEP * unit(cand), cand)
        if stop(q):
            end = stop(q)
            break
        if gap:                                     # fill the bridged gap with a straight segment
            pts.append(p.copy())
        gap = 0.0
        pts.append(q)
        resp.append(sample(R, q[None])[0])
        p = q
        if len(resp) == 8 or (ref is not None and len(resp) % 20 == 0):
            # follow slow brightness changes along the fiber, but never collapse
            med = np.median(resp[-60:])
            ref = med if ref is None else max(0.5 * ref, min(med, 2 * ref))
            Rc = np.minimum(Rd, 1.5 * ref)
    pts = np.array(pts)
    # drop trailing steps that hook off the heading (e.g. onto the edge of an occluding bead)
    # cut at the first step in the tail that turns sharply away from the heading before it, and after which
    # the rest of the tail stays turned away (a single jittery step is not a hook)
    def turn(d_head, d):
        return np.degrees(np.arctan2(d_head[0] * d[1] - d_head[1] * d[0], d_head @ d))
    for i in range(max(k_head + 1, len(pts) - 2 * k_head), len(pts)):
        d_head = pts[i - 1] - pts[i - 1 - k_head]
        if abs(turn(d_head, pts[i] - pts[i - 1])) > 25 and abs(turn(d_head, pts[-1] - pts[i - 1])) > 25:
            pts = pts[:i]
            break
    return pts, end


# ----------------------------------------------------------------------------- centerline
def fit_centerline(pts, n=N_OUT, lam=None):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    keep = np.r_[True, np.diff(s) > 1e-6]
    pts, s = pts[keep], s[keep]
    lam = lam if lam is not None else 5.0 * STEP ** 3      # smoothing strength
    sx, sy = make_smoothing_spline(s, pts[:, 0], lam=lam), make_smoothing_spline(s, pts[:, 1], lam=lam)
    # re-parameterize by the arc length of the smooth curve
    t = np.linspace(0, s[-1], 4 * n)
    xy = np.column_stack([sx(t), sy(t)])
    sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    su = np.linspace(0, sl[-1], n)
    tu = np.interp(su, sl, t)
    x, y = sx(tu), sy(tu)
    dx, dy = sx.derivative()(tu), sy.derivative()(tu)
    ddx, ddy = sx.derivative(2)(tu), sy.derivative(2)(tu)
    theta = np.arctan2(dy, dx)                                     # below horizontal, y down
    kappa = (dx * ddy - dy * ddx) / np.maximum(dx ** 2 + dy ** 2, 1e-12) ** 1.5
    return np.column_stack([su, su / su[-1], x, y, theta, kappa])


def edge_offsets(img, cl, half=EDGE_SEARCH, step=0.25):
    """Left/right fiber edge offsets [px] along the normal at every centerline sample (NaN = not found).

    The clear fiber shows up as a colour change against the uniform backdrop: bright streaks inside and a
    dark refraction rim at the boundary. Per cross-section the background colour is taken from both ends
    of the profile, and each edge is where the colour distance to it falls below EDGE_FRAC of its peak.
    Offsets are along the normal (-sin th, cos th), i.e. positive = right of the direction of travel."""
    f = cv2.GaussianBlur(img.astype(np.float32), (3, 3), 0)
    offs = np.arange(-half, half + step / 2, step)
    nb = int(4 / step)                                    # 4 px of background at each end
    lo, hi = np.full(len(cl), np.nan), np.full(len(cl), np.nan)
    for j, (x, y, th) in enumerate(cl[:, 2:5]):
        nrm = np.array([-np.sin(th), np.cos(th)])
        p = np.array([x, y]) + offs[:, None] * nrm
        c = np.stack([sample(f[..., k], p) for k in range(3)], 1)
        bl, br = np.median(c[:nb], 0), np.median(c[-nb:], 0)
        bg = bl + (br - bl) * np.linspace(0, 1, len(offs))[:, None]   # linear background across the fiber
        d = np.linalg.norm(c - bg, axis=1)
        noise = max(np.std(d[:nb]), np.std(d[-nb:]), 1.0)
        core = np.abs(offs) <= 4
        i0 = np.flatnonzero(core)[np.argmax(d[core])]
        thr = max(EDGE_FRAC * d[i0], 4 * noise)
        if d[i0] < 2 * thr:
            continue
        below = d < thr
        il = np.flatnonzero(below[:i0])                       # last background sample left of the peak
        ir = np.flatnonzero(below[i0:])                       # first background sample right of the peak
        if not len(il) or not len(ir):
            continue
        il, ir = il[-1], i0 + ir[0]
        # sub-pixel threshold crossings
        lo[j] = offs[il] + step * (thr - d[il]) / (d[il + 1] - d[il])
        hi[j] = offs[ir - 1] + step * (d[ir - 1] - thr) / (d[ir - 1] - d[ir])
    return lo, hi


def fit_outline(img, cl):
    """Model the fiber as a tube: smooth axis offset + slowly varying width (perspective only), fitted
    robustly to the measured edges. Returns the corrected centerline, left/right edges (N,2) and width (N,)."""
    lo, hi = edge_offsets(img, cl)
    s = cl[:, 0]
    w, c = hi - lo, 0.5 * (hi + lo)
    ok = np.isfinite(w)
    if ok.sum() < 20:
        raise RuntimeError("fiber edges not found")
    # a 2.5 mm rod only changes apparent width through perspective: fit a low-order polynomial,
    # iteratively dropping cross-sections whose width or centre is off (clamp, crossing edges, bead)
    for _ in range(5):
        pw = np.polyfit(s[ok], w[ok], 2)
        rw = w - np.polyval(pw, s)
        cs = make_smoothing_spline(s[ok], c[ok], lam=2e4)
        rc = c - cs(s)
        sw_, sc = 1.4826 * np.median(np.abs(rw[ok])), 1.4826 * np.median(np.abs(rc[ok]))
        new = np.isfinite(w) & (np.abs(rw) < max(3 * sw_, 0.5)) & (np.abs(rc) < max(3 * sc, 0.5))
        if (new == ok).all():
            break
        ok = new
    width = np.polyval(pw, s)
    nrm = np.column_stack([-np.sin(cl[:, 4]), np.cos(cl[:, 4])])
    # bridge rejected cross-sections straight across in the image (the tracked ridge often kinks there);
    # only before the first / after the last good one follow the ridge with the fitted offset
    axis = cl[:, 2:4] + np.where(ok, c, cs(s))[:, None] * nrm
    i0, i1 = np.flatnonzero(ok)[[0, -1]]
    for k in range(2):
        axis[i0:i1, k] = np.interp(s[i0:i1], s[ok], axis[ok, k])
    cl = fit_centerline(axis)
    nrm = np.column_stack([-np.sin(cl[:, 4]), np.cos(cl[:, 4])])
    width = np.interp(cl[:, 1], np.linspace(0, 1, len(width)), width)
    left = cl[:, 2:4] - 0.5 * width[:, None] * nrm
    right = cl[:, 2:4] + 0.5 * width[:, None] * nrm
    return cl, left, right, width, ok.mean()


def segment(img, clamp, init_dir=INIT_DIR):
    inside, tip = backdrop_mask(img), tip_mask(img)
    pts, end = track(ridge_map(img) * inside, inside, clamp, init_dir, tip)
    if len(pts) < 10:
        raise RuntimeError("tracking failed near the clamp point (check --clamp / --pick)")
    if end == "backdrop edge":
        # off the backdrop: keep going on the blue light the fiber carries, up to the tip
        d = pts[-1] - pts[max(0, len(pts) - 1 - int(HEADING_LEN / STEP))]
        pts, end = follow(blue_ridge_map(img), np.ones_like(inside), pts, np.degrees(np.arctan2(d[1], d[0])), tip)
    # the ridge follows the brightest streak inside the clear fiber, not its axis: re-centre between the edges
    cl, left, right, width, frac = fit_outline(img, fit_centerline(pts))
    poly = np.vstack([left, right[::-1]])
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(poly * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
    return dict(centerline=cl, left=left, right=right, widths=width, width=float(np.median(width)),
                edge_frac=frac, mask=(mask > 127).astype(np.uint8) * 255, end=end, inside=inside, raw=pts)


def overlay(img, res, clamp):
    v = img.copy()
    cnts, _ = cv2.findContours(res["inside"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(v, cnts, -1, (255, 255, 0), 1)
    for k, col in (("left", (255, 128, 0)), ("right", (255, 128, 0)), ("centerline", (0, 255, 0))):
        xy = res[k][:, 2:4] if k == "centerline" else res[k]
        cv2.polylines(v, [np.round(xy * 4).astype(np.int32)], False, col, 1, cv2.LINE_AA, shift=2)
    cv2.drawMarker(v, tuple(int(c) for c in clamp), (255, 0, 255), cv2.MARKER_SQUARE, 14, 2)
    cv2.drawMarker(v, tuple(int(c) for c in res["centerline"][-1, 2:4]), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
    L = res["centerline"][-1, 0]
    cv2.putText(v, f"L={L:.0f}px  w={res['width']:.1f}px ({res['width'] / FIBER_DIAM_MM:.2f} px/mm)  "
                   f"end: {res['end']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return v


# ----------------------------------------------------------------------------- CLI
def pick_clamp(img):
    """Click the clamp point (where the fiber leaves the clamp). Enter/Space to accept."""
    pt = []
    win = "click the clamp point, Enter to accept, Esc to cancel"

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pt[:] = [x, y]
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        v = img.copy()
        if pt:
            cv2.drawMarker(v, tuple(pt), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.imshow(win, v)
        k = cv2.waitKey(20)
        if k in (13, 32) and pt:
            break
        if k == 27 or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            raise SystemExit("no clamp point selected")
    cv2.destroyAllWindows()
    return pt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="image file or folder of images")
    ap.add_argument("--glob", default="*.png", help="image pattern when path is a folder")
    ap.add_argument("--out", help="output folder (default <folder>/seg)")
    ap.add_argument("--clamp", type=float, nargs=2, metavar=("X", "Y"), help="clamp point [px]")
    ap.add_argument("--init-dir", type=float, help=f"initial direction [deg below horizontal] (default {INIT_DIR})")
    ap.add_argument("--pick", action="store_true", help="click the clamp point again")
    ap.add_argument("--show", action="store_true", help="show each overlay (any key = next, q = stop)")
    a = ap.parse_args()

    path = Path(a.path)
    files = sorted(path.glob(a.glob)) if path.is_dir() else [path]
    folder = path if path.is_dir() else path.parent
    if not files:
        raise SystemExit(f"no images matching {a.glob} in {path}")
    out = Path(a.out) if a.out else folder / "seg"
    for sub in ("centerlines", "outlines", "masks", "overlays"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    cfg_path = folder / "seg_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    if a.clamp:
        cfg["clamp"] = list(a.clamp)
    if a.init_dir is not None:
        cfg["init_dir"] = a.init_dir
    if a.pick or "clamp" not in cfg:
        cfg["clamp"] = pick_clamp(cv2.imread(str(files[0])))
    cfg.setdefault("init_dir", INIT_DIR)
    cfg_path.write_text(json.dumps(cfg, indent=1))
    print(f"clamp {cfg['clamp']}, init dir {cfg['init_dir']} deg  ({cfg_path})")

    rows = []
    for f in files:
        img = cv2.imread(str(f))
        try:
            res = segment(img, cfg["clamp"], cfg["init_dir"])
        except RuntimeError as e:
            print(f"{f.name}: FAILED {e}")
            rows.append([f.name, "", "", "", "", "", "", "failed"])
            continue
        cl = res["centerline"]
        np.savetxt(out / "centerlines" / f"{f.stem}.csv", cl, delimiter=",", comments="", fmt="%.6g",
                   header="s_px,s_norm,x_px,y_px,theta_rad,curvature_per_px")
        np.savetxt(out / "outlines" / f"{f.stem}.csv",
                   np.column_stack([cl[:, 0], res["left"], res["right"], res["widths"]]), delimiter=",",
                   comments="", fmt="%.6g", header="s_px,xl_px,yl_px,xr_px,yr_px,width_px")
        cv2.imwrite(str(out / "masks" / f"{f.stem}.png"), res["mask"])
        ov = overlay(img, res, cfg["clamp"])
        cv2.imwrite(str(out / "overlays" / f"{f.stem}.png"), ov)
        rows.append([f.name, f"{cl[-1, 0]:.1f}", f"{res['width']:.2f}",
                     f"{res['width'] / FIBER_DIAM_MM:.3f}", f"{res['edge_frac']:.2f}", f"{cl[-1, 2]:.1f}", f"{cl[-1, 3]:.1f}",
                     res["end"]])
        print(f"{f.name}: L={cl[-1, 0]:.0f}px  width={res['width']:.1f}px ({res['edge_frac']:.0%} edges used)  end=({cl[-1, 2]:.0f}, {cl[-1, 3]:.0f})"
              f" [{res['end']}]")
        if a.show:
            cv2.imshow("segmentation", ov)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "length_px", "width_px", "px_per_mm", "edge_frac", "end_x_px", "end_y_px", "end_reason"])
        w.writerows(rows)
    print(f"results in {out}")


if __name__ == "__main__":
    main()
