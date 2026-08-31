"""
Host-lane extraction and quadratic fitting for high-speed tracking.

Row-wise sampling of YOLOPv2 lane-line / drivable-area masks, Canny
narrow-band search, and symmetric Hough fallback. All coordinates are
full-frame pixels. Quadratics use normalized row yn in [0, 1]:

    x = a * yn^2 + b * yn + c
    yn = (y - y_bot) / (y_top - y_bot)
"""

from typing import Callable, List, Optional, Tuple
import cv2
import numpy as np


PredFn = Callable[[float], Optional[float]]


def eval_quadratic(
    coeffs: np.ndarray,
    y: float,
    y_bot: float,
    y_top: float,
) -> float:
    """Evaluates x(y) = a*yn^2 + b*yn + c with yn mapped from [y_bot, y_top]."""
    denom = float(y_top) - float(y_bot)
    yn = 0.0 if abs(denom) < 1e-3 else (float(y) - float(y_bot)) / denom
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    return a * yn * yn + b * yn + c


def sample_quadratic(
    coeffs: np.ndarray,
    y_bot: float,
    y_top: float,
    n: int = 12,
    y_end: Optional[float] = None,
) -> np.ndarray:
    """Returns Nx2 image-space samples [[x, y], ...] from bottom toward y_end."""
    y_lo = float(y_bot)
    y_hi = float(y_end if y_end is not None else y_top)
    if n < 2:
        n = 2
    ys = np.linspace(y_lo, y_hi, n, dtype=np.float32)
    xs = np.array([eval_quadratic(coeffs, y, y_bot, y_top) for y in ys], dtype=np.float32)
    return np.column_stack([xs, ys])


def fit_quadratic(
    xs: np.ndarray,
    ys: np.ndarray,
    y_bot: float,
    y_top: float,
    min_points: int = 2,
    refine: bool = True,
) -> Optional[np.ndarray]:
    """Fits [a, b, c] mapping normalized y to x. Degenerate fits pad a (and b)."""
    if xs is None or ys is None:
        return None
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    if xs.size < min_points or xs.size != ys.size:
        return None
    denom = float(y_top) - float(y_bot)
    if abs(denom) < 1e-3:
        return None
    yn = (ys - float(y_bot)) / denom
    degree = 2 if xs.size >= 4 else 1
    # Near-field (bottom) anchors LDW; do not let noisy top rows lever the whole line.
    w = np.square(1.0 - 0.40 * np.clip(yn, 0.0, 1.0))
    try:
        raw = np.polyfit(yn, xs, deg=degree, w=w)
    except (np.linalg.LinAlgError, ValueError):
        return None
    coeffs = np.zeros(3, dtype=np.float32)
    coeffs[-raw.size :] = raw.astype(np.float32)
    if not np.all(np.isfinite(coeffs)):
        return None
    if refine and xs.size >= 6:
        resid = np.abs(xs - np.polyval(raw, yn))
        med = float(np.median(resid))
        thresh = max(6.0, 2.5 * med)
        keep = resid <= thresh
        if int(np.count_nonzero(keep)) >= min_points and not np.all(keep):
            return fit_quadratic(
                xs[keep], ys[keep], y_bot, y_top, min_points=min_points, refine=False
            )
    return coeffs


def gate_points(
    pts: List[Tuple[float, float]],
    pred_fn: Optional[PredFn],
    max_dx: float,
) -> List[Tuple[float, float]]:
    """Drops measurements farther than max_dx from the predicted x(y)."""
    if not pts or pred_fn is None or max_dx <= 0:
        return pts
    kept: List[Tuple[float, float]] = []
    for x, y in pts:
        px = pred_fn(y)
        if px is None or abs(float(x) - float(px)) <= max_dx:
            kept.append((float(x), float(y)))
    return kept


def occlude_tracks(
    image: np.ndarray,
    tracks: Optional[list],
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    y0: float = 0.0,
    pad: int = 14,
) -> np.ndarray:
    """Zeros tracked vehicle boxes so car contours cannot look like lane paint."""
    if image is None or not tracks:
        return image
    h, w = image.shape[:2]
    copied = False
    out = image
    for track in tracks:
        bbox = getattr(track, "bbox", None)
        if bbox is None:
            continue
        x1 = int((float(bbox.x) - pad) * x_scale)
        x2 = int((float(bbox.x) + float(bbox.w) + pad) * x_scale)
        y1 = int((float(bbox.y) - pad - y0) * y_scale)
        y2 = int((float(bbox.y) + float(bbox.h) + pad - y0) * y_scale)
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        if not copied:
            out = image.copy()
            copied = True
        out[y1:y2, x1:x2] = 0
    return out


def _row_runs(row: np.ndarray, min_width: int = 1) -> List[Tuple[float, float, float]]:
    """Paint blobs on a 1-D mask row as (centroid, start, end) with end exclusive."""
    if row.size < 2:
        return []
    mask = row > 0
    if not np.any(mask):
        return []
    padded = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(padded == 1)
    ends = np.flatnonzero(padded == -1)
    out: List[Tuple[float, float, float]] = []
    for start, end in zip(starts, ends):
        if end - start >= min_width:
            out.append((0.5 * float(start + end - 1), float(start), float(end)))
    return out


def _row_clusters(row: np.ndarray, min_width: int = 1) -> List[float]:
    """Centroids of consecutive nonzero runs in a 1-D mask row."""
    return [c for c, _lo, _hi in _row_runs(row, min_width=min_width)]


def _da_edges(row: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    ones = np.flatnonzero(row > 0)
    if ones.size < 2:
        return None, None
    return float(ones[0]), float(ones[-1])


def _assign_clusters(
    clusters: List[float],
    da_left: Optional[float],
    da_right: Optional[float],
    pred_l: Optional[float],
    pred_r: Optional[float],
    mid_x: float,
    width: float,
) -> Tuple[Optional[float], Optional[float]]:
    """Unique left/right assignment using DA edges, then track prediction, then midline."""
    if not clusters:
        return None, None

    # YOLO / DA edges define the host line. Track prediction only disambiguates
    # extra blobs (adjacent lane, a car that leaked past occlusion).
    t_left = da_left if da_left is not None else pred_l
    t_right = da_right if da_right is not None else pred_r
    if t_left is None:
        t_left = mid_x * 0.55
    if t_right is None:
        t_right = mid_x + 0.45 * (width - mid_x)

    gate = 0.14 * width
    if da_left is not None or da_right is not None:
        gate = max(12.0, 0.08 * width)
    elif pred_l is not None or pred_r is not None:
        gate = max(8.0, 0.06 * width)

    left_x: Optional[float] = None
    right_x: Optional[float] = None
    used = set()

    def take(target: float, prefer_left: bool) -> Optional[float]:
        best_i = -1
        best_d = gate
        for i, c in enumerate(clusters):
            if i in used:
                continue
            if prefer_left and c > mid_x + 0.08 * width:
                continue
            if (not prefer_left) and c < mid_x - 0.08 * width:
                continue
            d = abs(c - target)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i < 0:
            return None
        used.add(best_i)
        return clusters[best_i]

    # Prefer DA-edge / prediction targets; if both map to one blob, left wins first
    # then right retries remaining clusters.
    left_x = take(float(t_left), True)
    right_x = take(float(t_right), False)

    if left_x is None and right_x is None and len(clusters) == 1:
        c = clusters[0]
        if c < mid_x:
            left_x = c
        else:
            right_x = c
    return left_x, right_x


def extract_host_lane_segments(
    ll_mask: np.ndarray,
    da_mask: Optional[np.ndarray],
    y_top: int,
    y_bot: int,
    n_rows: int = 24,
    mask_width: int = 320,
    pred_left: Optional[PredFn] = None,
    pred_right: Optional[PredFn] = None,
) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float, float, float]]]:
    """
    Host left/right YOLO lane-line blobs per sample row.

    Each item is (x_center, y, x_lo, x_hi) in full-frame pixels. Association
    prefers drivable-area edges so the ego lines win over the next lane.
    """
    h, w = ll_mask.shape[:2]
    y_top = int(max(0, min(h - 2, y_top)))
    y_bot = int(max(y_top + 2, min(h, y_bot)))
    road_h = y_bot - y_top
    if road_h < 4 or w < 8:
        return [], []

    small_w = int(max(64, min(mask_width, w)))
    scale = small_w / float(w)
    small_h = max(8, road_h)
    ll_roi = ll_mask[y_top:y_bot, :]
    if ll_roi.shape[1] != small_w or ll_roi.shape[0] != small_h:
        small_ll = cv2.resize(ll_roi, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    else:
        small_ll = ll_roi
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    small_ll = cv2.dilate(small_ll, kernel, iterations=1)

    small_da: Optional[np.ndarray] = None
    if da_mask is not None and da_mask.shape[:2] == ll_mask.shape[:2]:
        da_roi = da_mask[y_top:y_bot, :]
        small_da = cv2.resize(da_roi, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

    n_rows = int(max(6, min(n_rows, small_h)))
    ys_small = np.linspace(0, small_h - 1, n_rows)
    left_segs: List[Tuple[float, float, float, float]] = []
    right_segs: List[Tuple[float, float, float, float]] = []
    mid_s = 0.5 * small_w

    for y_s in ys_small:
        yi = int(round(y_s))
        y_img = y_top + (yi + 0.5) * (road_h / float(small_h))
        runs = _row_runs(small_ll[yi], min_width=1)
        if not runs:
            continue
        clusters = [c for c, _lo, _hi in runs]
        run_by_centroid = {c: (lo, hi) for c, lo, hi in runs}
        da_l = da_r = None
        if small_da is not None:
            da_l, da_r = _da_edges(small_da[yi])
        pred_l = pred_left(y_img) if pred_left is not None else None
        pred_r = pred_right(y_img) if pred_right is not None else None
        pred_l_s = pred_l * scale if pred_l is not None else None
        pred_r_s = pred_r * scale if pred_r is not None else None
        xl_s, xr_s = _assign_clusters(
            clusters, da_l, da_r, pred_l_s, pred_r_s, mid_s, float(small_w)
        )
        if xl_s is not None:
            lo, hi = run_by_centroid.get(xl_s, (xl_s, xl_s + 1.0))
            left_segs.append((xl_s / scale, y_img, lo / scale, (hi - 1.0) / scale))
        if xr_s is not None:
            lo, hi = run_by_centroid.get(xr_s, (xr_s, xr_s + 1.0))
            right_segs.append((xr_s / scale, y_img, lo / scale, (hi - 1.0) / scale))
    return left_segs, right_segs


def extract_host_lane_points(
    ll_mask: np.ndarray,
    da_mask: Optional[np.ndarray],
    y_top: int,
    y_bot: int,
    n_rows: int = 24,
    mask_width: int = 320,
    pred_left: Optional[PredFn] = None,
    pred_right: Optional[PredFn] = None,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Host left/right paint centroids from a lane-line mask (full-frame px)."""
    left_segs, right_segs = extract_host_lane_segments(
        ll_mask=ll_mask,
        da_mask=da_mask,
        y_top=y_top,
        y_bot=y_bot,
        n_rows=n_rows,
        mask_width=mask_width,
        pred_left=pred_left,
        pred_right=pred_right,
    )
    return [(s[0], s[1]) for s in left_segs], [(s[0], s[1]) for s in right_segs]


def band_width_at_y(
    y: float,
    y_bot: float,
    y_top: float,
    band_bot: float,
    band_top: float,
) -> float:
    """Perspective search half-width: wide near the camera, tight near the horizon."""
    denom = float(y_bot) - float(y_top)
    if abs(denom) < 1e-3:
        return float(band_bot)
    t = (float(y) - float(y_top)) / denom
    t = float(np.clip(t, 0.0, 1.0))
    return float(band_top + t * (band_bot - band_top))


def _subpixel_peak(vals: np.ndarray, i: int) -> float:
    if i <= 0 or i >= int(vals.size) - 1:
        return float(i)
    a, b, c = float(vals[i - 1]), float(vals[i]), float(vals[i + 1])
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-6:
        return float(i)
    return float(i) + float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))


def intensity_ridge_x(
    gray: np.ndarray,
    y: float,
    x_pred: float,
    band_px: float,
    bgr: Optional[np.ndarray] = None,
    min_contrast: float = 16.0,
    max_band_px: float = 64.0,
) -> Optional[float]:
    """
    Paint-center x at row y. Expands the window if the bright run is clipped
    (near-field markings are wider than a fixed band; a clipped run walks inward).
    """
    h, w = gray.shape[:2]
    yi = int(round(y))
    if yi < 0 or yi >= h:
        return None
    half = max(6, int(round(band_px)))
    max_half = max(half, int(round(max_band_px)))
    xc = int(round(x_pred))

    def score_row(x1: int, x2: int) -> Optional[np.ndarray]:
        if x2 - x1 < 7:
            return None
        row = gray[yi, x1:x2].astype(np.float32)
        if bgr is not None and bgr.shape[0] == h and bgr.shape[1] == w and bgr.ndim == 3:
            pix = bgr[yi, x1:x2].astype(np.float32)
            yellow = np.maximum(0.0, 0.5 * (pix[:, 2] + pix[:, 1]) - pix[:, 0])
            row = np.maximum(row, yellow)
        if row.size >= 5:
            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
            row = np.convolve(row, kernel / kernel.sum(), mode="same")
        return row

    x_meas: Optional[float] = None
    for _ in range(5):
        x1 = max(0, xc - half)
        x2 = min(w, xc + half + 1)
        row = score_row(x1, x2)
        if row is None:
            break
        peak_i = int(np.argmax(row))
        med = float(np.median(row))
        contrast = float(row[peak_i] - med)
        if contrast < min_contrast:
            # Window is filled with paint (near-field yellow is wide) — expand to find edges.
            if med >= 80.0 and half < max_half:
                half = min(max_half, half + 14)
                continue
            if med >= 80.0:
                return float(x_pred)
            return None
        floor = med + 0.45 * contrast
        left_i = peak_i
        while left_i > 0 and row[left_i - 1] >= floor:
            left_i -= 1
        right_i = peak_i
        while right_i < row.size - 1 and row[right_i + 1] >= floor:
            right_i += 1
        hit_left = left_i == 0
        hit_right = right_i == row.size - 1
        if hit_left and hit_right:
            # Whole window is paint — stay on the prediction instead of sliding.
            return float(x_pred)
        if hit_left or hit_right:
            if half >= max_half:
                x_meas = float(x1) + 0.5 * (float(left_i) + float(right_i))
                break
            half = min(max_half, half + 14)
            continue
        if right_i - left_i < 1:
            x_meas = float(x1) + _subpixel_peak(row, peak_i)
        else:
            x_meas = float(x1) + 0.5 * (float(left_i) + float(right_i))
        break
    return x_meas


def canny_paint_center(
    edges: np.ndarray,
    y_img: float,
    x_pred: float,
    y_top: int,
    scale: float,
    band_px: float,
) -> Optional[float]:
    """Midpoint of the inner/outer Canny pair in the search band (paint center)."""
    eh, ew = edges.shape[:2]
    y_s = (float(y_img) - float(y_top)) * scale
    yi = int(round(y_s))
    if yi < 0 or yi >= eh:
        return None
    band_s = max(3.0, float(band_px) * scale)
    xc = float(x_pred) * scale
    x1 = max(0, int(xc - band_s))
    x2 = min(ew, int(xc + band_s) + 1)
    if x2 - x1 < 4:
        return None
    nz = np.flatnonzero(edges[yi, x1:x2] > 0)
    if nz.size < 2:
        return None
    left_e = float(nz[0])
    right_e = float(nz[-1])
    width_e = right_e - left_e
    if width_e < 2.0 or width_e > max(10.0, 1.35 * band_s):
        return None
    mid = 0.5 * (left_e + right_e)
    center = 0.5 * (x2 - x1)
    if abs(mid - center) > 0.55 * band_s:
        return None
    return (x1 + mid) / scale


# Debug / fusion source ids (visualizer colors these).
SRC_YELLOW = 0
SRC_RIDGE = 1
SRC_MATCH = 2
SRC_CANNY = 3
SRC_MASK = 4
SRC_HOUGH = 5

R_YELLOW = 3.5
R_RIDGE = 6.0
R_MATCH = 5.0
R_CANNY = 12.0
R_YOLO = 3.0  # lane-line mask is the primary association
R_MASK = R_YOLO
R_HOUGH = 28.0


def paint_width_at_y(y: float, y_bot: float, y_top: float, w_bot: float = 22.0, w_top: float = 5.0) -> float:
    """Expected paint bar width in pixels (perspective)."""
    denom = float(y_bot) - float(y_top)
    if abs(denom) < 1e-3:
        return float(w_bot)
    t = float(np.clip((float(y) - float(y_top)) / denom, 0.0, 1.0))
    return float(w_top + t * (w_bot - w_top))


def _score_row(
    gray: np.ndarray,
    yi: int,
    x1: int,
    x2: int,
    bgr: Optional[np.ndarray] = None,
    yellow_boost: float = 1.35,
) -> np.ndarray:
    row = gray[yi, x1:x2].astype(np.float32)
    if bgr is not None and bgr.ndim == 3 and bgr.shape[0] == gray.shape[0] and bgr.shape[1] == gray.shape[1]:
        pix = bgr[yi, x1:x2].astype(np.float32)
        yellow = np.maximum(0.0, 0.5 * (pix[:, 2] + pix[:, 1]) - pix[:, 0])
        row = np.maximum(row, yellow * yellow_boost)
    return row


def yellow_chroma_x(
    bgr: Optional[np.ndarray],
    y: float,
    x_pred: float,
    band_px: float,
    min_chroma: float = 22.0,
) -> Optional[float]:
    """Peak of yellow chroma (R+G)/2 - B. Ignores white paint and gray glare."""
    if bgr is None or bgr.ndim != 3:
        return None
    h, w = bgr.shape[:2]
    yi = int(round(y))
    if yi < 0 or yi >= h:
        return None
    half = max(8, int(round(band_px)))
    x1 = max(0, int(round(x_pred)) - half)
    x2 = min(w, int(round(x_pred)) + half + 1)
    if x2 - x1 < 7:
        return None
    pix = bgr[yi, x1:x2].astype(np.float32)
    chroma = np.maximum(0.0, 0.5 * (pix[:, 2] + pix[:, 1]) - pix[:, 0])
    if chroma.size >= 5:
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
        chroma = np.convolve(chroma, kernel / kernel.sum(), mode="same")
    i = int(np.argmax(chroma))
    if float(chroma[i]) < min_chroma:
        return None
    if i <= 1 or i >= chroma.size - 2:
        return None
    return float(x1) + _subpixel_peak(chroma, i)


def matched_filter_x(
    gray: np.ndarray,
    y: float,
    x_pred: float,
    band_px: float,
    bgr: Optional[np.ndarray] = None,
    paint_w: float = 8.0,
) -> Optional[float]:
    """1-D bright-bar matched filter on the paint score (gray ∪ yellow chroma)."""
    h, w = gray.shape[:2]
    yi = int(round(y))
    if yi < 0 or yi >= h:
        return None
    half = max(8, int(round(band_px)))
    x1 = max(0, int(round(x_pred)) - half)
    x2 = min(w, int(round(x_pred)) + half + 1)
    if x2 - x1 < 11:
        return None
    score = _score_row(gray, yi, x1, x2, bgr=bgr)
    pw = max(3, int(round(paint_w)))
    if pw % 2 == 0:
        pw += 1
    sh = max(2, pw // 2)
    k = np.concatenate(
        [-np.ones(sh, dtype=np.float32), np.ones(pw, dtype=np.float32), -np.ones(sh, dtype=np.float32)]
    )
    k -= float(k.mean())
    conv = np.convolve(score, k, mode="same")
    margin = sh + pw // 2
    if conv.size <= 2 * margin + 3:
        return None
    region = conv[margin : conv.size - margin]
    i = int(np.argmax(region)) + margin
    peak = float(conv[i])
    med = float(np.median(region))
    mad = float(np.median(np.abs(region - med))) + 1e-3
    if peak < med + 4.0 * mad:
        return None
    if float(score[i]) < float(np.median(score)) + 12.0:
        return None
    return float(x1) + _subpixel_peak(conv, i)


def measure_lane_row(
    gray: np.ndarray,
    edges: np.ndarray,
    y: float,
    x_pred: float,
    band_px: float,
    y_top: int,
    scale: float,
    bgr: Optional[np.ndarray] = None,
    y_bot: float = 0.0,
    y_roi_top: float = 0.0,
    max_band_px: float = 64.0,
    want_yellow: bool = False,
) -> Tuple[List[Tuple[float, float, int]], List[Tuple[float, float, int]]]:
    """
    Classical cues at one sample row.

    Returns (kalman_meas, debug_raw) where each item is (x, R_or_y, src_id).
    Kalman meas uses R as the second field; debug_raw uses image y.
    Correlated paint cues (yellow / ridge / matched-filter) are collapsed to a
    consensus so they cannot over-weight one another vs Canny / YOLO.
    """
    yf = float(y)
    pred = float(x_pred)
    half = float(band_px)
    debug: List[Tuple[float, float, int]] = []
    paint: List[Tuple[float, float, int]] = []  # x, R, src

    if want_yellow:
        x_y = yellow_chroma_x(bgr, yf, pred, half)
        if x_y is not None:
            paint.append((x_y, R_YELLOW, SRC_YELLOW))
            debug.append((x_y, yf, SRC_YELLOW))

    x_m = matched_filter_x(
        gray, yf, pred, half, bgr=bgr,
        paint_w=paint_width_at_y(yf, y_bot, y_roi_top) if y_bot > y_roi_top else 8.0,
    )
    if x_m is not None:
        paint.append((x_m, R_MATCH, SRC_MATCH))
        debug.append((x_m, yf, SRC_MATCH))

    x_r = intensity_ridge_x(gray, yf, pred, half, bgr=bgr, max_band_px=max_band_px)
    if x_r is not None:
        paint.append((x_r, R_RIDGE, SRC_RIDGE))
        debug.append((x_r, yf, SRC_RIDGE))

    meas: List[Tuple[float, float, int]] = []
    if paint:
        xs = np.array([p[0] for p in paint], dtype=np.float64)
        # Largest agreeing cluster within 6 px; else the lowest-R cue.
        best_src = min(paint, key=lambda p: p[1])
        cluster = [p for p in paint if abs(p[0] - best_src[0]) <= 6.0]
        if len(cluster) >= 2:
            z = float(np.median([p[0] for p in cluster]))
            r = float(min(p[1] for p in cluster)) / float(len(cluster))
            src = min(cluster, key=lambda p: p[1])[2]
            meas.append((z, max(2.0, r), src))
        else:
            # Disagree: keep only the most trusted cue (yellow > match > ridge).
            meas.append(best_src)
            for p in paint:
                if p is not best_src and abs(p[0] - best_src[0]) > 6.0:
                    # keep debug only; do not feed the Kalman a second paint peak
                    pass

    x_c = canny_paint_center(edges, yf, pred, y_top, scale, half)
    if x_c is not None:
        debug.append((x_c, yf, SRC_CANNY))
        if not meas or abs(x_c - meas[0][0]) > 3.0:
            meas.append((x_c, R_CANNY, SRC_CANNY))

    return meas, debug


def refine_lane_points(
    gray: np.ndarray,
    edges: np.ndarray,
    y_top: int,
    scale: float,
    y_samples: np.ndarray,
    pred_xs: np.ndarray,
    band_pxs: np.ndarray,
    bgr: Optional[np.ndarray] = None,
    max_band_px: float = 64.0,
) -> List[Tuple[float, float]]:
    """Paint-center measurements per row: intensity ridge, else Canny edge-pair midpoint."""
    pts: List[Tuple[float, float]] = []
    if y_samples.size == 0:
        return pts
    for y_img, px, band in zip(y_samples, pred_xs, band_pxs):
        yf = float(y_img)
        pred = float(px)
        half = float(band)
        x_r = intensity_ridge_x(gray, yf, pred, half, bgr=bgr, max_band_px=max_band_px)
        if x_r is not None:
            pts.append((x_r, yf))
            continue
        x_c = canny_paint_center(edges, yf, pred, y_top, scale, half)
        if x_c is not None:
            pts.append((x_c, yf))
    return pts


def narrow_band_points(
    edges: np.ndarray,
    y_top: int,
    scale: float,
    y_samples: np.ndarray,
    pred_xs: np.ndarray,
    band_px: float,
) -> List[Tuple[float, float]]:
    """Legacy helper: Canny paint-center (edge-pair midpoint), not nearest-edge."""
    pts: List[Tuple[float, float]] = []
    if y_samples.size == 0:
        return pts
    for y_img, px in zip(y_samples, pred_xs):
        x_c = canny_paint_center(edges, float(y_img), float(px), y_top, scale, band_px)
        if x_c is not None:
            pts.append((x_c, float(y_img)))
    return pts


def hough_lane_points(
    lines: Optional[np.ndarray],
    scale: float,
    y_top: int,
    y_bot: int,
    y_roi_top: int,
    img_w: int,
    min_length: float,
    y_samples: np.ndarray,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Symmetric Hough fallback. Left and right use mirrored geometric gates;
    right is NOT conditioned on the left line.
    """
    left_pts: List[Tuple[float, float]] = []
    right_pts: List[Tuple[float, float]] = []
    if lines is None:
        return left_pts, right_pts

    w = float(img_w)
    y_ref_bot = float(y_bot)
    y_ref_top = float(y_roi_top)

    for line in lines:
        l_arr = np.array(line).ravel()
        if l_arr.size < 4:
            continue
        x1_s, y1_s, x2_s, y2_s = (float(l_arr[0]), float(l_arr[1]), float(l_arr[2]), float(l_arr[3]))
        x1 = x1_s / scale
        x2 = x2_s / scale
        y1 = y1_s / scale + y_top
        y2 = y2_s / scale + y_top
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) < 1e-3 or abs(dy) < 1e-3:
            continue
        length = float(np.hypot(dx, dy))
        if length < min_length:
            continue
        slope = dy / dx
        angle = float(np.degrees(np.arctan(slope)))
        mid_x = 0.5 * (x1 + x2)
        xb = x1 + (y_ref_bot - y1) / slope
        xt = x1 + (y_ref_top - y1) / slope

        side: Optional[str] = None
        if -68.0 <= angle <= -12.0 and mid_x < w * 0.55:
            if 0.02 * w <= xb <= 0.52 * w and 0.18 * w <= xt <= 0.58 * w:
                side = "left"
        elif 12.0 <= angle <= 68.0 and mid_x >= 0.45 * w:
            if 0.48 * w <= xb <= 0.98 * w and 0.42 * w <= xt <= 0.82 * w:
                side = "right"
        if side is None:
            continue

        dest = left_pts if side == "left" else right_pts
        for y in y_samples:
            yf = float(y)
            if yf < min(y1, y2) - 8.0 or yf > max(y1, y2) + 8.0:
                continue
            x = x1 + (yf - y1) / slope
            if 0.0 <= x < w:
                dest.append((x, yf))
        dest.append((xb, y_ref_bot))
        dest.append((xt, y_ref_top))
        dest.append((0.5 * (x1 + x2), 0.5 * (y1 + y2)))
    return left_pts, right_pts


class RowAnchorTracker:
    """
    Independent 1-D Kalman filters at fixed image rows (UFLD-style anchors).

    Each row fuses yellow chroma, matched-filter, intensity ridge, Canny pair,
    and gated YOLO/Hough with its own R. The published line is the anchors
    themselves — a quadratic is only a readout, never the filter state.
    Gap rows (dashed paint) coast; they are not allowed to banana a global fit.
    """

    Q = 2.5
    P0 = 180.0
    P_MAX = 420.0
    CONF_MAX = 24.0
    CONF_HIT = 3.0
    CONF_MISS = 0.40

    def __init__(self, max_jump_px: float = 28.0):
        self.max_jump_px = float(max_jump_px)
        self.n = 0
        self.ys: Optional[np.ndarray] = None
        self.xs = np.zeros(0, dtype=np.float32)
        self.P = np.zeros(0, dtype=np.float32)
        self.age = np.zeros(0, dtype=np.int32)
        self.confidence: float = 0.0
        self.acquired: bool = False
        self.poly: Optional[np.ndarray] = None

    def reset(self):
        n = self.n
        self.xs = np.full(n, np.nan, dtype=np.float32)
        self.P = np.full(n, self.P0, dtype=np.float32)
        self.age = np.zeros(n, dtype=np.int32)
        self.confidence = 0.0
        self.acquired = False
        self.poly = None

    def set_ys(self, ys: np.ndarray):
        ys = np.asarray(ys, dtype=np.float32).ravel()
        if self.ys is not None and self.ys.size == ys.size and np.allclose(self.ys, ys, atol=2.0):
            return
        old_ys, old_xs, old_P = self.ys, self.xs, self.P
        self.n = int(ys.size)
        self.ys = ys
        self.xs = np.full(self.n, np.nan, dtype=np.float32)
        self.P = np.full(self.n, self.P0, dtype=np.float32)
        self.age = np.full(self.n, 99, dtype=np.int32)
        if old_ys is not None and old_xs.size > 0:
            ok = np.isfinite(old_xs)
            if int(np.count_nonzero(ok)) >= 2:
                order = np.argsort(old_ys[ok])
                self.xs = np.interp(ys, old_ys[ok][order], old_xs[ok][order]).astype(np.float32)
                self.P = np.full(self.n, float(np.median(old_P[ok])), dtype=np.float32)

    def predict(self):
        if self.n == 0:
            return
        self.P = np.minimum(self.P + self.Q, self.P_MAX)
        self.age = self.age + 1

    @property
    def valid(self) -> bool:
        if not self.acquired or self.ys is None:
            return False
        return int(np.count_nonzero(np.isfinite(self.xs))) >= 3 and self.confidence > 0.0

    @property
    def x(self) -> Optional[np.ndarray]:
        """Quadratic readout [a,b,c] for callers that still want a poly."""
        return self.poly

    def eval_x(self, y: float, y_bot: float = 0.0, y_top: float = 0.0) -> Optional[float]:
        if self.ys is None:
            return None
        ok = np.isfinite(self.xs)
        if int(np.count_nonzero(ok)) < 2:
            return None
        order = np.argsort(self.ys[ok])
        return float(np.interp(float(y), self.ys[ok][order], self.xs[ok][order]))

    def nearest_row(self, y: float) -> Optional[int]:
        if self.ys is None or self.n == 0:
            return None
        i = int(np.argmin(np.abs(self.ys - float(y))))
        step = float(np.abs(self.ys[1] - self.ys[0])) if self.n > 1 else 12.0
        if abs(float(self.ys[i]) - float(y)) > 0.65 * step + 4.0:
            return None
        return i

    def update_row(self, i: int, z: float, r: float, gate: float) -> bool:
        if i < 0 or i >= self.n:
            return False
        z = float(z)
        r = float(max(1.5, r))
        if not np.isfinite(self.xs[i]):
            self.xs[i] = np.float32(z)
            self.P[i] = np.float32(r)
            self.age[i] = 0
            return True
        innov = z - float(self.xs[i])
        if abs(innov) > float(gate):
            return False
        s = float(self.P[i]) + r
        k = float(self.P[i]) / s
        self.xs[i] = np.float32(float(self.xs[i]) + k * innov)
        self.P[i] = np.float32((1.0 - k) * float(self.P[i]))
        self.age[i] = 0
        return True

    def ingest_points(
        self,
        pts: List[Tuple[float, float]],
        r: float,
        gate: float,
        src: int = SRC_MASK,
    ) -> int:
        """Assign sparse (x,y) points (YOLO / Hough) to nearest rows."""
        n_ok = 0
        if not pts or self.ys is None:
            return 0
        for x, y in pts:
            i = self.nearest_row(float(y))
            if i is None:
                continue
            if self.update_row(i, float(x), r, gate):
                n_ok += 1
        return n_ok

    def fill_gaps(self):
        """Initialize never-seen rows from neighbors. Does not overwrite coasting rows."""
        if self.ys is None:
            return
        ok = np.isfinite(self.xs)
        if int(np.count_nonzero(ok)) < 2:
            return
        missing = np.flatnonzero(~ok)
        if missing.size == 0:
            return
        order = np.argsort(self.ys[ok])
        yp = self.ys[ok][order]
        xp = self.xs[ok][order]
        self.xs[missing] = np.interp(self.ys[missing], yp, xp).astype(np.float32)
        self.P[missing] = np.minimum(np.maximum(self.P[missing], 90.0), self.P_MAX)

    def neighbor_prior(self):
        """Soft pull of uncertain (dashed-gap) rows toward locked neighbors."""
        if self.ys is None or not self.acquired:
            return
        locked = np.isfinite(self.xs) & (self.P < 45.0) & (self.age <= 2)
        if int(np.count_nonzero(locked)) < 2:
            return
        order = np.argsort(self.ys[locked])
        prior = np.interp(self.ys, self.ys[locked][order], self.xs[locked][order])
        uncertain = np.isfinite(self.xs) & (self.P > 55.0)
        for i in np.flatnonzero(uncertain):
            self.update_row(int(i), float(prior[i]), r=48.0, gate=36.0)

    def commit(self, y_bot: float, y_top: float, min_rows: int = 3) -> bool:
        n_ok = int(np.count_nonzero(np.isfinite(self.xs)))
        n_fresh = int(np.count_nonzero(self.age <= 1))
        if n_ok >= min_rows:
            self.fill_gaps()
            self.neighbor_prior()
            self.acquired = True
            if n_fresh >= 2:
                self.confidence = min(self.CONF_MAX, self.confidence + self.CONF_HIT)
            else:
                self.confidence = max(1.0, self.confidence - self.CONF_MISS)
            self._refresh_poly(y_bot, y_top)
            return n_fresh >= 1
        self.confidence = max(0.0, self.confidence - 1.0)
        if self.confidence <= 0.0:
            self.reset()
        return False

    def _refresh_poly(self, y_bot: float, y_top: float):
        ok = np.isfinite(self.xs)
        if int(np.count_nonzero(ok)) < 2:
            self.poly = None
            return
        w = 1.0 / (self.P[ok] + 1.0)
        # Reuse fit_quadratic's yn weighting by repeating high-weight points.
        # Cheaper than changing fit_quadratic: pass inverse-P as sample weights
        # via a local polyfit here so locked near-field rows dominate the readout.
        xs = self.xs[ok].astype(np.float64)
        ys = self.ys[ok].astype(np.float64)
        denom = float(y_top) - float(y_bot)
        if abs(denom) < 1e-3:
            self.poly = None
            return
        yn = (ys - float(y_bot)) / denom
        ww = w * np.square(1.0 - 0.40 * np.clip(yn, 0.0, 1.0))
        degree = 2 if xs.size >= 4 else 1
        try:
            raw = np.polyfit(yn, xs, deg=degree, w=ww)
        except (np.linalg.LinAlgError, ValueError):
            self.poly = fit_quadratic(xs, ys, y_bot, y_top, min_points=2)
            return
        coeffs = np.zeros(3, dtype=np.float32)
        coeffs[-raw.size :] = raw.astype(np.float32)
        self.poly = coeffs if np.all(np.isfinite(coeffs)) else None

    def polyline(self) -> Optional[np.ndarray]:
        if self.ys is None:
            return None
        ok = np.isfinite(self.xs)
        if int(np.count_nonzero(ok)) < 2:
            return None
        return np.column_stack([self.xs[ok], self.ys[ok]]).astype(np.float32)


class SideKalman:
    """Independent 3-state Kalman on quadratic coefficients [a, b, c]."""

    Q = np.diag([0.04, 0.8, 4.0]).astype(np.float32)
    P_MAX = np.diag([40.0, 200.0, 800.0]).astype(np.float32)
    CONF_MAX = 24.0
    CONF_HIT = 4.0
    CONF_MISS = 1.0

    def __init__(self, max_poly_a: float = 48.0, max_jump_px: float = 28.0, max_c_step: float = 1.5):
        self.max_poly_a = float(max_poly_a)
        self.max_jump_px = float(max_jump_px)
        self.max_c_step = float(max_c_step)
        self.x: Optional[np.ndarray] = None
        self.P = np.eye(3, dtype=np.float32) * 400.0
        self.confidence: float = 0.0
        self.acquired: bool = False
        self.c_slow: Optional[float] = None

    def reset(self):
        self.x = None
        self.P = np.eye(3, dtype=np.float32) * 400.0
        self.confidence = 0.0
        self.acquired = False
        self.c_slow = None

    def predict(self):
        if self.x is not None:
            self.P = np.minimum(self.P + self.Q, self.P_MAX)

    def eval_x(self, y: float, y_bot: float, y_top: float) -> Optional[float]:
        if self.x is None:
            return None
        return eval_quadratic(self.x, y, y_bot, y_top)

    def miss(self):
        self.confidence = max(0.0, self.confidence - self.CONF_MISS)
        if self.confidence <= 0.0:
            self.reset()

    def hold(self):
        """Coast through an occluded / rejected frame without dropping the lock."""
        if self.valid:
            self.confidence = max(1.0, self.confidence - 0.25)

    @property
    def valid(self) -> bool:
        return self.acquired and self.x is not None and self.confidence > 0.0

    def _clamp_a(self, coeffs: np.ndarray) -> np.ndarray:
        out = coeffs.astype(np.float32).copy()
        lim = self.max_poly_a
        out[0] = float(np.clip(out[0], -lim, lim))
        return out

    def _agrees(self, meas: np.ndarray) -> bool:
        if self.x is None:
            return True
        # Compare x at yn = 0, 0.5, 1 (bottom / mid / top)
        for yn in (0.0, 0.5, 1.0):
            xm = float(meas[0] * yn * yn + meas[1] * yn + meas[2])
            xp = float(self.x[0] * yn * yn + self.x[1] * yn + self.x[2])
            if abs(xm - xp) > self.max_jump_px:
                return False
        return True

    def update_points(
        self,
        pts: List[Tuple[float, float]],
        y_bot: float,
        y_top: float,
        min_points: int = 3,
    ) -> bool:
        if len(pts) < min_points:
            if self.valid:
                self.hold()
                return False
            self.miss()
            return False
        arr = np.asarray(pts, dtype=np.float64)
        meas = fit_quadratic(arr[:, 0], arr[:, 1], y_bot, y_top, min_points=2)
        if meas is None:
            if self.valid:
                self.hold()
                return False
            self.miss()
            return False
        meas = self._clamp_a(meas)
        if not self._agrees(meas):
            self.hold()
            return False
        self._kalman_update(meas, n_pts=len(pts))
        if self.x is not None:
            self.x = self._clamp_a(self.x)
            c = float(self.x[2])
            if self.c_slow is None:
                self.c_slow = c
            else:
                self.c_slow += float(np.clip(c - self.c_slow, -self.max_c_step, self.max_c_step))
            self.x[2] = np.float32(self.c_slow)
        self.confidence = min(self.CONF_MAX, self.confidence + self.CONF_HIT)
        self.acquired = True
        return True

    def _kalman_update(self, meas: np.ndarray, n_pts: int):
        meas = meas.astype(np.float32)
        r_scale = 80.0 / float(max(3, n_pts))
        r = r_scale * np.diag([36.0, 120.0, 500.0]).astype(np.float32)
        if self.x is None:
            self.x = meas.copy()
            self.P = r.copy()
            return
        innov = meas - self.x
        s = self.P + r
        try:
            k = self.P @ np.linalg.inv(s)
        except np.linalg.LinAlgError:
            self.x = 0.85 * self.x + 0.15 * meas
            return
        self.x = self.x + k @ innov
        eye = np.eye(3, dtype=np.float32)
        self.P = (eye - k) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
