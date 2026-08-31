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


def _row_clusters(row: np.ndarray, min_width: int = 1) -> List[float]:
    """Centroids of consecutive nonzero runs in a 1-D mask row."""
    if row.size < 2:
        return []
    mask = row > 0
    if not np.any(mask):
        return []
    padded = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(padded == 1)
    ends = np.flatnonzero(padded == -1)
    out: List[float] = []
    for start, end in zip(starts, ends):
        if end - start >= min_width:
            out.append(0.5 * float(start + end - 1))
    return out


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

    t_left = pred_l if pred_l is not None else da_left
    t_right = pred_r if pred_r is not None else da_right
    if t_left is None:
        t_left = mid_x * 0.55
    if t_right is None:
        t_right = mid_x + 0.45 * (width - mid_x)

    # Prediction exists: tight pixel gate so a car-sized DA jump cannot steal the line.
    gate = 0.14 * width
    if pred_l is not None or pred_r is not None:
        gate = max(8.0, 0.035 * width)

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
    """
    Samples host left/right paint from a lane-line mask.

    Prefers clusters on the drivable-area left/right edges so the ego right
    line is chosen over the next lane. Returns lists of (x, y) in full-frame px.
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
    left_pts: List[Tuple[float, float]] = []
    right_pts: List[Tuple[float, float]] = []
    mid_s = 0.5 * small_w

    for y_s in ys_small:
        yi = int(round(y_s))
        y_img = y_top + (yi + 0.5) * (road_h / float(small_h))
        clusters = _row_clusters(small_ll[yi], min_width=1)
        if not clusters:
            continue
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
            left_pts.append((xl_s / scale, y_img))
        if xr_s is not None:
            right_pts.append((xr_s / scale, y_img))
    return left_pts, right_pts


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
