"""
Directional paired-edge centerline detection for MODIS gridded data.

Workflow:
1. Read MYD021 files and interpolate to rotated grid.
2. Detect positive and negative directional edges.
3. Pair opposite-sign parallel edges to get centerline candidates.
4. For each variable, merge PN and NP centerlines separately, then combine them.
5. Combine Reflectance and BT-diff centerlines.
6. Use RANSAC to connect nearby collinear centerline pixels and redraw them as straight 1-pixel lines.

Output combined figure:
Row 1: Reflectance original | positive edges | negative edges | merged centers
Row 2: BT diff original     | positive edges | negative edges | merged centers
Right panel: straightened combined one-pixel centerlines from Reflectance + BT diff
"""

import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from netCDF4 import Dataset
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.ndimage import convolve, map_coordinates
from skimage.draw import line as draw_line
from skimage.measure import label
from skimage.morphology import binary_dilation, disk, thin


# ============================================================
# Paths
# ============================================================

INPUT_DIR = "/home/chenyiqi/260306_shiptrack_detect/MYD021_SE_Pacific"
CACHE_DIR = "/home/chenyiqi/260306_shiptrack_detect/cache"
LINES_OUT_DIR = "/home/chenyiqi/260306_shiptrack_detect/lines_figs"


# ============================================================
# Target region parameters
# ============================================================

CENTER_LON = -75.5
CENTER_LAT = -22.5
SIDE_LENGTH_KM = 750.0
RESOLUTION_M = 4000.0

CROP_MARGIN_DEG = 0.30
EARTH_RADIUS_KM = 6371.0

# Planck constants
h = 6.62607015e-34
c = 2.99792458e8
k = 1.380649e-23


# ============================================================
# File range control, 1-based inclusive
# ============================================================

RUN_FILE_START = 8
RUN_FILE_END = 8


# ============================================================
# Directional edge detection settings
# ============================================================

ANGLE_STEP_DEG = 20
KERNEL_DIRECTIONS_DEG = list(range(0, 180, ANGLE_STEP_DEG))

EDGE_KERNEL_WIDTH_PIX = 10
EDGE_KERNEL_ZERO_GAP_PIX = 0
EDGE_KERNEL_LENGTH_PIX = None

EDGE_RESPONSE_PERCENTILE = 85


# ============================================================
# Paired-center settings
# ============================================================

MIN_PAIR_DISTANCE_PIX = 2
MAX_PAIR_DISTANCE_PIX = int(1.5 * EDGE_KERNEL_WIDTH_PIX)

MERGE_CLOSE_LINE_RADIUS_PIX = 1
MIN_CENTERLINE_LENGTH_PIX = 20


# ============================================================
# RANSAC straight-line consolidation settings
# ============================================================

DO_RANSAC_STRAIGHTEN = True

RANSAC_RANDOM_SEED = 0
RANSAC_MAX_LINES = 80
RANSAC_TRIALS_PER_LINE = 300
RANSAC_DISTANCE_TOL_PIX = 3.0
RANSAC_MIN_POINTS = 30
RANSAC_MIN_LINE_LENGTH_PIX = 50
RANSAC_GAP_TOL_PIX = 12


# ============================================================
# Plot settings
# ============================================================

CENTER_OVERLAY_COLOR = (0.0, 0.0, 0.0)
CENTER_OVERLAY_ALPHA = 0.95


# ============================================================
# MODIS netCDF field paths
# ============================================================

LAT_PATH = "HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Geolocation Fields/Latitude"
LON_PATH = "HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Geolocation Fields/Longitude"
REFSB_500_PATH = "HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_500_Aggr1km_RefSB"
EMISSIVE_PATH = "HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_Emissive"


# ============================================================
# Grid / geolocation helpers
# ============================================================

def lon_km_to_deg(dx_km, lat):
    return dx_km / (EARTH_RADIUS_KM * np.cos(np.deg2rad(lat)) * np.pi / 180.0)


def lat_km_to_deg(dy_km):
    return dy_km / (EARTH_RADIUS_KM * np.pi / 180.0)


def build_square_grid(center_lon, center_lat, side_km, resolution_m):
    n_cells = int(side_km * 1000.0 / resolution_m)
    half = side_km / 2.0
    local_x = np.linspace(-half, half, n_cells)
    local_y = np.linspace(-half, half, n_cells)
    local_xx, local_yy = np.meshgrid(local_x, local_y)

    grid_lon = center_lon + lon_km_to_deg(local_xx, center_lat)
    grid_lat = center_lat + lat_km_to_deg(local_yy)
    return grid_lon, grid_lat, n_cells, n_cells


def quick_check_square_in_swath(square_lon, square_lat, swath_lon, swath_lat):
    finite = np.isfinite(swath_lon) & np.isfinite(swath_lat)
    if not np.any(finite):
        return False

    if (np.nanmin(square_lon) < np.nanmin(swath_lon) or
        np.nanmax(square_lon) > np.nanmax(swath_lon) or
        np.nanmin(square_lat) < np.nanmin(swath_lat) or
        np.nanmax(square_lat) > np.nanmax(swath_lat)):
        return False

    n_rows, n_cols = swath_lon.shape
    swath_poly = np.column_stack([
        [swath_lon[0, 0], swath_lon[0, -1], swath_lon[-1, -1], swath_lon[-1, 0]],
        [swath_lat[0, 0], swath_lat[0, -1], swath_lat[-1, -1], swath_lat[-1, 0]],
    ])
    if not np.all(np.isfinite(swath_poly)):
        return False

    square_pts = np.column_stack([square_lon, square_lat])
    return bool(np.all(Path(swath_poly).contains_points(square_pts, radius=1e-9)))


def normalize_longitude_if_dateline_crossed(lon):
    lon = lon.copy()
    finite = lon[np.isfinite(lon)]
    if finite.size > 0 and np.nanmax(finite) - np.nanmin(finite) > 180.0:
        lon[lon < 0.0] += 360.0
    return lon


def load_myd021_file_list(input_dir):
    files = sorted(glob(os.path.join(input_dir, "*.nc")))
    if not files:
        raise ValueError(f"No nc files found in: {input_dir}")
    return files


def read_nc_field(dataset, field_path):
    """Read a netCDF field and convert masked values to NaN."""
    arr = dataset[field_path][:]
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    return np.asarray(arr, dtype=float)


def parse_band_names(band_names_attr):
    if isinstance(band_names_attr, bytes):
        band_names_attr = band_names_attr.decode("utf-8")
    return [s.strip() for s in str(band_names_attr).split(",")]


def get_band_index(variable, band_name):
    return parse_band_names(variable.getncattr("band_names")).index(str(band_name))


def read_and_scale_band(variable, band_index, scales_attr, offsets_attr):
    data = np.asarray(variable[band_index, :, :], dtype=float)
    if np.ma.isMaskedArray(data):
        data = data.filled(np.nan)
    if "_FillValue" in variable.ncattrs():
        data[data == variable.getncattr("_FillValue")] = np.nan

    scales = np.asarray(variable.getncattr(scales_attr), dtype=float)
    offsets = np.asarray(variable.getncattr(offsets_attr), dtype=float)
    return (data - offsets[band_index]) * scales[band_index]


def radiance2tb(radiance, wavelength_um):
    wavelength_m = wavelength_um * 1e-6
    b = radiance * 1e6  # W m-2 sr-1 um-1 -> W m-2 sr-1 m-1
    tb = np.full_like(radiance, np.nan, dtype=float)
    mask = np.isfinite(b) & (b > 0)
    if np.any(mask):
        c1, c2 = 2.0 * h * c**2, h * c / k
        tb[mask] = c2 / (wavelength_m * np.log(1.0 + c1 / (wavelength_m**5 * b[mask])))
    return tb


def resize_2d(array, target_shape):
    if array.shape == target_shape:
        return array
    src_y = np.arange(array.shape[0], dtype=float)
    src_x = np.arange(array.shape[1], dtype=float)
    tgt_y = np.linspace(0, array.shape[0] - 1, target_shape[0])
    tgt_x = np.linspace(0, array.shape[1] - 1, target_shape[1])
    interp = RegularGridInterpolator((src_y, src_x), array, bounds_error=False, fill_value=None)
    tgt_yy, tgt_xx = np.meshgrid(tgt_y, tgt_x, indexing="ij")
    return interp((tgt_yy, tgt_xx))


def resample_to_grid(data, src_lon, src_lat, tgt_lon, tgt_lat, margin_deg=0.30):
    src_lon, src_lat, data = (np.asarray(x, dtype=float) for x in (src_lon, src_lat, data))

    lon_min, lon_max = np.nanmin(tgt_lon) - margin_deg, np.nanmax(tgt_lon) + margin_deg
    lat_min, lat_max = np.nanmin(tgt_lat) - margin_deg, np.nanmax(tgt_lat) + margin_deg

    near = (np.isfinite(src_lon) & np.isfinite(src_lat) & np.isfinite(data) &
            (src_lon >= lon_min) & (src_lon <= lon_max) &
            (src_lat >= lat_min) & (src_lat <= lat_max))

    if np.count_nonzero(near) < 10:
        return np.full(tgt_lon.shape, np.nan), np.full(tgt_lon.shape, False)

    try:
        interp = LinearNDInterpolator(np.column_stack((src_lon[near], src_lat[near])),
                                       data[near], fill_value=np.nan)
        gridded = interp(np.column_stack((tgt_lon.ravel(), tgt_lat.ravel()))).reshape(tgt_lon.shape)
    except Exception as exc:
        print(f"Interpolation failed: {exc}")
        return np.full(tgt_lon.shape, np.nan), np.full(tgt_lon.shape, False)

    return gridded, np.isfinite(gridded)


# ============================================================
# Basic helpers (edge detection)
# ============================================================

def odd_int(value):
    value = max(int(round(value)), 1)
    return value if value % 2 else value + 1


def get_color_limits(data):
    finite = np.isfinite(data)
    if not np.any(finite):
        return 0, 1
    vmin, vmax = np.nanpercentile(data, 2), np.nanpercentile(data, 98)
    return (0, 1) if (not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin) else (vmin, vmax)


def direction_vectors(angle_deg):
    theta = np.deg2rad(angle_deg)
    return np.cos(theta), np.sin(theta), -np.sin(theta), np.cos(theta)


def sample_shifted(array, dy, dx, order=1, cval=np.nan):
    h, w = array.shape
    yy, xx = np.mgrid[0:h, 0:w]
    return map_coordinates(array, np.array([yy + dy, xx + dx]), order=order, mode="constant", cval=cval)


def sample_mask(mask, dy, dx):
    return sample_shifted(mask.astype(float), dy, dx, order=0, cval=0.0) > 0.5


# ============================================================
# Edge kernel
# ============================================================

def make_edge_kernel(angle_deg):
    edge_width = odd_int(EDGE_KERNEL_WIDTH_PIX)
    gap = max(int(round(EDGE_KERNEL_ZERO_GAP_PIX)), 0)
    kernel_length = odd_int(max(9, edge_width * 8 + 1)) if EDGE_KERNEL_LENGTH_PIX is None else odd_int(EDGE_KERNEL_LENGTH_PIX)

    kernel_size = odd_int(max(kernel_length, 2 * edge_width + gap + 4))
    half = kernel_size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]

    theta = np.deg2rad(angle_deg)
    u = xx * np.cos(theta) + yy * np.sin(theta)
    v = -xx * np.sin(theta) + yy * np.cos(theta)

    half_len, gap_half = kernel_length / 2.0, gap / 2.0
    pos = (v > gap_half) & (v <= gap_half + edge_width) & (np.abs(u) <= half_len)
    neg = (v < -gap_half) & (v >= -gap_half - edge_width) & (np.abs(u) <= half_len)

    n_pos, n_neg = np.count_nonzero(pos), np.count_nonzero(neg)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Invalid edge kernel. Check EDGE_KERNEL_WIDTH_PIX.")

    kernel = np.zeros((kernel_size, kernel_size), dtype=float)
    kernel[pos], kernel[neg] = 1.0 / n_pos, -1.0 / n_neg
    return kernel


# ============================================================
# Centerline processing
# ============================================================

def remove_short_centerlines(center_mask, min_length_pix):
    if center_mask is None or not np.any(center_mask):
        return np.zeros_like(center_mask, dtype=bool)

    labeled = label(center_mask.astype(bool), connectivity=2)
    cleaned = np.zeros_like(center_mask, dtype=bool)
    for lab in range(1, labeled.max() + 1):
        if np.count_nonzero(labeled == lab) >= min_length_pix:
            cleaned |= (labeled == lab)
    return cleaned


def merge_filter_thin_centerlines(center_mask):
    if center_mask is None or not np.any(center_mask):
        return np.zeros_like(center_mask, dtype=bool)
    out = center_mask.astype(bool)
    if MERGE_CLOSE_LINE_RADIUS_PIX > 0:
        out = binary_dilation(out, disk(int(round(MERGE_CLOSE_LINE_RADIUS_PIX))))
    out = thin(out)
    out = remove_short_centerlines(out, MIN_CENTERLINE_LENGTH_PIX)
    return thin(out)


def merge_centerlines_by_angle(center_by_angle, shape):
    final = np.zeros(shape, dtype=bool)
    for mask in center_by_angle.values():
        final |= merge_filter_thin_centerlines(mask)
    return final


def merge_product_centerlines(pn_by_angle, np_by_angle, shape):
    return merge_centerlines_by_angle(pn_by_angle, shape) | merge_centerlines_by_angle(np_by_angle, shape)


# ============================================================
# RANSAC straight-line consolidation
# ============================================================

def points_yx_to_xy(points_yx):
    return np.column_stack((points_yx[:, 1], points_yx[:, 0])).astype(float)


def fit_line_tls(points_yx):
    pts_xy = points_yx_to_xy(points_yx)
    center = np.mean(pts_xy, axis=0)
    if pts_xy.shape[0] < 2:
        return center, np.array([1.0, 0.0])
    _, _, vh = np.linalg.svd(pts_xy - center, full_matrices=False)
    direction = vh[0]
    norm = np.linalg.norm(direction)
    return center, direction / norm if norm > 0 else np.array([1.0, 0.0])


def line_distance_xy(points_yx, point_xy, direction_xy):
    pts_xy = points_yx_to_xy(points_yx)
    dx, dy = pts_xy[:, 0] - point_xy[0], pts_xy[:, 1] - point_xy[1]
    return np.abs(dx * direction_xy[1] - dy * direction_xy[0])


def line_projection_xy(points_yx, point_xy, direction_xy):
    return (points_yx_to_xy(points_yx) - point_xy) @ direction_xy


def draw_line_segment_to_mask(mask, point_xy, direction_xy, t0, t1):
    p0, p1 = point_xy + direction_xy * t0, point_xy + direction_xy * t1
    rr, cc = draw_line(int(round(p0[1])), int(round(p0[0])),
                       int(round(p1[1])), int(round(p1[0])))
    valid = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
    mask[rr[valid], cc[valid]] = True


def draw_inlier_line_segments(mask, inlier_points_yx):
    if inlier_points_yx.shape[0] < RANSAC_MIN_POINTS:
        return 0

    point_xy, direction_xy = fit_line_tls(inlier_points_yx)
    proj = line_projection_xy(inlier_points_yx, point_xy, direction_xy)
    order = np.argsort(proj)
    sorted_proj, sorted_pts = proj[order], inlier_points_yx[order]

    segments = np.split(sorted_pts, np.where(np.diff(sorted_proj) > RANSAC_GAP_TOL_PIX)[0] + 1)
    n_drawn = 0

    for seg in segments:
        if seg.shape[0] < max(5, RANSAC_MIN_POINTS // 3):
            continue
        p, d = fit_line_tls(seg)
        t = line_projection_xy(seg, p, d)
        t0, t1 = np.min(t), np.max(t)
        if (t1 - t0) >= RANSAC_MIN_LINE_LENGTH_PIX:
            draw_line_segment_to_mask(mask, p, d, t0, t1)
            n_drawn += 1
    return n_drawn


def ransac_straighten_centerlines(center_mask):
    if not DO_RANSAC_STRAIGHTEN:
        return thin(center_mask.astype(bool))
    if center_mask is None or not np.any(center_mask):
        return np.zeros_like(center_mask, dtype=bool)

    points_yx = np.column_stack(np.where(center_mask))
    if points_yx.shape[0] < RANSAC_MIN_POINTS:
        return np.zeros_like(center_mask, dtype=bool)

    rng = np.random.default_rng(RANSAC_RANDOM_SEED)
    remaining = np.ones(points_yx.shape[0], dtype=bool)
    output = np.zeros_like(center_mask, dtype=bool)
    n_lines = 0

    while n_lines < RANSAC_MAX_LINES:
        rem_idx = np.where(remaining)[0]
        if rem_idx.size < RANSAC_MIN_POINTS:
            break

        rem_pts = points_yx[rem_idx]
        best_inliers, best_count, best_span = None, 0, 0.0

        for _ in range(RANSAC_TRIALS_PER_LINE):
            if rem_pts.shape[0] < 2:
                break
            i1, i2 = rng.choice(rem_pts.shape[0], size=2, replace=False)
            p1, p2 = rem_pts[i1], rem_pts[i2]
            d = np.array([p2[1] - p1[1], p2[0] - p1[0]], dtype=float)
            norm = np.linalg.norm(d)
            if norm < 5:
                continue
            d = d / norm

            dist = line_distance_xy(rem_pts, np.array([p1[1], p1[0]]), d)
            local = dist <= RANSAC_DISTANCE_TOL_PIX
            count = np.count_nonzero(local)
            if count < RANSAC_MIN_POINTS:
                continue
            span = np.max(line_projection_xy(rem_pts[local], np.array([p1[1], p1[0]]), d)) - \
                   np.min(line_projection_xy(rem_pts[local], np.array([p1[1], p1[0]]), d))
            if span < RANSAC_MIN_LINE_LENGTH_PIX:
                continue
            if count > best_count or (count == best_count and span > best_span):
                best_inliers, best_count, best_span = local, count, span

        if best_inliers is None:
            break

        cand = rem_pts[best_inliers]
        ref_p, ref_d = fit_line_tls(cand)
        ref_dist = line_distance_xy(rem_pts, ref_p, ref_d)
        refined = ref_dist <= RANSAC_DISTANCE_TOL_PIX
        refined_pts = rem_pts[refined]

        if refined_pts.shape[0] < RANSAC_MIN_POINTS:
            break

        n_drawn = draw_inlier_line_segments(output, refined_pts)
        remaining[rem_idx[refined]] = False
        if n_drawn > 0:
            n_lines += n_drawn

    output = thin(output)
    output = remove_short_centerlines(output, MIN_CENTERLINE_LENGTH_PIX)
    return thin(output)


# ============================================================
# Edge detection and pairing
# ============================================================

def edge_peak_masks(response, angle_deg, threshold):
    valid = np.isfinite(response)
    if not np.any(valid):
        return np.zeros_like(response, dtype=bool), np.zeros_like(response, dtype=bool)

    _, _, dx_norm, dy_norm = direction_vectors(angle_deg)
    r_plus = sample_shifted(response, dy_norm, dx_norm, order=1, cval=np.nan)
    r_minus = sample_shifted(response, -dy_norm, -dx_norm, order=1, cval=np.nan)

    pos = valid & (response > threshold) & (response >= r_plus) & (response >= r_minus)
    neg = valid & (response < -threshold) & (response <= r_plus) & (response <= r_minus)
    return pos, neg


def paired_centers_for_angle(pos_edges, neg_edges, angle_deg):
    _, _, dx_norm, dy_norm = direction_vectors(angle_deg)
    centers_pn = np.zeros_like(pos_edges, dtype=bool)
    centers_np = np.zeros_like(pos_edges, dtype=bool)

    for dist in range(MIN_PAIR_DISTANCE_PIX, MAX_PAIR_DISTANCE_PIX + 1):
        half = dist / 2.0
        dy, dx = dy_norm * half, dx_norm * half
        pp, pm = sample_mask(pos_edges, dy, dx), sample_mask(pos_edges, -dy, -dx)
        np_, nm = sample_mask(neg_edges, dy, dx), sample_mask(neg_edges, -dy, -dx)
        centers_pn |= pp & nm
        centers_np |= np_ & pm
    return centers_pn, centers_np


def detect_directional_edges(data):
    finite = np.isfinite(data)
    if not np.any(finite):
        return np.zeros_like(data, dtype=bool), np.zeros_like(data, dtype=bool), np.zeros_like(data, dtype=bool)

    data_filled = np.where(finite, data, np.nanmedian(data[finite]))
    all_abs = []
    pn_by_angle, np_by_angle = {}, {}
    all_pos = np.zeros_like(data, dtype=bool)
    all_neg = np.zeros_like(data, dtype=bool)

    for angle in KERNEL_DIRECTIONS_DEG:
        response = convolve(data_filled, make_edge_kernel(angle), mode="nearest")
        response[~finite] = np.nan

        pos, neg = edge_peak_masks(response, angle, 0)
        all_pos |= pos
        all_neg |= neg

        valid_resp = np.isfinite(response)
        if np.any(valid_resp):
            all_abs.append(np.abs(response[valid_resp]))

    if not all_abs:
        return np.zeros_like(data, dtype=bool), np.zeros_like(data, dtype=bool), np.zeros_like(data, dtype=bool)

    threshold = np.nanpercentile(np.concatenate(all_abs), EDGE_RESPONSE_PERCENTILE)

    for angle in KERNEL_DIRECTIONS_DEG:
        response = convolve(data_filled, make_edge_kernel(angle), mode="nearest")
        response[~finite] = np.nan

        pos, neg = edge_peak_masks(response, angle, threshold)
        cp, cn = paired_centers_for_angle(pos, neg, angle)
        cp &= finite
        cn &= finite
        pn_by_angle[angle] = cp
        np_by_angle[angle] = cn

    centers = merge_product_centerlines(pn_by_angle, np_by_angle, data.shape) & finite
    return all_pos, all_neg, centers


# ============================================================
# Plotting
# ============================================================

def overlay_mask(ax, mask, extent, color=(0, 0, 0), alpha=0.95):
    if mask is None or not np.any(mask):
        return
    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[..., :3] = color
    rgba[..., 3] = mask.astype(float) * alpha
    ax.imshow(rgba, extent=extent, origin="lower", interpolation="none")


def plot_row(ax_row, grid, cmap, vmin, vmax, title, pos_edges, neg_edges, extent):
    """Plot one row: original | positive edges | negative edges, with lon/lat axes."""
    im = ax_row[0].imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax,
                          extent=extent, origin="lower", interpolation="none")
    ax_row[0].set_title(title)
    ax_row[0].set_xlabel("Longitude")
    ax_row[0].set_ylabel("Latitude")

    ax_row[1].imshow(pos_edges, cmap="gray", extent=extent, origin="lower", interpolation="none")
    ax_row[1].set_title(f"{title} positive edges")
    ax_row[1].set_xlabel("Longitude")
    ax_row[1].set_ylabel("Latitude")

    ax_row[2].imshow(neg_edges, cmap="gray", extent=extent, origin="lower", interpolation="none")
    ax_row[2].set_title(f"{title} negative edges")
    ax_row[2].set_xlabel("Longitude")
    ax_row[2].set_ylabel("Latitude")

    plt.colorbar(im, ax=ax_row[0], fraction=0.046, pad=0.04)


def plot_combined(stem, ref_grid, tb_grid, ref_pos, ref_neg, ref_centers,
                  tb_pos, tb_neg, tb_centers, combined_raw, combined_straight,
                  grid_lon, grid_lat):
    ref_vmin, ref_vmax = get_color_limits(ref_grid)
    tb_vmin, tb_vmax = get_color_limits(tb_grid)

    # extent = [lon_left, lon_right, lat_bottom, lat_top]
    extent = [grid_lon[0, 0], grid_lon[0, -1], grid_lat[0, 0], grid_lat[-1, 0]]

    fig = plt.figure(figsize=(38, 18), dpi=200)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.15, hspace=0.12)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
    ax_combined = fig.add_subplot(gs[:, 3])

    plot_row(axes[0], ref_grid, "jet", ref_vmin, ref_vmax, "2.1 um Reflectance", ref_pos, ref_neg, extent)
    plot_row(axes[1], tb_grid, "RdBu_r", tb_vmin, tb_vmax, "BT Diff: 11 um - 3.7 um", tb_pos, tb_neg, extent)

    im = ax_combined.imshow(tb_grid, cmap="RdBu_r", vmin=tb_vmin, vmax=tb_vmax,
                            extent=extent, origin="lower", interpolation="none")
    overlay_mask(ax_combined, combined_straight, extent,
                 color=CENTER_OVERLAY_COLOR, alpha=CENTER_OVERLAY_ALPHA)
    ax_combined.set_title(
        f"BT Diff + straightened centers\nraw={np.count_nonzero(combined_raw)} px, "
        f"straight={np.count_nonzero(combined_straight)} px")
    ax_combined.set_xlabel("Longitude")
    ax_combined.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax_combined, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"File: {stem} | edge width={EDGE_KERNEL_WIDTH_PIX} pix | "
        f"angle step={ANGLE_STEP_DEG}° | percentile={EDGE_RESPONSE_PERCENTILE} | "
        f"pair dist={MIN_PAIR_DISTANCE_PIX}-{MAX_PAIR_DISTANCE_PIX} pix | "
        f"RANSAC tol={RANSAC_DISTANCE_TOL_PIX} pix",
        fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = os.path.join(LINES_OUT_DIR, f"{stem}_res{RESOLUTION_M}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# Main processing
# ============================================================

def read_band_and_grid(dataset, band_name, var_path, scales_attr, offsets_attr,
                       lat, lon, grid_lon, grid_lat):
    """Read a MODIS band, resize geolocation, and resample to target grid."""
    var = dataset[var_path]
    idx = get_band_index(var, band_name)
    data = read_and_scale_band(var, idx, scales_attr, offsets_attr)
    lat_resized = resize_2d(lat, data.shape)
    lon_resized = resize_2d(lon, data.shape)
    return resample_to_grid(data, lon_resized, lat_resized, grid_lon, grid_lat, margin_deg=CROP_MARGIN_DEG)


def process_one_file(stem, ref_grid, tb_grid, grid_lon, grid_lat):
    ref_pos, ref_neg, ref_centers = detect_directional_edges(ref_grid)
    tb_pos, tb_neg, tb_centers = detect_directional_edges(tb_grid)

    combined_raw = ref_centers | tb_centers
    combined_straight = ransac_straighten_centerlines(combined_raw)

    plot_combined(stem, ref_grid, tb_grid, ref_pos, ref_neg, ref_centers,
                  tb_pos, tb_neg, tb_centers, combined_raw, combined_straight,
                  grid_lon, grid_lat)
    return np.count_nonzero(combined_straight)


def process_nc_files():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(LINES_OUT_DIR, exist_ok=True)

    grid_lon, grid_lat, _, _ = build_square_grid(
        CENTER_LON, CENTER_LAT, SIDE_LENGTH_KM, RESOLUTION_M)

    square_corners = np.column_stack([
        [grid_lon[0, 0], grid_lon[0, -1], grid_lon[-1, -1], grid_lon[-1, 0]],
        [grid_lat[0, 0], grid_lat[0, -1], grid_lat[-1, -1], grid_lat[-1, 0]],
    ])
    square_lon, square_lat = square_corners[:, 0], square_corners[:, 1]

    file_list = load_myd021_file_list(INPUT_DIR)
    start_idx = max(RUN_FILE_START - 1, 0)
    end_idx = None if RUN_FILE_END is None else min(RUN_FILE_END, len(file_list))

    files_to_process = file_list[start_idx:end_idx]
    if len(files_to_process) == 0:
        raise RuntimeError("No files selected. Check RUN_FILE_START and RUN_FILE_END.")

    total_center_pixels = 0
    n_processed = 0
    n_skipped = 0

    for hdf_file in files_to_process:
        stem = os.path.splitext(os.path.basename(hdf_file))[0]

        try:
            dataset = Dataset(hdf_file, "r")
        except OSError:
            n_skipped += 1
            continue

        try:
            lat = read_nc_field(dataset, LAT_PATH)
            lon = read_nc_field(dataset, LON_PATH)
            lon = normalize_longitude_if_dateline_crossed(lon)

            if not quick_check_square_in_swath(square_lon, square_lat, lon, lat):
                n_skipped += 1
                continue

            ref_grid, ref_valid = read_band_and_grid(
                dataset, 7, REFSB_500_PATH, "reflectance_scales", "reflectance_offsets",
                lat, lon, grid_lon, grid_lat)

            emissive_var = dataset[EMISSIVE_PATH]
            rad_11 = read_and_scale_band(emissive_var, get_band_index(emissive_var, 31),
                                         "radiance_scales", "radiance_offsets")
            rad_37 = read_and_scale_band(emissive_var, get_band_index(emissive_var, 20),
                                         "radiance_scales", "radiance_offsets")
            tb_diff = radiance2tb(rad_11, 11.0) - radiance2tb(rad_37, 3.7)

            lat_tb = resize_2d(lat, tb_diff.shape)
            lon_tb = resize_2d(lon, tb_diff.shape)
            tb_grid, tb_valid = resample_to_grid(tb_diff, lon_tb, lat_tb, grid_lon, grid_lat,
                                                  margin_deg=CROP_MARGIN_DEG)

            if not np.all(ref_valid) or not np.all(tb_valid):
                n_skipped += 1
                continue

            np.save(os.path.join(CACHE_DIR, f"{stem}_ref_2.1um.npy"), ref_grid)
            np.save(os.path.join(CACHE_DIR, f"{stem}_tb11_minus_tb3.7.npy"), tb_grid)

            total_center_pixels += process_one_file(stem, ref_grid, tb_grid, grid_lon, grid_lat)
            n_processed += 1

        finally:
            dataset.close()

    print(f"Processed={n_processed}, skipped={n_skipped}, straight_center_pixels={total_center_pixels}")


def main():
    process_nc_files()


if __name__ == "__main__":
    main()
   