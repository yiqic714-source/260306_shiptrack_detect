"""
Line detection for MODIS gridded data.

Workflow:
1. Read MYD021 files and interpolate to a base (non-rotated) square grid.
2. Remove 2D large-scale trend using median filtering.
3. Apply morphological enhancement (dilation for Ref, erosion for BT).
4. Second trend removal after morphological enhancement.
5. Fuse: ref_morph_detrend - tb_morph_detrend.
6. Binarize: top 10% of fused image.
7. Skeletonize the binary image.
8. Extract straight lines using probabilistic Hough transform.
9. Group duplicate nearby parallel Hough segments.
   Shorter segments use a larger angle tolerance during duplicate grouping.
10. Replace each duplicate group with one fitted representative line.
11. After duplicate-group fitting, remove line segments shorter than 60 km.
12. Plot only one output panel: BT original image with final fitted lines.
13. Ref and BT are assumed to share the same geolocation mapping, so lat/lon resizing is done once.
"""

import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from netCDF4 import Dataset
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.ndimage import median_filter
from skimage.morphology import skeletonize
from skimage.transform import probabilistic_hough_line


# ============================================================
# Paths
# ============================================================

INPUT_DIR = "/home/chenyiqi/260306_shiptrack_detect/MYD021_SE_Pacific"
LINES_OUT_DIR = "/home/chenyiqi/260306_shiptrack_detect/lines_figs"


# ============================================================
# Target region parameters
# ============================================================

CENTER_LON = -77
CENTER_LAT = -20
SIDE_LENGTH_KM = 750.0
RESOLUTION_M = 1000.0

CROP_MARGIN_DEG = 0.30
EARTH_RADIUS_KM = 6371.0

# Planck constants
h = 6.62607015e-34
c = 2.99792458e8
k = 1.380649e-23


# ============================================================
# File range control, 1-based inclusive
# ============================================================

RUN_FILE_START = 74
RUN_FILE_END = 100


# ============================================================
# Detection settings
# ============================================================

# Median filter kernel size for background removal (must be odd)
TREND_FILTER_SIZE = 101

# Hough transform parameters (probabilistic Hough)
HOUGH_THRESHOLD = 10          # accumulator threshold for probabilistic Hough
HOUGH_MIN_LINE_LENGTH = 45    # minimum line length in pixels for Hough transform
HOUGH_MAX_LINE_GAP = 30       # maximum gap between segments to connect
FINAL_MIN_LINE_LENGTH_KM = 60.0  # remove lines shorter than this after NMS

# Morphological enhancement kernel size
MORPH_KERNEL_SIZE = 20

# Hough line duplicate grouping / fitting settings
# Two Hough segments are treated as duplicates only when all three
# conditions are satisfied:
#   1) similar direction,
#   2) small perpendicular distance,
#   3) overlapping projection or only a small gap along the line direction.
#
# The angle threshold is now length-dependent. Shorter segments are less
# reliable in angle, so they use a larger tolerance. The threshold is computed
# from the shorter one of the two compared line segments:
#   length <= LINE_NMS_SHORT_LENGTH_PX -> LINE_NMS_ANGLE_DEG_SHORT
#   length >= LINE_NMS_LONG_LENGTH_PX  -> LINE_NMS_ANGLE_DEG_LONG
#   intermediate lengths are linearly interpolated.
LINE_NMS_ANGLE_DEG_SHORT = 35.0
LINE_NMS_ANGLE_DEG_LONG = 5.0
LINE_NMS_SHORT_LENGTH_PX = float(HOUGH_MIN_LINE_LENGTH)
LINE_NMS_LONG_LENGTH_PX = 150.0
LINE_NMS_PERP_DIST_PX = 15.0
LINE_NMS_MAX_PROJ_GAP_PX = 30.0


# ============================================================
# MODIS netCDF field paths
# ============================================================

LAT_PATH = "HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Geolocation Fields/Latitude"
LON_PATH = "HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Geolocation Fields/Longitude"
REFSB_500_PATH = "HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_500_Aggr1km_RefSB"
EMISSIVE_PATH = "HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_Emissive"


# ============================================================
# Grid / geolocation helpers (same as line_detect_band.py)
# ============================================================

def lon_km_to_deg(dx_km, lat):
    return dx_km / (EARTH_RADIUS_KM * np.cos(np.deg2rad(lat)) * np.pi / 180.0)


def lat_km_to_deg(dy_km):
    return dy_km / (EARTH_RADIUS_KM * np.pi / 180.0)


def build_base_grid(center_lon, center_lat, side_km, resolution_m):
    """Non-rotated square grid for base maps."""
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
    b = radiance * 1e6
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
# Trend removal and fusion
# ============================================================

def remove_trend_median(data, filter_size):
    """
    Remove 2D large-scale trend using median filtering.
    The background is estimated by a large-kernel median filter,
    then subtracted from the original data.

    Parameters
    ----------
    data : ndarray
        2D array with possible NaN values.
    filter_size : int
        Kernel size for median filter (must be odd).

    Returns
    -------
    residual : ndarray
        Data minus background trend.
    background : ndarray
        Estimated background (median filtered).
    """
    # Fill NaN with the global median for filtering
    global_median = np.nanmedian(data)
    data_filled = np.where(np.isfinite(data), data, global_median)

    # Apply median filter to estimate background
    background = median_filter(data_filled, size=filter_size)

    # Residual
    residual = data - background
    return residual, background


# ============================================================
# Morphological enhancement
# ============================================================

def enhance_morphological(data, erode_small=True, kernel_size=3):
    """
    Enhance features by morphological operations.

    For ref (erode_small=True): apply grey_dilation to the whole image.
        - Small values (dark) get replaced by neighboring larger values → "eroded away"
        - Large values (bright) get expanded → "dilated"
        Net effect: bright features are enhanced, dark features suppressed.

    For BT (erode_small=False): apply grey_erosion to the whole image.
        - Large values (bright) get replaced by neighboring smaller values → "eroded away"
        - Small values (dark) get expanded → "dilated"
        Net effect: dark features are enhanced, bright features suppressed.

    Parameters
    ----------
    data : ndarray
        2D image.
    erode_small : bool
        If True, apply dilation (erode small values, dilate large values).
        If False, apply erosion (erode large values, dilate small values).
    kernel_size : int
        Size of the morphological kernel.

    Returns
    -------
    enhanced : ndarray
        Morphologically enhanced image.
    """
    from scipy.ndimage import grey_erosion, grey_dilation

    # Normalize to [0, 1] first
    valid = np.isfinite(data)
    if not np.any(valid):
        return np.zeros_like(data)
    vmin, vmax = np.nanpercentile(data, 2), np.nanpercentile(data, 98)
    if vmax <= vmin:
        return np.zeros_like(data)
    normed = (data - vmin) / (vmax - vmin)
    normed = np.clip(normed, 0, 1)

    # Fill NaN with 0 for morphological operations
    filled = np.where(np.isfinite(normed), normed, 0.0)

    if erode_small:
        # Dilation: small values get replaced by neighboring larger values
        enhanced = grey_dilation(filled, size=kernel_size)
    else:
        # Erosion: large values get replaced by neighboring smaller values
        enhanced = grey_erosion(filled, size=kernel_size)

    return enhanced


# ============================================================
# Hough line duplicate suppression
# ============================================================

def line_length_pixels(line):
    """Return line-segment length in pixels."""
    (x1, y1), (x2, y2) = line
    return float(np.hypot(x2 - x1, y2 - y1))


def line_angle_deg(line):
    """Return line direction angle in degrees, modulo 180."""
    (x1, y1), (x2, y2) = line
    angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
    return angle % 180.0


def angle_difference_deg(angle1, angle2):
    """Smallest difference between two undirected line angles."""
    diff = abs((angle1 - angle2 + 90.0) % 180.0 - 90.0)
    return float(diff)


def dynamic_angle_threshold_deg(line1, line2,
                                short_angle_deg=LINE_NMS_ANGLE_DEG_SHORT,
                                long_angle_deg=LINE_NMS_ANGLE_DEG_LONG,
                                short_length_px=LINE_NMS_SHORT_LENGTH_PX,
                                long_length_px=LINE_NMS_LONG_LENGTH_PX):
    """
    Return a length-dependent angle threshold for duplicate suppression.

    The shorter segment controls the tolerance. Short Hough segments have less
    stable directions, so they are allowed a larger angle difference.
    """
    len_short = min(line_length_pixels(line1), line_length_pixels(line2))

    if long_length_px <= short_length_px:
        return float(long_angle_deg)

    # Clamp length to the interpolation range.
    len_clamped = min(max(len_short, short_length_px), long_length_px)
    frac = (len_clamped - short_length_px) / (long_length_px - short_length_px)

    # frac = 0 -> short_angle_deg; frac = 1 -> long_angle_deg
    return float(short_angle_deg + frac * (long_angle_deg - short_angle_deg))


def lines_are_duplicates(line_ref, line_new,
                         perp_dist_thresh_px=LINE_NMS_PERP_DIST_PX,
                         max_proj_gap_px=LINE_NMS_MAX_PROJ_GAP_PX):
    """
    Decide whether two Hough line segments are duplicate detections.

    The reference line is usually the longer line that has already been kept.
    A new line is suppressed only if it is:
      1) nearly parallel to the reference line,
      2) close to it in the perpendicular direction,
      3) overlapping with it along the line direction, or separated by only a
         small along-line gap.
    """
    len_ref = line_length_pixels(line_ref)
    len_new = line_length_pixels(line_new)
    if len_ref <= 0 or len_new <= 0:
        return False

    angle_ref = line_angle_deg(line_ref)
    angle_new = line_angle_deg(line_new)
    angle_thresh_deg = dynamic_angle_threshold_deg(line_ref, line_new)
    if angle_difference_deg(angle_ref, angle_new) > angle_thresh_deg:
        return False

    p1 = np.array(line_ref[0], dtype=float)
    p2 = np.array(line_ref[1], dtype=float)
    q1 = np.array(line_new[0], dtype=float)
    q2 = np.array(line_new[1], dtype=float)

    # Unit vector along the reference line and its normal vector.
    u = p2 - p1
    u = u / np.linalg.norm(u)
    n = np.array([-u[1], u[0]], dtype=float)

    # Perpendicular distance between the two segments, measured using the
    # midpoint of the new segment relative to the reference line.
    mid_new = 0.5 * (q1 + q2)
    perp_dist = abs(np.dot(mid_new - p1, n))
    if perp_dist > perp_dist_thresh_px:
        return False

    # Projection intervals along the reference-line direction.
    p_proj = np.array([np.dot(p1 - p1, u), np.dot(p2 - p1, u)])
    q_proj = np.array([np.dot(q1 - p1, u), np.dot(q2 - p1, u)])
    p_min, p_max = float(np.min(p_proj)), float(np.max(p_proj))
    q_min, q_max = float(np.min(q_proj)), float(np.max(q_proj))

    # If intervals overlap, gap = 0. Otherwise, gap is the shortest distance
    # between the two projected intervals.
    proj_gap = max(0.0, max(p_min, q_min) - min(p_max, q_max))
    if proj_gap > max_proj_gap_px:
        return False

    return True


def fit_line_to_group(line_group):
    """
    Fit one representative line segment to a group of duplicate Hough segments.

    All endpoints from the duplicate segments are used. The line direction is
    estimated by PCA / total least squares, and the final segment spans the
    minimum-to-maximum projection of all endpoints along the fitted direction.
    """
    if not line_group:
        return None

    points = []
    for (x1, y1), (x2, y2) in line_group:
        points.append([x1, y1])
        points.append([x2, y2])
    points = np.asarray(points, dtype=float)

    if points.shape[0] < 2:
        return None

    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # Degenerate case: all points nearly identical.
    if np.allclose(centered, 0.0):
        x, y = centroid
        return ((float(x), float(y)), (float(x), float(y)))

    # PCA direction. Vt[0] is the axis with maximum variance.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    norm = np.linalg.norm(direction)
    if norm <= 0:
        return None
    direction = direction / norm

    proj = centered @ direction
    p_start = centroid + np.min(proj) * direction
    p_end = centroid + np.max(proj) * direction

    return ((float(p_start[0]), float(p_start[1])),
            (float(p_end[0]), float(p_end[1])))


def suppress_duplicate_lines(lines_pixels,
                             perp_dist_thresh_px=LINE_NMS_PERP_DIST_PX,
                             max_proj_gap_px=LINE_NMS_MAX_PROJ_GAP_PX):
    """
    Greedy line grouping followed by fitted-line replacement.

    Lines are sorted from long to short. Each line is assigned to the first
    existing group whose representative line is similar in angle, close in
    perpendicular distance, and overlapping/gapped along the line direction.

    Unlike the previous NMS version, a duplicate group is not represented by
    the longest segment. Instead, all endpoints in that group are fitted with
    PCA / total least squares, producing one representative fitted line.
    """
    if not lines_pixels:
        return []

    sorted_lines = sorted(lines_pixels, key=line_length_pixels, reverse=True)

    groups = []
    group_representatives = []

    for line in sorted_lines:
        assigned = False
        for i, representative in enumerate(group_representatives):
            if lines_are_duplicates(
                representative,
                line,
                perp_dist_thresh_px=perp_dist_thresh_px,
                max_proj_gap_px=max_proj_gap_px
            ):
                groups[i].append(line)
                assigned = True
                break

        if not assigned:
            groups.append([line])
            # Keep the longest line as the grouping anchor. The final output
            # for this group will still be the fitted line, not this anchor.
            group_representatives.append(line)

    fitted_lines = []
    for group in groups:
        fitted_line = fit_line_to_group(group)
        if fitted_line is not None and line_length_pixels(fitted_line) > 0:
            fitted_lines.append(fitted_line)

    return fitted_lines


# ============================================================
# Line detection
# ============================================================

def detect_lines(ref_residual, tb_residual, grid_lon, grid_lat,
                 hough_threshold, hough_min_line_length,
                 hough_max_line_gap):
    """
    Apply morphological enhancement, second trend removal, fusion,
    binarization, skeletonization and probabilistic Hough transform.

    Duplicate nearby, nearly parallel and overlapping/gapped Hough segments
    are grouped first. Each duplicate group is then replaced by one fitted
    representative line. Lines shorter than FINAL_MIN_LINE_LENGTH_KM are
    removed after duplicate-group fitting.

    Returns
    -------
    lines_lonlat : list of (lon1, lat1, lon2, lat2)
        Final fitted representative lines in lon/lat coordinates.
    """
    n_rows, n_cols = ref_residual.shape

    # Apply morphological enhancement.
    ref_enhanced = enhance_morphological(
        ref_residual, erode_small=True, kernel_size=MORPH_KERNEL_SIZE)
    tb_enhanced = enhance_morphological(
        tb_residual, erode_small=False, kernel_size=MORPH_KERNEL_SIZE)

    # Second trend removal after morphological enhancement.
    ref_morph_detrend, _ = remove_trend_median(ref_enhanced, TREND_FILTER_SIZE)
    tb_morph_detrend, _ = remove_trend_median(tb_enhanced, TREND_FILTER_SIZE)

    # Fuse: ref_morph_detrend - tb_morph_detrend.
    diff = ref_morph_detrend - tb_morph_detrend

    # Take top pixels as binary.
    threshold_top = np.nanpercentile(diff, 90)
    binary = np.where(np.isfinite(diff) & (diff >= threshold_top), 1, 0)

    # Skeletonize the binary image to get single-pixel-wide lines.
    skeleton = skeletonize(binary.astype(bool))

    # Probabilistic Hough transform on the skeleton.
    # Returns list of line segments: [(x1, y1), (x2, y2)].
    lines_pixels = probabilistic_hough_line(
        skeleton,
        threshold=hough_threshold,
        line_length=hough_min_line_length,
        line_gap=hough_max_line_gap
    )

    # Do not apply the old km-length filter before duplicate suppression.
    candidate_lines_pixels = list(lines_pixels)

    # Group duplicate Hough line segments and replace each group with one fitted line.
    kept_lines_pixels = suppress_duplicate_lines(candidate_lines_pixels)

    # After duplicate suppression, remove short remaining segments.
    final_lines_pixels = []
    for (x1, y1), (x2, y2) in kept_lines_pixels:
        dx_km = (x2 - x1) * RESOLUTION_M / 1000.0
        dy_km = (y2 - y1) * RESOLUTION_M / 1000.0
        length_km = np.hypot(dx_km, dy_km)
        if length_km >= FINAL_MIN_LINE_LENGTH_KM:
            final_lines_pixels.append(((x1, y1), (x2, y2)))

    print(
        f"Hough lines: raw={len(lines_pixels)}, "
        f"after duplicate-group fitting={len(kept_lines_pixels)}, "
        f"after final {FINAL_MIN_LINE_LENGTH_KM:g} km filter={len(final_lines_pixels)}"
    )

    def safe_lookup(y, x):
        yi = int(round(y))
        xi = int(round(x))
        if 0 <= yi < n_rows and 0 <= xi < n_cols:
            return float(grid_lon[yi, xi]), float(grid_lat[yi, xi])
        return None

    lines_lonlat = []
    for (x1, y1), (x2, y2) in final_lines_pixels:
        p1 = safe_lookup(y1, x1)
        p2 = safe_lookup(y2, x2)
        if p1 is not None and p2 is not None:
            lines_lonlat.append((p1[0], p1[1], p2[0], p2[1]))

    return lines_lonlat


# ============================================================
# Plotting
# ============================================================

def get_color_limits(data):
    finite = np.isfinite(data)
    if not np.any(finite):
        return 0, 1
    vmin, vmax = np.nanpercentile(data, 2), np.nanpercentile(data, 98)
    return (0, 1) if (not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin) else (vmin, vmax)


def plot_results(stem, tb_grid, fitted_lines_lonlat, grid_lon, grid_lat):
    """Plot one output panel: BT original image with final fitted lines."""
    tb_vmin, tb_vmax = get_color_limits(tb_grid)
    extent = [grid_lon[0, 0], grid_lon[0, -1], grid_lat[0, 0], grid_lat[-1, 0]]

    # Keep only lines whose endpoints are inside the base grid extent.
    lon_min, lon_max = extent[0], extent[1]
    lat_min, lat_max = extent[2], extent[3]
    filtered_lines = []
    for lon1, lat1, lon2, lat2 in fitted_lines_lonlat:
        if (lon_min <= lon1 <= lon_max and lat_min <= lat1 <= lat_max and
            lon_min <= lon2 <= lon_max and lat_min <= lat2 <= lat_max):
            filtered_lines.append((lon1, lat1, lon2, lat2))

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    im = ax.imshow(
        tb_grid,
        cmap="gray",
        vmin=tb_vmin,
        vmax=tb_vmax,
        extent=extent,
        origin="lower",
        interpolation="none"
    )

    for lon1, lat1, lon2, lat2 in filtered_lines:
        ax.plot([lon1, lon2], [lat1, lat2], "-", color="red", linewidth=1.6)

    ax.set_title(f"BT Diff + fitted lines ({len(filtered_lines)} lines)", fontsize=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(stem, fontsize=12, y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(LINES_OUT_DIR, f"{stem}_bt_fitted_lines.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"Saved BT fitted-line figure for {stem} ({len(filtered_lines)} final lines)")


# ============================================================
# Main processing
# ============================================================

def read_band_raw(dataset, band_name, var_path, scales_attr, offsets_attr):
    """Read one scaled MODIS band without interpolation."""
    var = dataset[var_path]
    idx = get_band_index(var, band_name)
    return read_and_scale_band(var, idx, scales_attr, offsets_attr)


def process_nc_files():
    os.makedirs(LINES_OUT_DIR, exist_ok=True)

    # Build base (non-rotated) grid for base maps.
    base_lon, base_lat, _, _ = build_base_grid(
        CENTER_LON, CENTER_LAT, SIDE_LENGTH_KM, RESOLUTION_M)

    square_corners = np.column_stack([
        [base_lon[0, 0], base_lon[0, -1], base_lon[-1, -1], base_lon[-1, 0]],
        [base_lat[0, 0], base_lat[0, -1], base_lat[-1, -1], base_lat[-1, 0]],
    ])
    square_lon, square_lat = square_corners[:, 0], square_corners[:, 1]

    file_list = load_myd021_file_list(INPUT_DIR)
    start_idx = max(RUN_FILE_START - 1, 0)
    end_idx = None if RUN_FILE_END is None else min(RUN_FILE_END, len(file_list))

    files_to_process = file_list[start_idx:end_idx]
    if len(files_to_process) == 0:
        raise RuntimeError("No files selected. Check RUN_FILE_START and RUN_FILE_END.")

    for hdf_file in files_to_process:
        stem = os.path.splitext(os.path.basename(hdf_file))[0]

        try:
            dataset = Dataset(hdf_file, "r")
        except OSError:
            print(f"Could not open: {hdf_file}")
            continue

        try:
            lat = read_nc_field(dataset, LAT_PATH)
            lon = read_nc_field(dataset, LON_PATH)
            lon = normalize_longitude_if_dateline_crossed(lon)

            if not quick_check_square_in_swath(square_lon, square_lat, lon, lat):
                print(f"Skipped (out of swath): {stem}")
                continue

            # Read raw Ref and BT-difference fields first. Ref and BT are assumed
            # to share the same geolocation mapping, so lon/lat are resized once
            # and reused for both gridded variables.
            ref_raw = read_band_raw(
                dataset, 7, REFSB_500_PATH, "reflectance_scales", "reflectance_offsets")

            emissive_var = dataset[EMISSIVE_PATH]
            rad_11 = read_and_scale_band(emissive_var, get_band_index(emissive_var, 31),
                                         "radiance_scales", "radiance_offsets")
            rad_37 = read_and_scale_band(emissive_var, get_band_index(emissive_var, 20),
                                         "radiance_scales", "radiance_offsets")
            tb_diff = radiance2tb(rad_11, 11.0) - radiance2tb(rad_37, 3.7)

            if ref_raw.shape != tb_diff.shape:
                raise ValueError(
                    f"Ref and BT arrays do not have the same shape for {stem}: "
                    f"ref={ref_raw.shape}, tb={tb_diff.shape}. "
                    "The shared-geolocation shortcut requires equal shapes."
                )

            common_lat = resize_2d(lat, ref_raw.shape)
            common_lon = resize_2d(lon, ref_raw.shape)

            ref_base, ref_valid = resample_to_grid(
                ref_raw, common_lon, common_lat, base_lon, base_lat,
                margin_deg=CROP_MARGIN_DEG)
            tb_base, tb_valid = resample_to_grid(
                tb_diff, common_lon, common_lat, base_lon, base_lat,
                margin_deg=CROP_MARGIN_DEG)

            if not np.all(ref_valid) or not np.all(tb_valid):
                print(f"Skipped (invalid data): {stem}")
                continue

            # Step 1: Remove 2D trend using median filtering.
            ref_residual, _ = remove_trend_median(ref_base, TREND_FILTER_SIZE)
            tb_residual, _ = remove_trend_median(tb_base, TREND_FILTER_SIZE)

            # Step 2: Detect final fitted lines.
            lines_lonlat = detect_lines(
                ref_residual, tb_residual, base_lon, base_lat,
                HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP)

            # Step 3: Plot only BT original image + final fitted lines.
            plot_results(stem, tb_base, lines_lonlat, base_lon, base_lat)

        finally:
            dataset.close()

    print(f"Done processing {len(files_to_process)} selected file(s)")


def main():
    process_nc_files()


if __name__ == "__main__":
    main()