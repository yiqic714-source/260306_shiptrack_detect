"""
Band-based symmetric centerline detection for MODIS gridded data.

Workflow:
1. Read MYD021 files and interpolate to a base (non-rotated) square grid.
2. For each angle θ in [0, 20, ..., 160]:
   a. Build a rotated grid where rows = along-axis (θ), cols = cross-axis (θ+90°).
   b. Interpolate Reflectance and BT-diff data onto the rotated grid.
   c. Divide the along-axis direction into bands, sum each band along cross-axis → 1D curves.
   d. Find symmetric points (negative correlation peaks) on each 1D curve.
   e. Match symmetric points across adjacent bands; fit straight lines to matched groups.
   f. Map fitted lines back to lon/lat coordinates.
3. Plot 4 figures:
   - Base map 1: Reflectance 2.1um (non-rotated)
   - Base map 2: BT diff (non-rotated)
   - Base map 3: Reflectance + detected lines from all angles
   - Base map 4: BT diff + detected lines from all angles
"""

import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from netCDF4 import Dataset
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.signal import detrend


# ============================================================
# Paths
# ============================================================

INPUT_DIR = "/home/chenyiqi/260306_shiptrack_detect/MYD021_SE_Pacific"
CACHE_DIR = "/home/chenyiqi/260306_shiptrack_detect/cache"
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

RUN_FILE_START = 8
RUN_FILE_END = 8


# ============================================================
# Angle settings
# ============================================================

ANGLE_STEP_DEG = 20
ANGLES_DEG = list(range(0, 180, ANGLE_STEP_DEG))  # [0, 20, ..., 160]


# ============================================================
# Band / symmetry detection settings
# ============================================================

BAND_WIDTH_PIX = 75          # number of along-axis rows per band
SYMMETRY_WINDOW_PIX = 13     # half-window for correlation (in cross-axis pixels)
MIN_SYMMETRY_DIST_PIX = 8    # minimum distance between symmetric points
N_TOP_SYMMETRY = 10           # max number of symmetric points per 1D curve
ADJACENT_BAND_TOL_PIX = 5    # max cross-axis offset for matching adjacent bands
MIN_CONSECUTIVE_BANDS = 2    # min consecutive bands to confirm a line


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


def build_rotated_grid(center_lon, center_lat, side_km, resolution_m, angle_deg):
    """
    Build a square grid rotated by `angle_deg`.
    Grid rows run along the `angle_deg` direction (along-axis).
    Grid columns run perpendicular (cross-axis, angle_deg + 90°).
    """
    n_cells = int(side_km * 1000.0 / resolution_m)
    half = side_km / 2.0
    # cross-axis coordinate (columns)
    local_x = np.linspace(-half, half, n_cells)
    # along-axis coordinate (rows)
    local_y = np.linspace(-half, half, n_cells)
    local_xx, local_yy = np.meshgrid(local_x, local_y)

    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    # Rotate: (local_xx, local_yy) by angle_deg
    rot_x = local_xx * cos_a - local_yy * sin_a
    rot_y = local_xx * sin_a + local_yy * cos_a

    grid_lon = center_lon + lon_km_to_deg(rot_x, center_lat)
    grid_lat = center_lat + lat_km_to_deg(rot_y)
    return grid_lon, grid_lat, n_cells, n_cells


def build_base_grid(center_lon, center_lat, side_km, resolution_m):
    """Non-rotated square grid for base maps."""
    return build_rotated_grid(center_lon, center_lat, side_km, resolution_m, 0.0)


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
# Band processing
# ============================================================

def sum_bands_to_1d(data_rot, band_width_pix):
    """
    Divide the along-axis (rows) of data_rot into bands.
    Sum each band along the cross-axis (columns) to produce 1D curves.

    Parameters
    ----------
    data_rot : ndarray, shape (n_along, n_cross)
        Data on the rotated grid (rows=along-axis, cols=cross-axis).
    band_width_pix : int
        Number of along-axis rows per band.

    Returns
    -------
    curves : list of 1D ndarrays
        Each curve is the cross-axis sum for one band.
    band_centers : list of float
        Along-axis center index of each band (in pixel coordinates).
    band_edges : list of (float, float)
        Along-axis start and end index of each band (tangent positions).
    """
    n_along, n_cross = data_rot.shape
    curves = []
    band_centers = []
    band_edges = []
    for start in range(0, n_along, band_width_pix):
        end = min(start + band_width_pix, n_along)
        band_data = data_rot[start:end, :]
        # Sum along cross-axis, then normalize by number of valid pixels
        valid = np.isfinite(band_data)
        col_sum = np.nansum(band_data, axis=0)
        col_count = np.sum(valid, axis=0)
        col_count = np.maximum(col_count, 1)  # avoid division by zero
        curve = col_sum / col_count
        curves.append(curve)
        band_centers.append((start + end - 1) / 2.0)
        band_edges.append((float(start), float(end - 1)))
    return curves, band_centers, band_edges


def find_symmetric_points_1d(curve, cross_centers_km, window_pix, min_distance_pix, n_top):
    """
    Find symmetric points (negative correlation peaks) in a 1D curve.

    Parameters
    ----------
    curve : 1D ndarray
        The 1D cross-axis profile.
    cross_centers_km : 1D ndarray
        Cross-axis coordinate values in km.
    window_pix : int
        Half-window size for correlation calculation.
    min_distance_pix : int
        Minimum distance between detected symmetric points.
    n_top : int
        Maximum number of top symmetric points to return.

    Returns
    -------
    top_indices : list of int
        Indices of symmetric points in the curve.
    top_corrs : list of float
        Correlation values at those points.
    """
    n = len(curve)
    if n < 2 * window_pix + 3:
        return [], []

    # Detrend the entire curve
    mask_all = np.isfinite(curve)
    curve_detrend = curve.copy()
    if mask_all.sum() > 0:
        curve_detrend[mask_all] = detrend(curve[mask_all], type='linear')

    # Compute negative correlation at each valid position
    neg_corr = np.full(n, np.nan)
    for i in range(window_pix, n - window_pix):
        left = curve_detrend[i - window_pix:i]
        right = curve_detrend[i + 1:i + 1 + window_pix]
        mask = np.isfinite(left) & np.isfinite(right)
        if mask.sum() >= 5:
            corr = np.corrcoef(left[mask], right[mask])[0, 1]
            if np.isfinite(corr):
                neg_corr[i] = corr

    # Find top negative correlations with minimum distance constraint
    valid_idx = np.where(np.isfinite(neg_corr))[0]
    if len(valid_idx) == 0:
        return [], []

    # Sort by most negative correlation (ascending)
    sorted_idx = valid_idx[np.argsort(neg_corr[valid_idx])]
    top_idx = []
    top_corrs = []
    for idx in sorted_idx:
        if all(abs(idx - exist) > min_distance_pix for exist in top_idx):
            top_idx.append(int(idx))
            top_corrs.append(float(neg_corr[idx]))
        if len(top_idx) == n_top:
            break

    return top_idx, top_corrs


def match_adjacent_bands(all_band_sym_points, tol_pix, min_consecutive):
    """
    Match symmetric points across adjacent bands to form line candidates.

    Parameters
    ----------
    all_band_sym_points : list of list of (cross_idx, corr)
        For each band, a list of (cross-axis index, correlation).
    tol_pix : int
        Maximum cross-axis offset for matching adjacent bands.
    min_consecutive : int
        Minimum number of consecutive bands to form a line.

    Returns
    -------
    line_groups : list of dict
        Each dict has:
          - 'band_indices': list of band indices in the group
          - 'cross_indices': list of cross-axis indices (one per band)
          - 'corrs': list of correlation values
    """
    n_bands = len(all_band_sym_points)
    if n_bands == 0:
        return []

    # Greedy matching: start from each band, propagate forward
    used = [set() for _ in range(n_bands)]  # track used points per band
    line_groups = []

    for start_band in range(n_bands):
        for start_pt_idx, (start_cross, start_corr) in enumerate(all_band_sym_points[start_band]):
            if start_pt_idx in used[start_band]:
                continue

            # Try to build a chain forward
            chain_bands = [start_band]
            chain_cross = [start_cross]
            chain_corrs = [start_corr]
            used[start_band].add(start_pt_idx)

            current_cross = start_cross
            for b in range(start_band + 1, n_bands):
                # Find the closest symmetric point in band b to current_cross
                best_match = None
                best_dist = tol_pix + 1
                for pt_idx, (cross, corr) in enumerate(all_band_sym_points[b]):
                    if pt_idx in used[b]:
                        continue
                    dist = abs(cross - current_cross)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = (pt_idx, cross, corr)

                if best_match is None or best_dist > tol_pix:
                    break

                pt_idx, cross, corr = best_match
                chain_bands.append(b)
                chain_cross.append(cross)
                chain_corrs.append(corr)
                used[b].add(pt_idx)
                current_cross = cross

            if len(chain_bands) >= min_consecutive:
                line_groups.append({
                    'band_indices': chain_bands,
                    'cross_indices': chain_cross,
                    'corrs': chain_corrs,
                })

    return line_groups


def fit_line_from_group(group, band_centers, band_edges, cross_centers_km, n_along, n_cross):
    """
    Fit a straight line to matched symmetric points and compute endpoints.

    The line is defined in the rotated grid's pixel coordinate system:
      - x = cross-axis pixel index
      - y = along-axis pixel index

    Endpoints are extended to the outermost band tangents (edges/boundaries).

    Parameters
    ----------
    group : dict
        Matched band group from match_adjacent_bands.
    band_centers : list of float
        Along-axis center index of each band.
    band_edges : list of (float, float)
        Along-axis start and end index of each band (tangent positions).
    cross_centers_km : 1D ndarray
        Cross-axis coordinate values in km.
    n_along : int
        Number of along-axis pixels in the rotated grid.
    n_cross : int
        Number of cross-axis pixels in the rotated grid.

    Returns
    -------
    line_coords : tuple of (y1, x1, y2, x2) in pixel coordinates, or None if fitting fails.
    """
    band_idx = group['band_indices']
    cross_idx = group['cross_indices']

    if len(band_idx) < 2:
        return None

    # Points: (along_center, cross_center) in pixel coordinates
    y_pts = np.array([band_centers[b] for b in band_idx])
    x_pts = np.array(cross_idx)

    # Fit line: x = a * y + b  (cross-axis as function of along-axis)
    valid = np.isfinite(y_pts) & np.isfinite(x_pts)
    if np.count_nonzero(valid) < 2:
        return None

    y_valid = y_pts[valid]
    x_valid = x_pts[valid]

    A = np.column_stack([y_valid, np.ones_like(y_valid)])
    try:
        a, b = np.linalg.lstsq(A, x_valid, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    # Endpoints: extend to the outermost band tangents (edges)
    y1 = band_edges[band_idx[0]][0]   # start edge of first band
    y2 = band_edges[band_idx[-1]][1]  # end edge of last band
    x1 = a * y1 + b
    x2 = a * y2 + b

    # Clamp to grid bounds
    x1 = np.clip(x1, 0, n_cross - 1)
    x2 = np.clip(x2, 0, n_cross - 1)

    return (y1, x1, y2, x2)


def map_line_to_lonlat(line_pixels, grid_lon, grid_lat):
    """
    Map a line from rotated-grid pixel coordinates to lon/lat.

    Parameters
    ----------
    line_pixels : tuple (y1, x1, y2, x2)
        Line endpoints in pixel coordinates (along, cross).
    grid_lon : ndarray
        Longitude grid of the rotated grid.
    grid_lat : ndarray
        Latitude grid of the rotated grid.

    Returns
    -------
    (lon1, lat1, lon2, lat2) or None if out of bounds.
    """
    y1, x1, y2, x2 = line_pixels
    n_rows, n_cols = grid_lon.shape

    def safe_lookup(y, x):
        yi = int(round(y))
        xi = int(round(x))
        if 0 <= yi < n_rows and 0 <= xi < n_cols:
            return float(grid_lon[yi, xi]), float(grid_lat[yi, xi])
        return None

    p1 = safe_lookup(y1, x1)
    p2 = safe_lookup(y2, x2)
    if p1 is None or p2 is None:
        return None
    return (p1[0], p1[1], p2[0], p2[1])


def detect_lines_at_angle(ref_rot, tb_rot, grid_lon, grid_lat,
                          band_width_pix, window_pix, min_dist_pix, n_top,
                          tol_pix, min_consecutive):
    """
    Detect lines in a given rotated grid using band-based symmetric point matching.

    Parameters
    ----------
    ref_rot : ndarray
        Reflectance on rotated grid.
    tb_rot : ndarray
        BT diff on rotated grid.
    grid_lon, grid_lat : ndarray
        Lon/lat of the rotated grid.
    ... (detection parameters)

    Returns
    -------
    lines_lonlat : list of (lon1, lat1, lon2, lat2)
        Detected lines in lon/lat coordinates.
    """
    n_along, n_cross = ref_rot.shape
    cross_centers_km = np.linspace(-SIDE_LENGTH_KM / 2, SIDE_LENGTH_KM / 2, n_cross)

    lines_lonlat = []

    for data_rot, name in [(ref_rot, 'ref'), (tb_rot, 'tb')]:
        curves, band_centers, band_edges = sum_bands_to_1d(data_rot, band_width_pix)

        # Find symmetric points in each band
        all_band_sym_points = []
        for curve in curves:
            top_idx, top_corrs = find_symmetric_points_1d(
                curve, cross_centers_km, window_pix, min_dist_pix, n_top)
            all_band_sym_points.append(list(zip(top_idx, top_corrs)))

        # Match across adjacent bands
        groups = match_adjacent_bands(all_band_sym_points, tol_pix, min_consecutive)

        # Fit lines and map to lon/lat
        for group in groups:
            line_pixels = fit_line_from_group(
                group, band_centers, band_edges, cross_centers_km, n_along, n_cross)
            if line_pixels is None:
                continue
            line_lonlat = map_line_to_lonlat(line_pixels, grid_lon, grid_lat)
            if line_lonlat is not None:
                lines_lonlat.append(line_lonlat)

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


def plot_results(stem, ref_grid, tb_grid, all_lines_lonlat, grid_lon, grid_lat):
    """
    Plot 4 figures:
      1. Reflectance base map
      2. BT diff base map
      3. Reflectance + detected lines
      4. BT diff + detected lines
    """
    ref_vmin, ref_vmax = get_color_limits(ref_grid)
    tb_vmin, tb_vmax = get_color_limits(tb_grid)

    extent = [grid_lon[0, 0], grid_lon[0, -1], grid_lat[0, 0], grid_lat[-1, 0]]

    # ---- Figure 1: Reflectance base ----
    fig1, ax1 = plt.subplots(figsize=(10, 8), dpi=200)
    im1 = ax1.imshow(ref_grid, cmap="jet", vmin=ref_vmin, vmax=ref_vmax,
                     extent=extent, origin="lower", interpolation="none")
    ax1.set_title(f"{stem} - Reflectance 2.1um")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig1.tight_layout()
    fig1.savefig(os.path.join(LINES_OUT_DIR, f"{stem}_band_ref_base.png"),
                 bbox_inches="tight", dpi=200)
    plt.close(fig1)

    # ---- Figure 2: BT diff base ----
    fig2, ax2 = plt.subplots(figsize=(10, 8), dpi=200)
    im2 = ax2.imshow(tb_grid, cmap="RdBu_r", vmin=tb_vmin, vmax=tb_vmax,
                     extent=extent, origin="lower", interpolation="none")
    ax2.set_title(f"{stem} - BT Diff 11um - 3.7um")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    fig2.savefig(os.path.join(LINES_OUT_DIR, f"{stem}_band_tb_base.png"),
                 bbox_inches="tight", dpi=200)
    plt.close(fig2)

    # ---- Figure 3: Reflectance + lines ----
    fig3, ax3 = plt.subplots(figsize=(10, 8), dpi=200)
    ax3.imshow(ref_grid, cmap="jet", vmin=ref_vmin, vmax=ref_vmax,
               extent=extent, origin="lower", interpolation="none")
    for lon1, lat1, lon2, lat2 in all_lines_lonlat:
        ax3.plot([lon1, lon2], [lat1, lat2], '-', color='black', linewidth=1.5)
    ax3.set_title(f"{stem} - Reflectance + detected lines ({len(all_lines_lonlat)} lines)")
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    fig3.tight_layout()
    fig3.savefig(os.path.join(LINES_OUT_DIR, f"{stem}_band_ref_lines.png"),
                 bbox_inches="tight", dpi=200)
    plt.close(fig3)

    # ---- Figure 4: BT diff + lines ----
    fig4, ax4 = plt.subplots(figsize=(10, 8), dpi=200)
    ax4.imshow(tb_grid, cmap="RdBu_r", vmin=tb_vmin, vmax=tb_vmax,
               extent=extent, origin="lower", interpolation="none")
    for lon1, lat1, lon2, lat2 in all_lines_lonlat:
        ax4.plot([lon1, lon2], [lat1, lat2], '-', color='black', linewidth=1.5)
    ax4.set_title(f"{stem} - BT Diff + detected lines ({len(all_lines_lonlat)} lines)")
    ax4.set_xlabel("Longitude")
    ax4.set_ylabel("Latitude")
    fig4.tight_layout()
    fig4.savefig(os.path.join(LINES_OUT_DIR, f"{stem}_band_tb_lines.png"),
                 bbox_inches="tight", dpi=200)
    plt.close(fig4)

    print(f"Saved 4 figures for {stem} ({len(all_lines_lonlat)} lines)")


# ============================================================
# Main processing
# ============================================================

def read_band_and_grid(dataset, band_name, var_path, scales_attr, offsets_attr,
                       lat, lon, grid_lon, grid_lat):
    var = dataset[var_path]
    idx = get_band_index(var, band_name)
    data = read_and_scale_band(var, idx, scales_attr, offsets_attr)
    lat_resized = resize_2d(lat, data.shape)
    lon_resized = resize_2d(lon, data.shape)
    return resample_to_grid(data, lon_resized, lat_resized, grid_lon, grid_lat, margin_deg=CROP_MARGIN_DEG)


def process_nc_files():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(LINES_OUT_DIR, exist_ok=True)

    # Build base (non-rotated) grid for base maps
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

            # Read and interpolate to base grid for base maps
            ref_base, ref_valid = read_band_and_grid(
                dataset, 7, REFSB_500_PATH, "reflectance_scales", "reflectance_offsets",
                lat, lon, base_lon, base_lat)

            emissive_var = dataset[EMISSIVE_PATH]
            rad_11 = read_and_scale_band(emissive_var, get_band_index(emissive_var, 31),
                                         "radiance_scales", "radiance_offsets")
            rad_37 = read_and_scale_band(emissive_var, get_band_index(emissive_var, 20),
                                         "radiance_scales", "radiance_offsets")
            tb_diff = radiance2tb(rad_11, 11.0) - radiance2tb(rad_37, 3.7)

            lat_tb = resize_2d(lat, tb_diff.shape)
            lon_tb = resize_2d(lon, tb_diff.shape)
            tb_base, tb_valid = resample_to_grid(tb_diff, lon_tb, lat_tb, base_lon, base_lat,
                                                  margin_deg=CROP_MARGIN_DEG)

            if not np.all(ref_valid) or not np.all(tb_valid):
                print(f"Skipped (invalid data): {stem}")
                continue

            # Detect lines at each angle
            all_lines_lonlat = []

            for angle in ANGLES_DEG:
                # Build rotated grid for this angle
                rot_lon, rot_lat, _, _ = build_rotated_grid(
                    CENTER_LON, CENTER_LAT, SIDE_LENGTH_KM, RESOLUTION_M, angle)

                # Interpolate data onto rotated grid
                ref_rot, _ = resample_to_grid(ref_base, base_lon, base_lat,
                                               rot_lon, rot_lat, margin_deg=CROP_MARGIN_DEG)
                tb_rot, _ = resample_to_grid(tb_base, base_lon, base_lat,
                                              rot_lon, rot_lat, margin_deg=CROP_MARGIN_DEG)

                # Detect lines at this angle
                lines = detect_lines_at_angle(
                    ref_rot, tb_rot, rot_lon, rot_lat,
                    BAND_WIDTH_PIX, SYMMETRY_WINDOW_PIX, MIN_SYMMETRY_DIST_PIX, N_TOP_SYMMETRY,
                    ADJACENT_BAND_TOL_PIX, MIN_CONSECUTIVE_BANDS)
                all_lines_lonlat.extend(lines)

            # Plot results
            plot_results(stem, ref_base, tb_base, all_lines_lonlat, base_lon, base_lat)

        finally:
            dataset.close()

    print(f"Done processing {stem}")


def main():
    process_nc_files()


if __name__ == "__main__":
    main()