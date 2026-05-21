"""
Line detection for MODIS gridded data.

Workflow:
1. Read MYD021 files and interpolate to a base (non-rotated) square grid.
2. Remove 2D large-scale trend using median filtering.
3. Apply morphological enhancement (dilation for Ref, erosion for BT).
4. Second trend removal after morphological enhancement.
5. Fuse: ref_morph_detrend - tb_morph_detrend.
6. Binarize: top 10% of fused image.
7. Clean binary with dilation+erosion, then skeletonize.
8. Extract straight lines using probabilistic Hough transform.
9. Filter short lines and lines outside the base grid extent.
10. Plot: 4x4 subplots showing all intermediate steps.
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
# Detection settings
# ============================================================

# Median filter kernel size for background removal (must be odd)
TREND_FILTER_SIZE = 101

# Hough transform parameters (probabilistic Hough)
HOUGH_THRESHOLD = 10          # accumulator threshold for probabilistic Hough
HOUGH_MIN_LINE_LENGTH = 50    # minimum line length in pixels
HOUGH_MAX_LINE_GAP = 30       # maximum gap between segments to connect
HOUGH_MIN_LINE_LENGTH_KM = 50.0  # minimum line length in km

# Morphological enhancement kernel size
MORPH_KERNEL_SIZE = 20

# Binary cleanup: dilation then erosion iterations
BINARY_DILATE_ITER = 1
BINARY_ERODE_ITER = 1



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
    from scipy.interpolate import RegularGridInterpolator
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
# Line detection
# ============================================================

def detect_lines(ref_residual, tb_residual, grid_lon, grid_lat,
                 hough_threshold, hough_min_line_length,
                 hough_max_line_gap, min_line_length_km):
    """
    Apply morphological enhancement, second trend removal, fuse,
    binarize (top 10%), clean with binary morph, skeletonize,
    then extract straight lines using probabilistic Hough transform.

    Parameters
    ----------
    ref_residual, tb_residual : ndarray
        Residual images after trend removal.
    grid_lon, grid_lat : ndarray
        Lon/lat of the base grid.
    hough_threshold : int
        Accumulator threshold for probabilistic Hough.
    hough_min_line_length : int
        Minimum line length in pixels for probabilistic Hough.
    hough_max_line_gap : int
        Maximum gap between segments to connect.
    min_line_length_km : float
        Minimum line length in km.

    Returns
    -------
    lines_lonlat : list of (lon1, lat1, lon2, lat2)
        Detected lines in lon/lat coordinates.
    ref_norm : ndarray
        Normalized Reflectance residual (before morphological enhancement).
    tb_norm : ndarray
        Normalized BT diff residual (before morphological enhancement).
    ref_enhanced : ndarray
        Morphologically enhanced Reflectance residual.
    tb_enhanced : ndarray
        Morphologically enhanced BT diff residual.
    ref_morph_detrend : ndarray
        Reflectance after morph + second trend removal.
    tb_morph_detrend : ndarray
        BT diff after morph + second trend removal.
    diff_norm : ndarray
        Normalized diff (ref - bt) after morph+detrend.
    binary : ndarray
        Binary image (top 10% of diff).
    binary_clean : ndarray
        Binary after dilation+erosion.
    skeleton : ndarray
        Skeletonized binary (single-pixel-wide lines).
    """
    n_rows, n_cols = ref_residual.shape

    # Normalize residuals to [0, 1] for display (before morph)
    def normalize_to_01(arr):
        valid = np.isfinite(arr)
        if not np.any(valid):
            return np.zeros_like(arr)
        vmin, vmax = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
        if vmax <= vmin:
            return np.zeros_like(arr)
        normed = (arr - vmin) / (vmax - vmin)
        return np.clip(normed, 0, 1)

    ref_norm = normalize_to_01(ref_residual)
    tb_norm = normalize_to_01(tb_residual)

    # Apply morphological enhancement
    ref_enhanced = enhance_morphological(ref_residual, erode_small=True, kernel_size=MORPH_KERNEL_SIZE)
    tb_enhanced = enhance_morphological(tb_residual, erode_small=False, kernel_size=MORPH_KERNEL_SIZE)

    # Second trend removal after morphological enhancement
    ref_morph_detrend, _ = remove_trend_median(ref_enhanced, TREND_FILTER_SIZE)
    tb_morph_detrend, _ = remove_trend_median(tb_enhanced, TREND_FILTER_SIZE)

    # Normalize the post-morph detrended images for display
    ref_morph_detrend_norm = normalize_to_01(ref_morph_detrend)
    tb_morph_detrend_norm = normalize_to_01(tb_morph_detrend)

    # Fuse: ref_morph_detrend - tb_morph_detrend
    diff = ref_morph_detrend - tb_morph_detrend
    diff_norm = normalize_to_01(diff)

    # Take top 10% as binary
    threshold_top10 = np.nanpercentile(diff, 90)
    binary = np.where(np.isfinite(diff) & (diff >= threshold_top10), 1, 0)

    # Clean binary image: dilate then erode to connect nearby pixels
    from scipy.ndimage import binary_dilation, binary_erosion
    binary_clean = binary_dilation(binary, iterations=BINARY_DILATE_ITER)
    binary_clean = binary_erosion(binary_clean, iterations=BINARY_ERODE_ITER)

    # Skeletonize the cleaned binary image to get single-pixel-wide lines
    skeleton = skeletonize(binary_clean.astype(bool))

    # Probabilistic Hough transform on the skeleton
    # Returns list of line segments: [(x1, y1), (x2, y2)]
    lines_pixels = probabilistic_hough_line(
        skeleton,
        threshold=hough_threshold,
        line_length=hough_min_line_length,
        line_gap=hough_max_line_gap
    )

    # Filter by minimum length in km and map to lon/lat
    lines_lonlat = []
    for (x1, y1), (x2, y2) in lines_pixels:
        dx_km = (x2 - x1) * RESOLUTION_M / 1000.0
        dy_km = (y2 - y1) * RESOLUTION_M / 1000.0
        length_km = np.hypot(dx_km, dy_km)
        if length_km < min_line_length_km:
            continue

        def safe_lookup(y, x):
            yi = int(round(y))
            xi = int(round(x))
            if 0 <= yi < n_rows and 0 <= xi < n_cols:
                return float(grid_lon[yi, xi]), float(grid_lat[yi, xi])
            return None

        p1 = safe_lookup(y1, x1)
        p2 = safe_lookup(y2, x2)
        if p1 is not None and p2 is not None:
            lines_lonlat.append((p1[0], p1[1], p2[0], p2[1]))

    return lines_lonlat, ref_norm, tb_norm, ref_enhanced, tb_enhanced, ref_morph_detrend_norm, tb_morph_detrend_norm, diff_norm, binary, binary_clean, skeleton


# ============================================================
# Plotting
# ============================================================

def get_color_limits(data):
    finite = np.isfinite(data)
    if not np.any(finite):
        return 0, 1
    vmin, vmax = np.nanpercentile(data, 2), np.nanpercentile(data, 98)
    return (0, 1) if (not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin) else (vmin, vmax)


def plot_results(stem, ref_grid, tb_grid, all_lines_lonlat, grid_lon, grid_lat,
                 ref_norm=None, tb_norm=None,
                 ref_enhanced=None, tb_enhanced=None,
                 ref_morph_detrend=None, tb_morph_detrend=None,
                 diff_norm=None, binary=None,
                 binary_clean=None, skeleton=None):
    """
    Plot a single figure with 4x4 subplots:
      Row 0: Ref original | Ref residual (before morph) | Ref enhanced (after morph) | Ref morph + detrend
      Row 1: BT original  | BT residual (before morph)  | BT enhanced (after morph)  | BT morph + detrend
      Row 2: Diff (ref - bt) | Binary (top 10%) | Binary cleaned (dil+erode) | Skeleton
      Row 3: Ref + lines  | BT + lines | (info) | (empty)

    Lines whose endpoints fall outside the base grid extent are filtered out.
    """
    ref_vmin, ref_vmax = get_color_limits(ref_grid)
    tb_vmin, tb_vmax = get_color_limits(tb_grid)

    extent = [grid_lon[0, 0], grid_lon[0, -1], grid_lat[0, 0], grid_lat[-1, 0]]

    # Filter lines: keep only those with both endpoints inside the base grid extent
    lon_min, lon_max = extent[0], extent[1]
    lat_min, lat_max = extent[2], extent[3]
    filtered_lines = []
    for lon1, lat1, lon2, lat2 in all_lines_lonlat:
        if (lon_min <= lon1 <= lon_max and lat_min <= lat1 <= lat_max and
            lon_min <= lon2 <= lon_max and lat_min <= lat2 <= lat_max):
            filtered_lines.append((lon1, lat1, lon2, lat2))
    all_lines_lonlat = filtered_lines

    fig, axes = plt.subplots(4, 4, figsize=(24, 20), dpi=200)

    # ---- Row 0: Reflectance ----
    # (0,0) Ref original
    ax = axes[0, 0]
    im = ax.imshow(ref_grid, cmap="gray", vmin=ref_vmin, vmax=ref_vmax,
                   extent=extent, origin="lower", interpolation="none")
    ax.set_title("Reflectance 2.1um (original)", fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (0,1) Ref residual (before morph)
    ax = axes[0, 1]
    if ref_norm is not None:
        im = ax.imshow(ref_norm, cmap="gray", vmin=0, vmax=1,
                       extent=extent, origin="lower", interpolation="none")
        ax.set_title("Reflectance residual (before morph)", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # (0,2) Ref enhanced (after morph)
    ax = axes[0, 2]
    if ref_enhanced is not None:
        im = ax.imshow(ref_enhanced, cmap="gray", vmin=0, vmax=1,
                       extent=extent, origin="lower", interpolation="none")
        ax.set_title(f"Reflectance enhanced (kernel={MORPH_KERNEL_SIZE})", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # (0,3) Ref morph + detrend
    ax = axes[0, 3]
    if ref_morph_detrend is not None:
        im = ax.imshow(ref_morph_detrend, cmap="gray", vmin=0, vmax=1,
                       extent=extent, origin="lower", interpolation="none")
        ax.set_title("Reflectance morph + detrend", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # ---- Row 1: BT diff ----
    # (1,0) BT original
    ax = axes[1, 0]
    im = ax.imshow(tb_grid, cmap="gray", vmin=tb_vmin, vmax=tb_vmax,
                   extent=extent, origin="lower", interpolation="none")
    ax.set_title("BT Diff 11um - 3.7um (original)", fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (1,1) BT residual (before morph)
    ax = axes[1, 1]
    if tb_norm is not None:
        im = ax.imshow(tb_norm, cmap="gray", vmin=0, vmax=1,
                       extent=extent, origin="lower", interpolation="none")
        ax.set_title("BT Diff residual (before morph)", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # (1,2) BT enhanced (after morph)
    ax = axes[1, 2]
    if tb_enhanced is not None:
        im = ax.imshow(tb_enhanced, cmap="gray", vmin=0, vmax=1,
                       extent=extent, origin="lower", interpolation="none")
        ax.set_title(f"BT Diff enhanced (kernel={MORPH_KERNEL_SIZE})", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # (1,3) BT morph + detrend
    ax = axes[1, 3]
    if tb_morph_detrend is not None:
        im = ax.imshow(tb_morph_detrend, cmap="gray", vmin=0, vmax=1,
                       extent=extent, origin="lower", interpolation="none")
        ax.set_title("BT Diff morph + detrend", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # ---- Row 2: Diff and Binary ----
    # (2,0) Diff (ref - bt)
    ax = axes[2, 0]
    if diff_norm is not None:
        im = ax.imshow(diff_norm, cmap="gray", vmin=0, vmax=1,
                       extent=extent, origin="lower", interpolation="none")
        ax.set_title("Diff (ref - bt) after morph+detrend", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # (2,1) Binary (top 10%)
    ax = axes[2, 1]
    if binary is not None:
        ax.imshow(binary, cmap="gray", vmin=0, vmax=1,
                  extent=extent, origin="lower", interpolation="none")
        ax.set_title("Binary (top 10% of diff)", fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # (2,2) Binary cleaned (dil+erode)
    ax = axes[2, 2]
    if binary_clean is not None:
        ax.imshow(binary_clean, cmap="gray", vmin=0, vmax=1,
                  extent=extent, origin="lower", interpolation="none")
        ax.set_title("Binary cleaned (dilate+erode)", fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # (2,3) Skeleton
    ax = axes[2, 3]
    if skeleton is not None:
        ax.imshow(skeleton, cmap="gray", vmin=0, vmax=1,
                  extent=extent, origin="lower", interpolation="none")
        ax.set_title("Skeleton (single-pixel lines)", fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # ---- Row 3: Data + lines ----
    # (3,0) Reflectance + lines
    ax = axes[3, 0]
    ax.imshow(ref_grid, cmap="gray", vmin=ref_vmin, vmax=ref_vmax,
              extent=extent, origin="lower", interpolation="none")
    for lon1, lat1, lon2, lat2 in all_lines_lonlat:
        ax.plot([lon1, lon2], [lat1, lat2], '-', color='red', linewidth=1.5)
    ax.set_title(f"Reflectance + lines ({len(all_lines_lonlat)} lines)", fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # (3,1) BT diff + lines
    ax = axes[3, 1]
    ax.imshow(tb_grid, cmap="gray", vmin=tb_vmin, vmax=tb_vmax,
              extent=extent, origin="lower", interpolation="none")
    for lon1, lat1, lon2, lat2 in all_lines_lonlat:
        ax.plot([lon1, lon2], [lat1, lat2], '-', color='red', linewidth=1.5)
    ax.set_title(f"BT Diff + lines ({len(all_lines_lonlat)} lines)", fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # (3,2) Info
    ax = axes[3, 2]
    ax.axis("off")
    info_text = (
        f"File: {stem}\n"
        f"Filter size: {TREND_FILTER_SIZE}\n"
        f"Morph kernel: {MORPH_KERNEL_SIZE}\n"
        f"Hough threshold: {HOUGH_THRESHOLD}\n"
        f"Hough line length: {HOUGH_MIN_LINE_LENGTH}\n"
        f"Hough line gap: {HOUGH_MAX_LINE_GAP}\n"
        f"Min line length: {HOUGH_MIN_LINE_LENGTH_KM} km\n"
        f"Binary dilate iter: {BINARY_DILATE_ITER}\n"
        f"Binary erode iter: {BINARY_ERODE_ITER}\n"
        f"Detected lines: {len(all_lines_lonlat)}"
    )
    ax.text(0.5, 0.5, info_text, transform=ax.transAxes,
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # (3,3) empty
    ax = axes[3, 3]
    ax.axis("off")

    fig.suptitle(stem, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(LINES_OUT_DIR, f"{stem}_canny_combined.png"),
                bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Saved canny combined figure for {stem} ({len(all_lines_lonlat)} lines)")


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

            # Read and interpolate to base grid
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

            # Step 1: Remove 2D trend using median filtering
            ref_residual, ref_background = remove_trend_median(ref_base, TREND_FILTER_SIZE)
            tb_residual, tb_background = remove_trend_median(tb_base, TREND_FILTER_SIZE)

            # Step 2: Detect lines
            (lines_lonlat, ref_norm, tb_norm, ref_enhanced, tb_enhanced,
             ref_morph_detrend, tb_morph_detrend,
             diff_norm, binary, binary_clean, skeleton) = detect_lines(
                ref_residual, tb_residual, base_lon, base_lat,
                HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP,
                HOUGH_MIN_LINE_LENGTH_KM)

            # Step 3: Plot results
            plot_results(stem, ref_base, tb_base, lines_lonlat, base_lon, base_lat,
                         ref_norm=ref_norm, tb_norm=tb_norm,
                         ref_enhanced=ref_enhanced, tb_enhanced=tb_enhanced,
                         ref_morph_detrend=ref_morph_detrend, tb_morph_detrend=tb_morph_detrend,
                         diff_norm=diff_norm, binary=binary,
                         binary_clean=binary_clean, skeleton=skeleton)

        finally:
            dataset.close()

    print(f"Done processing {stem}")


def main():
    process_nc_files()


if __name__ == "__main__":
    main()
