"""
Line detection for MODIS gridded data using fixed-direction projection.

Workflow:
1. Read MYD021 files and interpolate to a base non-rotated 1 km grid.
2. Check whether the target square region is inside the satellite swath.
3. Read Ref 2.1 um and BT diff 11 um - 3.7 um.
4. Ref and BT share the same geolocation mapping, so lat/lon resizing is done once.
5. Remove 2D large-scale background from Ref and BT using median filtering.
6. Apply morphological enhancement: dilation for Ref and erosion for BT.
7. Remove large-scale background again after morphological enhancement.
8. Fuse by Ref_enhanced_detrended - BT_enhanced_detrended.
9. Binarize the high-value part of the fused image.
10. Sum the binary image along lines oriented north 30 degrees west.
11. Find local peaks whose along-line sum exceeds a threshold.
12. Plot Ref and BT interpolated to the 1 km grid, and Figure a: binary image with the corresponding north-30-west lines.
"""

import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from netCDF4 import Dataset
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


# ============================================================
# Paths
# ============================================================

INPUT_DIR = "/home/chenyiqi/260306_shiptrack_detect/MYD021_SE_Pacific"
OUT_DIR = "/home/chenyiqi/260306_shiptrack_detect/projection_figs"

SAVE_FIGURES = True


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
RUN_FILE_END = 48


# ============================================================
# Processing settings
# ============================================================

# Median filter kernel size for background removal. Must be odd.
TREND_FILTER_SIZE = 101

# Morphological enhancement kernel size.
MORPH_KERNEL_SIZE = 20

# Binary threshold. 85 means the highest 15% of fused-image pixels are selected.
BINARY_PERCENTILE = 90.0

# Projection-line direction: north 30 degrees west.
# In the local grid, x is eastward and y is northward.
PROJECTION_WEST_OF_NORTH_DEG = 30.0

# A projection-bin peak is accepted only when the binary sum along that direction
# is greater than this threshold. Since the input is binary, this is the number
# of selected pixels along each N30W-parallel line. With a 1 km grid, the value
# is approximately the total selected length in km along that line.
PROJECTION_SUM_THRESHOLD = 150.0

# Minimum separation between nearby detected peaks in projection bins.
PEAK_MIN_DISTANCE_PX = 20

# Optional prominence requirement for peak detection. Use 0.0 to disable.
PEAK_PROMINENCE = 0.02

# Limit the number of lines drawn. Set to None to draw all accepted peaks.
MAX_PEAK_LINES = None


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


def build_base_grid(center_lon, center_lat, side_km, resolution_m):
    """Build a non-rotated square grid centered on the target region."""
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


def read_band_raw(dataset, band_name, var_path, scales_attr, offsets_attr):
    """Read one scaled MODIS band without interpolation."""
    var = dataset[var_path]
    idx = get_band_index(var, band_name)
    return read_and_scale_band(var, idx, scales_attr, offsets_attr)


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
        interp = LinearNDInterpolator(
            np.column_stack((src_lon[near], src_lat[near])),
            data[near],
            fill_value=np.nan
        )
        gridded = interp(np.column_stack((tgt_lon.ravel(), tgt_lat.ravel()))).reshape(tgt_lon.shape)
    except Exception as exc:
        print(f"Interpolation failed: {exc}")
        return np.full(tgt_lon.shape, np.nan), np.full(tgt_lon.shape, False)

    return gridded, np.isfinite(gridded)


# ============================================================
# Image processing
# ============================================================

def remove_trend_median(data, filter_size):
    """Remove 2D large-scale background using median filtering."""
    global_median = np.nanmedian(data)
    data_filled = np.where(np.isfinite(data), data, global_median)
    background = median_filter(data_filled, size=filter_size)
    residual = data - background
    return residual, background


def enhance_morphological(data, erode_small=True, kernel_size=3):
    """
    Morphologically enhance residual images.

    For Ref, use dilation to expand bright features.
    For BT diff, use erosion to expand dark features.
    """
    from scipy.ndimage import grey_erosion, grey_dilation

    valid = np.isfinite(data)
    if not np.any(valid):
        return np.zeros_like(data)

    vmin, vmax = np.nanpercentile(data, 2), np.nanpercentile(data, 98)
    if vmax <= vmin:
        return np.zeros_like(data)

    normed = (data - vmin) / (vmax - vmin)
    normed = np.clip(normed, 0.0, 1.0)
    filled = np.where(np.isfinite(normed), normed, 0.0)

    if erode_small:
        return grey_dilation(filled, size=kernel_size)
    return grey_erosion(filled, size=kernel_size)


def make_binary_from_ref_bt(ref_residual, tb_residual):
    """
    Create the fused image and binary high-value image.

    Returns
    -------
    binary : ndarray of bool
        High-value binary image from the fused field.
    diff : ndarray
        Fused image = Ref morph+detrend - BT morph+detrend.
    """
    ref_enhanced = enhance_morphological(
        ref_residual, erode_small=True, kernel_size=MORPH_KERNEL_SIZE)
    tb_enhanced = enhance_morphological(
        tb_residual, erode_small=False, kernel_size=MORPH_KERNEL_SIZE)

    ref_morph_detrend, _ = remove_trend_median(ref_enhanced, TREND_FILTER_SIZE)
    tb_morph_detrend, _ = remove_trend_median(tb_enhanced, TREND_FILTER_SIZE)

    diff = ref_morph_detrend - tb_morph_detrend
    threshold = np.nanpercentile(diff, BINARY_PERCENTILE)
    binary = np.isfinite(diff) & (diff >= threshold)

    return binary, diff


# ============================================================
# Projection along N30W direction
# ============================================================

def get_projection_vectors(west_of_north_deg=30.0):
    """
    Return unit vectors for a line pointing north-west and its normal.

    x points eastward, y points northward.
    North 30 degrees west has direction u = (-sin30, cos30).
    The normal vector n is perpendicular to u, so lines parallel to u satisfy
    dot((x, y), n) = constant.
    """
    a = np.deg2rad(west_of_north_deg)
    u = np.array([-np.sin(a), np.cos(a)], dtype=float)
    u = u / np.linalg.norm(u)
    n = np.array([u[1], -u[0]], dtype=float)
    n = n / np.linalg.norm(n)
    return u, n


def projection_sum_along_direction(binary, west_of_north_deg=30.0):
    """
    Sum a binary image along lines parallel to N30W.

    The returned profile is indexed by the perpendicular coordinate q. Each
    profile value is the binary sum along one line family member.
    """
    n_rows, n_cols = binary.shape
    yy, xx = np.indices(binary.shape, dtype=float)

    # Centered pixel coordinates. Since RESOLUTION_M = 1000, one pixel is ~1 km.
    x = xx - 0.5 * (n_cols - 1)
    y = yy - 0.5 * (n_rows - 1)

    _, normal = get_projection_vectors(west_of_north_deg)
    q = x * normal[0] + y * normal[1]

    q_min = np.floor(np.nanmin(q))
    q_max = np.ceil(np.nanmax(q))
    bin_edges = np.arange(q_min - 0.5, q_max + 1.5, 1.0)
    q_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_id = np.digitize(q.ravel(), bin_edges) - 1
    valid = (bin_id >= 0) & (bin_id < len(q_centers))

    values = binary.astype(float).ravel()
    sums = np.bincount(bin_id[valid], weights=values[valid], minlength=len(q_centers))
    counts = np.bincount(bin_id[valid], minlength=len(q_centers))

    profile = np.full(len(q_centers), np.nan, dtype=float)
    ok = counts > 0
    profile[ok] = sums[ok]

    return q_centers, profile


def find_projection_peaks(q_centers, profile):
    """Find local projection peaks exceeding the specified sum threshold."""
    valid_profile = np.where(np.isfinite(profile), profile, -np.inf)

    peaks, props = find_peaks(
        valid_profile,
        height=PROJECTION_SUM_THRESHOLD,
        distance=PEAK_MIN_DISTANCE_PX,
        prominence=PEAK_PROMINENCE if PEAK_PROMINENCE > 0 else None
    )

    if peaks.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    # Sort by peak height from high to low.
    heights = props["peak_heights"]
    order = np.argsort(heights)[::-1]
    peaks = peaks[order]
    heights = heights[order]

    if MAX_PEAK_LINES is not None:
        peaks = peaks[:MAX_PEAK_LINES]
        heights = heights[:MAX_PEAK_LINES]

    # Sort back by q position for cleaner plotting.
    order_q = np.argsort(q_centers[peaks])
    return peaks[order_q], heights[order_q]


def line_segment_for_q(q_value, image_shape, west_of_north_deg=30.0):
    """
    Convert a projection peak q into a clipped pixel-space line segment.

    Pixel coordinates are centered: x = col - center_col, y = row - center_row.
    The line is q = dot((x, y), normal) and is clipped to the image rectangle.
    """
    n_rows, n_cols = image_shape
    x_min = -0.5 * (n_cols - 1)
    x_max = 0.5 * (n_cols - 1)
    y_min = -0.5 * (n_rows - 1)
    y_max = 0.5 * (n_rows - 1)

    direction, normal = get_projection_vectors(west_of_north_deg)

    # A point on the line.
    p0 = q_value * normal

    candidates = []
    eps = 1e-12

    # Intersections with x = x_min and x = x_max.
    if abs(direction[0]) > eps:
        for xb in (x_min, x_max):
            t = (xb - p0[0]) / direction[0]
            yb = p0[1] + t * direction[1]
            if y_min - 1e-9 <= yb <= y_max + 1e-9:
                candidates.append((xb, yb))

    # Intersections with y = y_min and y = y_max.
    if abs(direction[1]) > eps:
        for yb in (y_min, y_max):
            t = (yb - p0[1]) / direction[1]
            xb = p0[0] + t * direction[0]
            if x_min - 1e-9 <= xb <= x_max + 1e-9:
                candidates.append((xb, yb))

    if len(candidates) < 2:
        return None

    # Remove near-duplicate corner intersections.
    unique = []
    for pt in candidates:
        if not any(np.hypot(pt[0] - qpt[0], pt[1] - qpt[1]) < 1e-6 for qpt in unique):
            unique.append(pt)

    if len(unique) < 2:
        return None

    # Choose the two farthest points.
    max_dist = -1.0
    p_start, p_end = unique[0], unique[1]
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            dist = np.hypot(unique[i][0] - unique[j][0], unique[i][1] - unique[j][1])
            if dist > max_dist:
                max_dist = dist
                p_start, p_end = unique[i], unique[j]

    return p_start, p_end


def centered_pixel_to_lonlat(x_centered, y_centered, grid_lon, grid_lat):
    """Map centered pixel coordinates to lon/lat on the regular base grid."""
    n_rows, n_cols = grid_lon.shape
    col = x_centered + 0.5 * (n_cols - 1)
    row = y_centered + 0.5 * (n_rows - 1)

    lon = grid_lon[0, 0] + (col / (n_cols - 1)) * (grid_lon[0, -1] - grid_lon[0, 0])
    lat = grid_lat[0, 0] + (row / (n_rows - 1)) * (grid_lat[-1, 0] - grid_lat[0, 0])
    return float(lon), float(lat)


def projection_peak_lines_lonlat(q_centers, peak_indices, image_shape, grid_lon, grid_lat):
    """Return lon/lat line segments corresponding to accepted projection peaks."""
    lines_lonlat = []
    for peak_idx in peak_indices:
        q_value = float(q_centers[peak_idx])
        segment = line_segment_for_q(
            q_value,
            image_shape,
            west_of_north_deg=PROJECTION_WEST_OF_NORTH_DEG
        )
        if segment is None:
            continue

        (x1, y1), (x2, y2) = segment
        lon1, lat1 = centered_pixel_to_lonlat(x1, y1, grid_lon, grid_lat)
        lon2, lat2 = centered_pixel_to_lonlat(x2, y2, grid_lon, grid_lat)
        lines_lonlat.append((lon1, lat1, lon2, lat2, q_value))

    return lines_lonlat


# ============================================================
# Plotting
# ============================================================

def get_grid_extent(grid_lon, grid_lat):
    """Return plotting extent of the base grid."""
    return [grid_lon[0, 0], grid_lon[0, -1], grid_lat[0, 0], grid_lat[-1, 0]]


def get_color_limits(data):
    """Return robust color limits for a gridded image."""
    finite = np.isfinite(data)
    if not np.any(finite):
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(data, 2), np.nanpercentile(data, 98)
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= vmin:
        return 0.0, 1.0
    return float(vmin), float(vmax)


def plot_ref_bt_binary_with_projection_lines(stem, ref_grid, tb_grid, binary,
                                             peak_lines_lonlat, grid_lon, grid_lat):
    """Plot Ref and BT interpolated to 1 km, plus binary image with N30W lines."""
    extent = get_grid_extent(grid_lon, grid_lat)
    ref_vmin, ref_vmax = get_color_limits(ref_grid)
    tb_vmin, tb_vmax = get_color_limits(tb_grid)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)

    ax = axes[0]
    im = ax.imshow(
        ref_grid,
        cmap="gray",
        vmin=ref_vmin,
        vmax=ref_vmax,
        extent=extent,
        origin="lower",
        interpolation="none"
    )
    ax.set_title("Ref 2.1 µm interpolated to 1 km", fontsize=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im = ax.imshow(
        tb_grid,
        cmap="gray",
        vmin=tb_vmin,
        vmax=tb_vmax,
        extent=extent,
        origin="lower",
        interpolation="none"
    )
    ax.set_title("BT diff 11 µm - 3.7 µm interpolated to 1 km", fontsize=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    ax.imshow(
        binary.astype(float),
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=extent,
        origin="lower",
        interpolation="none"
    )

    for line_id, (lon1, lat1, lon2, lat2, q_value) in enumerate(peak_lines_lonlat, start=1):
        ax.plot([lon1, lon2], [lat1, lat2], "-", color="red", linewidth=1.5)
        ax.text(
            0.5 * (lon1 + lon2),
            0.5 * (lat1 + lat2),
            str(line_id),
            color="red",
            fontsize=8,
            ha="center",
            va="center"
        )

    ax.set_title(
        f"Figure a: binary + N{PROJECTION_WEST_OF_NORTH_DEG:g}W sum peaks "
        f"({len(peak_lines_lonlat)} lines)",
        fontsize=10
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.suptitle(stem, fontsize=12, y=0.98)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, f"{stem}_ref_bt_binary_N30W_sum_lines.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"Saved Ref/BT/Figure a for {stem}: {out_path}")


def plot_projection_profile(stem, q_centers, profile, peak_indices):
    """Optional diagnostic plot of the 1D projection-sum profile."""
    fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
    ax.plot(q_centers, profile, "-", linewidth=1.0)
    ax.axhline(PROJECTION_SUM_THRESHOLD, linestyle="--", linewidth=1.0)
    if peak_indices.size > 0:
        ax.plot(q_centers[peak_indices], profile[peak_indices], "o", markersize=4)
    ax.set_xlabel("Perpendicular coordinate q (pixels, ~km)")
    ax.set_ylabel("Binary sum along N30W")
    ax.set_title("Projection-sum profile")
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, f"{stem}_N30W_projection_sum_profile.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"Saved projection-sum profile for {stem}: {out_path}")


# ============================================================
# Main processing
# ============================================================

def process_nc_files():
    if SAVE_FIGURES:
        os.makedirs(OUT_DIR, exist_ok=True)

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

            # Read Ref 2.1 um.
            ref_raw = read_band_raw(
                dataset,
                7,
                REFSB_500_PATH,
                "reflectance_scales",
                "reflectance_offsets"
            )

            # Read BT diff = BT(11 um) - BT(3.7 um).
            emissive_var = dataset[EMISSIVE_PATH]
            rad_11 = read_and_scale_band(
                emissive_var,
                get_band_index(emissive_var, 31),
                "radiance_scales",
                "radiance_offsets"
            )
            rad_37 = read_and_scale_band(
                emissive_var,
                get_band_index(emissive_var, 20),
                "radiance_scales",
                "radiance_offsets"
            )
            tb_diff = radiance2tb(rad_11, 11.0) - radiance2tb(rad_37, 3.7)

            if ref_raw.shape != tb_diff.shape:
                raise ValueError(
                    f"Ref and BT arrays do not have the same shape for {stem}: "
                    f"ref={ref_raw.shape}, tb={tb_diff.shape}. "
                    "The shared-geolocation shortcut requires equal shapes."
                )

            # Ref and BT share the same geolocation mapping.
            common_lat = resize_2d(lat, ref_raw.shape)
            common_lon = resize_2d(lon, ref_raw.shape)

            ref_base, ref_valid = resample_to_grid(
                ref_raw,
                common_lon,
                common_lat,
                base_lon,
                base_lat,
                margin_deg=CROP_MARGIN_DEG
            )
            tb_base, tb_valid = resample_to_grid(
                tb_diff,
                common_lon,
                common_lat,
                base_lon,
                base_lat,
                margin_deg=CROP_MARGIN_DEG
            )

            if not np.all(ref_valid) or not np.all(tb_valid):
                print(f"Skipped (invalid data): {stem}")
                continue

            # Remove large-scale background.
            ref_residual, _ = remove_trend_median(ref_base, TREND_FILTER_SIZE)
            tb_residual, _ = remove_trend_median(tb_base, TREND_FILTER_SIZE)

            # Create binary image from fused Ref-BT enhanced signal.
            binary, diff = make_binary_from_ref_bt(ref_residual, tb_residual)

            # Sum binary image along N30W lines and find peak positions.
            q_centers, profile = projection_sum_along_direction(
                binary,
                west_of_north_deg=PROJECTION_WEST_OF_NORTH_DEG
            )
            peak_indices, peak_heights = find_projection_peaks(q_centers, profile)
            peak_lines_lonlat = projection_peak_lines_lonlat(
                q_centers,
                peak_indices,
                binary.shape,
                base_lon,
                base_lat
            )

            print(
                f"{stem}: binary percentile={BINARY_PERCENTILE:g}, "
                f"projection peaks={len(peak_lines_lonlat)}"
            )
            if len(peak_lines_lonlat) > 0:
                for i, (line, height) in enumerate(zip(peak_lines_lonlat, peak_heights), start=1):
                    lon1, lat1, lon2, lat2, q_value = line
                    print(
                        f"  peak {i}: q={q_value:.1f}, sum={height:.1f}, "
                        f"line=({lon1:.4f},{lat1:.4f}) -> ({lon2:.4f},{lat2:.4f})"
                    )

            # Figure a: binary image before any skeleton/Hough processing,
            # with the N30W projection-peak lines overlaid.
            if SAVE_FIGURES:
                plot_ref_bt_binary_with_projection_lines(
                    stem,
                    ref_base,
                    tb_base,
                    binary,
                    peak_lines_lonlat,
                    base_lon,
                    base_lat
                )
                plot_projection_profile(stem, q_centers, profile, peak_indices)

        finally:
            dataset.close()

    print(f"Done processing {len(files_to_process)} selected file(s)")


def main():
    process_nc_files()


if __name__ == "__main__":
    main()
