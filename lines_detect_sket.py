"""
Detect anomaly-band linear structures in MODIS gridded data.

This version does NOT use Canny edge detection.

For each processed file, this script:
1. Loads / re-generates gridded data from MYD021 NC files
2. Computes local background
3. Detects regions whose values are higher or lower than the local background
4. Cleans the anomaly mask
5. Plots the cleaned anomaly mask before skeletonize
6. Skeletonizes the anomaly bands into centerlines
7. Applies Hough Transform to the skeleton
8. Saves only one combined figure per file:
   original data, mask before skeletonize, skeleton before Hough, and final Hough lines

Usage:
    python lines_detect_anomaly_band.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import uniform_filter
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_objects,
    skeletonize,
)
from skimage.transform import probabilistic_hough_line


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CACHE_DIR = "/home/chenyiqi/260306_shiptrack_detect/cache"
LINES_OUT_DIR = "/home/chenyiqi/260306_shiptrack_detect/lines_figs"

RUN_FILE_START = 7
RUN_FILE_END = 8
# ---------------------------------------------------------------------------
# Anomaly-band detection parameters
# ---------------------------------------------------------------------------

# Local background scale.
# Larger value gives a smoother background and is better for broader cloud bands.
BACKGROUND_RADIUS_PIX = 250

# Anomaly threshold:
# threshold = DETECTION_SENSITIVITY * percentile(abs(anomaly), ANOMALY_PERCENTILE)
# Smaller value detects weaker anomaly bands but may increase false positives.
ANOMALY_PERCENTILE = 90
DETECTION_SENSITIVITY = 0.5

# Morphological cleanup.
# Closing connects broken bands.
# Opening removes small noisy patches.
MORPH_CLOSE_RADIUS = 0
MORPH_OPEN_RADIUS = 0
MIN_OBJECT_SIZE = 0

# Hough Transform on skeletonized anomaly bands.
HOUGH_LINE_LENGTH = 40
HOUGH_LINE_GAP = 25
HOUGH_THRESHOLD = 10

# Contrast filter.
# Smaller value keeps weaker anomaly lines.
CONTRAST_FRACTION = 0.06

# Plot settings.
LINE_COLOR = "black"
PLOT_LINEWIDTH = 3.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def local_background(data, radius=15):
    """
    NaN-aware local mean background.

    The local background is computed within a window of:
        (2 * radius + 1) x (2 * radius + 1)
    """
    finite = np.isfinite(data)
    filled = np.where(finite, data, 0.0)

    size = 2 * radius + 1

    local_mean_with_nan_as_zero = uniform_filter(
        filled,
        size=size,
        mode="nearest",
    )

    valid_fraction = uniform_filter(
        finite.astype(float),
        size=size,
        mode="nearest",
    )

    bg = np.full_like(data, np.nan, dtype=float)

    good = valid_fraction > 0.5
    bg[good] = local_mean_with_nan_as_zero[good] / valid_fraction[good]

    return bg


def filter_lines_by_contrast(lines, data, bg, dynamic_range):
    """
    Keep only Hough lines where pixels along the line are consistently
    brighter or consistently darker than the local background.

    This removes simple boundaries or mixed-sign structures.
    """
    if lines is None or len(lines) == 0:
        return [], []

    filtered_lines = []
    line_scores = []

    if not np.isfinite(dynamic_range) or dynamic_range <= 0:
        return [], []

    min_contrast = CONTRAST_FRACTION * dynamic_range
    h, w = data.shape

    for line in lines:
        (x1, y1), (x2, y2) = line

        length = int(np.hypot(x2 - x1, y2 - y1))
        if length < 3:
            continue

        n_sample = max(length, 10)

        xs = np.linspace(x1, x2, n_sample).astype(int)
        ys = np.linspace(y1, y2, n_sample).astype(int)

        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs = xs[valid]
        ys = ys[valid]

        if len(xs) < 3:
            continue

        line_vals = data[ys, xs]
        bg_vals = bg[ys, xs]

        diffs = line_vals - bg_vals
        diffs = diffs[np.isfinite(diffs)]

        if len(diffs) < 3:
            continue

        n_pos = np.sum(diffs > 0)
        n_neg = np.sum(diffs < 0)
        total = n_pos + n_neg

        if total < 3:
            continue

        # Require most pixels on the line to have the same anomaly sign.
        sign_consistency = max(n_pos, n_neg) / total
        if sign_consistency < 0.8:
            continue

        mean_abs_diff = np.mean(np.abs(diffs))
        if mean_abs_diff < min_contrast:
            continue

        filtered_lines.append(line)
        line_scores.append(mean_abs_diff / dynamic_range)

    return filtered_lines, line_scores


def detect_lines(data):
    """
    Detect line-like regions whose values are higher or lower than
    the surrounding local background.

    This does NOT detect edges. It detects anomaly bands:

        anomaly = data - local_background

    Then:
        anomaly band mask -> morphology cleanup -> skeleton -> Hough Transform

    Returns:
        filtered_lines
        clean_mask_before_skeleton
        skeleton
    """
    finite = np.isfinite(data)

    if not np.any(finite):
        empty = np.zeros_like(data, dtype=bool)
        return [], empty, empty

    # 1. Local background
    bg = local_background(data, radius=BACKGROUND_RADIUS_PIX)

    # 2. Local anomaly
    anomaly = data - bg
    valid = np.isfinite(anomaly)

    if not np.any(valid):
        empty = np.zeros_like(data, dtype=bool)
        return [], empty, empty

    # 3. Robust anomaly threshold
    anomaly_scale = np.nanpercentile(
        np.abs(anomaly[valid]),
        ANOMALY_PERCENTILE,
    )

    if not np.isfinite(anomaly_scale) or anomaly_scale <= 0:
        empty = np.zeros_like(data, dtype=bool)
        return [], empty, empty

    threshold = DETECTION_SENSITIVITY * anomaly_scale

    # 4. Detect both positive and negative anomaly bands
    anomaly_mask = np.abs(anomaly) > threshold
    anomaly_mask[~valid] = False

    # 5. Morphological cleanup
    clean_mask = anomaly_mask.astype(bool)

    if MORPH_CLOSE_RADIUS > 0:
        clean_mask = binary_closing(
            clean_mask,
            disk(MORPH_CLOSE_RADIUS),
        )

    if MORPH_OPEN_RADIUS > 0:
        clean_mask = binary_opening(
            clean_mask,
            disk(MORPH_OPEN_RADIUS),
        )

    clean_mask = remove_small_objects(
        clean_mask.astype(bool),
        min_size=MIN_OBJECT_SIZE,
    )

    # This is the image before skeletonize
    mask_before_skeleton = clean_mask.copy()

    if not np.any(mask_before_skeleton):
        empty = np.zeros_like(data, dtype=bool)
        return [], mask_before_skeleton, empty

    # 6. Convert broad anomaly bands to 1-pixel-wide centerlines
    skeleton = skeletonize(mask_before_skeleton)

    if not np.any(skeleton):
        return [], mask_before_skeleton, skeleton

    # 7. Hough Transform on the skeleton, not on edges
    lines = probabilistic_hough_line(
        skeleton,
        threshold=HOUGH_THRESHOLD,
        line_length=HOUGH_LINE_LENGTH,
        line_gap=HOUGH_LINE_GAP,
    )

    if lines is None or len(lines) == 0:
        return [], mask_before_skeleton, skeleton

    # 8. Keep only lines consistently brighter or darker than background
    dynamic_range = (
        np.nanpercentile(data[finite], 98)
        - np.nanpercentile(data[finite], 2)
    )

    filtered_lines, _ = filter_lines_by_contrast(
        lines,
        data,
        bg,
        dynamic_range,
    )

    return filtered_lines, mask_before_skeleton, skeleton


def get_color_limits(data):
    finite = np.isfinite(data)

    if not np.any(finite):
        return 0, 1

    vmin = np.nanpercentile(data, 2)
    vmax = np.nanpercentile(data, 98)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 0, 1

    return vmin, vmax


def plot_combined_figure(
    data_ref,
    data_tb,
    mask_ref,
    mask_tb,
    skeleton_ref,
    skeleton_tb,
    lines_ref,
    lines_tb,
    stem,
):
    """
    Save a 2 x 4 combined figure.

    Row 1: Reflectance
        col 1: original reflectance
        col 2: clean anomaly mask before skeletonize
        col 3: anomaly skeleton before Hough
        col 4: reflectance + Hough lines

    Row 2: BT difference
        col 1: original BT difference
        col 2: clean anomaly mask before skeletonize
        col 3: anomaly skeleton before Hough
        col 4: BT difference + Hough lines
    """
    fig, axes = plt.subplots(2, 4, figsize=(36, 18), dpi=200)

    vr_min, vr_max = get_color_limits(data_ref)
    vt_min, vt_max = get_color_limits(data_tb)

    # -----------------------------------------------------------------------
    # Row 1: Reflectance
    # -----------------------------------------------------------------------

    ax = axes[0, 0]
    im = ax.imshow(
        data_ref,
        cmap="jet",
        vmin=vr_min,
        vmax=vr_max,
        origin="upper",
        interpolation="none",
        rasterized=True,
    )
    ax.set_title("2.1 um Reflectance")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    ax.imshow(
        mask_ref,
        cmap="gray",
        origin="upper",
        interpolation="none",
        rasterized=True,
    )
    ax.set_title("Reflectance anomaly mask before skeletonize")
    ax.axis("off")

    ax = axes[0, 2]
    ax.imshow(
        skeleton_ref,
        cmap="gray",
        origin="upper",
        interpolation="none",
        rasterized=True,
    )
    ax.set_title("Reflectance skeleton before Hough")
    ax.axis("off")

    ax = axes[0, 3]
    im = ax.imshow(
        data_ref,
        cmap="jet",
        vmin=vr_min,
        vmax=vr_max,
        origin="upper",
        interpolation="none",
        rasterized=True,
    )
    ax.set_title(f"Reflectance + Hough lines ({len(lines_ref)})")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for line in lines_ref:
        (x1, y1), (x2, y2) = line
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=LINE_COLOR,
            linewidth=PLOT_LINEWIDTH,
            alpha=0.9,
        )

    # -----------------------------------------------------------------------
    # Row 2: BT difference
    # -----------------------------------------------------------------------

    ax = axes[1, 0]
    im = ax.imshow(
        data_tb,
        cmap="RdBu_r",
        vmin=vt_min,
        vmax=vt_max,
        origin="upper",
        interpolation="none",
        rasterized=True,
    )
    ax.set_title("BT Difference: 11 um - 3.7 um")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    ax.imshow(
        mask_tb,
        cmap="gray",
        origin="upper",
        interpolation="none",
        rasterized=True,
    )
    ax.set_title("BT-difference anomaly mask before skeletonize")
    ax.axis("off")

    ax = axes[1, 2]
    ax.imshow(
        skeleton_tb,
        cmap="gray",
        origin="upper",
        interpolation="none",
        rasterized=True,
    )
    ax.set_title("BT-difference skeleton before Hough")
    ax.axis("off")

    ax = axes[1, 3]
    im = ax.imshow(
        data_tb,
        cmap="RdBu_r",
        vmin=vt_min,
        vmax=vt_max,
        origin="upper",
        interpolation="none",
        rasterized=True,
    )
    ax.set_title(f"BT Difference + Hough lines ({len(lines_tb)})")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for line in lines_tb:
        (x1, y1), (x2, y2) = line
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=LINE_COLOR,
            linewidth=PLOT_LINEWIDTH,
            alpha=0.9,
        )

    fig.suptitle(f"File: {stem}", fontsize=16, y=0.995)
    fig.tight_layout(pad=1.2, rect=[0, 0, 1, 0.98])

    output_path = os.path.join(LINES_OUT_DIR, f"{stem}_lines_combined.png")
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"  Saved: {output_path}")


def process_file(file_stem, data_ref, data_tb):
    """
    Detect anomaly-band lines in both reflectance and BT difference data.

    Only one combined figure is saved.
    """
    print(f"\nProcessing anomaly-band line detection: {file_stem}")

    lines_ref, mask_ref, skeleton_ref = detect_lines(data_ref)
    print(f"  Reflectance: {len(lines_ref)} lines detected")

    lines_tb, mask_tb, skeleton_tb = detect_lines(data_tb)
    print(f"  BT Diff:     {len(lines_tb)} lines detected")

    plot_combined_figure(
        data_ref,
        data_tb,
        mask_ref,
        mask_tb,
        skeleton_ref,
        skeleton_tb,
        lines_ref,
        lines_tb,
        file_stem,
    )

    return lines_ref, lines_tb


def process_nc_files_immediately():
    """
    Re-process NC files using functions from plt_rotated_myd021.py.

    Each valid file is processed immediately after loading and interpolation.
    """
    sys.path.insert(0, "/home/chenyiqi/260306_shiptrack_detect")
    import plt_rotated_myd021 as rot

    grid_lon, grid_lat, n_rows, n_cols = rot.build_rotated_square_grid(
        rot.CENTER_LON,
        rot.CENTER_LAT,
        rot.SIDE_LENGTH_KM,
        rot.RESOLUTION_M,
        rot.ROTATION_ANGLE_DEG,
    )

    square_corners_lon, square_corners_lat = rot.get_square_corners_from_grid(
        grid_lon,
        grid_lat,
    )

    file_list = rot.load_myd021_file_list(rot.INPUT_DIR)
    files_to_process = file_list[RUN_FILE_START:RUN_FILE_END]

    total_files_checked = 0
    total_files_processed = 0
    total_files_skipped = 0
    total_ref_lines = 0
    total_tb_lines = 0

    print(f"Total files from rot.INPUT_DIR: {len(file_list)}")
    print(f"Files selected after FILE_START/FILE_END: {len(files_to_process)}")
    print(f"Target grid dimensions: {n_rows} x {n_cols}")
    print("Loading NC files and processing each valid file immediately...")

    if len(files_to_process) == 0:
        raise RuntimeError(
            "No files selected. Check rot.INPUT_DIR, rot.FILE_START, and rot.FILE_END."
        )

    for hdf_file in files_to_process:
        total_files_checked += 1
        stem = os.path.splitext(os.path.basename(hdf_file))[0]

        print(f"\nLoading: {stem}")

        try:
            dataset = Dataset(hdf_file, "r")
        except OSError:
            total_files_skipped += 1
            print(f"  Cannot open file, skipping: {hdf_file}")
            continue

        try:
            lat = rot.read_nc_field(dataset, rot.LAT_PATH)
            lon = rot.read_nc_field(dataset, rot.LON_PATH)
            lon = rot.normalize_longitude_if_dateline_crossed(lon)

            if not rot.quick_check_square_in_swath(
                square_corners_lon,
                square_corners_lat,
                lon,
                lat,
            ):
                total_files_skipped += 1
                print("  Target square outside swath, skipping")
                continue

            # Read 2.1 um reflectance
            ref_var = dataset[rot.REFSB_500_PATH]
            ref_index = rot.get_band_index(ref_var, 7)

            ref_21 = rot.read_and_scale_band(
                ref_var,
                ref_index,
                "reflectance_scales",
                "reflectance_offsets",
            )

            # Read emissive bands and calculate BT11 - BT3.7
            emissive_var = dataset[rot.EMISSIVE_PATH]

            bt11_index = rot.get_band_index(emissive_var, 31)
            bt37_index = rot.get_band_index(emissive_var, 20)

            rad_11 = rot.read_and_scale_band(
                emissive_var,
                bt11_index,
                "radiance_scales",
                "radiance_offsets",
            )

            rad_37 = rot.read_and_scale_band(
                emissive_var,
                bt37_index,
                "radiance_scales",
                "radiance_offsets",
            )

            tb_11 = rot.radiance2tb(rad_11, 11.0)
            tb_37 = rot.radiance2tb(rad_37, 3.7)
            tb_diff = tb_11 - tb_37

            # Resize geolocation to match data fields
            lat_ref = rot.resize_2d(lat, ref_21.shape)
            lon_ref = rot.resize_2d(lon, ref_21.shape)

            if tb_diff.shape == ref_21.shape:
                lat_tb = lat_ref
                lon_tb = lon_ref
            else:
                lat_tb = rot.resize_2d(lat, tb_diff.shape)
                lon_tb = rot.resize_2d(lon, tb_diff.shape)

            # Interpolate to rotated grid
            print("  Interpolating reflectance...")
            ref_grid, ref_valid = rot.resample_to_grid(
                ref_21,
                lon_ref,
                lat_ref,
                grid_lon,
                grid_lat,
                margin_deg=getattr(rot, "CROP_MARGIN_DEG", 0.30),
            )

            print("  Interpolating BT difference...")
            tb_grid, tb_valid = rot.resample_to_grid(
                tb_diff,
                lon_tb,
                lat_tb,
                grid_lon,
                grid_lat,
                margin_deg=getattr(rot, "CROP_MARGIN_DEG", 0.30),
            )

            if not np.all(ref_valid) or not np.all(tb_valid):
                total_files_skipped += 1
                print("  Gaps in data, skipping")
                continue

            # Optional cache save
            np.save(os.path.join(CACHE_DIR, f"{stem}_ref_2.1um.npy"), ref_grid)
            np.save(os.path.join(CACHE_DIR, f"{stem}_tb11_minus_tb3.7.npy"), tb_grid)

            # Detect and save immediately
            lines_ref, lines_tb = process_file(stem, ref_grid, tb_grid)

            total_files_processed += 1
            total_ref_lines += len(lines_ref)
            total_tb_lines += len(lines_tb)

        finally:
            dataset.close()

    return {
        "checked": total_files_checked,
        "processed": total_files_processed,
        "skipped": total_files_skipped,
        "ref_lines": total_ref_lines,
        "tb_lines": total_tb_lines,
    }


def main():
    os.makedirs(LINES_OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    stats = process_nc_files_immediately()

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Files checked:           {stats['checked']}")
    print(f"  Files processed:         {stats['processed']}")
    print(f"  Files skipped:           {stats['skipped']}")
    print(f"  Total reflectance lines: {stats['ref_lines']}")
    print(f"  Total BT diff lines:     {stats['tb_lines']}")
    print(f"  Output directory:        {LINES_OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()