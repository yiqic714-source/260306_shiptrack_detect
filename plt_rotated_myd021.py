import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from netCDF4 import Dataset
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator


INPUT_DIR = "/home/chenyiqi/260306_shiptrack_detect/MYD021_SE_Pacific"
OUT_DIR = "/home/chenyiqi/260306_shiptrack_detect/figs"

# Process range for testing.
# Use FILE_END = None to process all files after FILE_START.
FILE_START = 3
FILE_END = 101

# Square region parameters
CENTER_LON = -75.5
CENTER_LAT = -22.5
SIDE_LENGTH_KM = 750.0
RESOLUTION_M = 1000.0

# Orientation: north 10° west means the square is rotated 10° clockwise
# in the local x-y coordinate system used here.
ROTATION_ANGLE_DEG = 0.0

# Interpolation speed-up parameter.
# Only source pixels within the target-grid bounding box plus this margin are used.
CROP_MARGIN_DEG = 0.30

LAT_PATH = "HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Geolocation Fields/Latitude"
LON_PATH = "HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Geolocation Fields/Longitude"
REFSB_500_PATH = "HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_500_Aggr1km_RefSB"
EMISSIVE_PATH = "HDFEOS/SWATHS/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_Emissive"

h = 6.62607015e-34
c = 2.99792458e8
k = 1.380649e-23

EARTH_RADIUS_KM = 6371.0


def lon_km_to_deg(dx_km, lat):
    """Convert km in longitude direction to degrees at a given latitude."""
    return dx_km / (EARTH_RADIUS_KM * np.cos(np.deg2rad(lat)) * np.pi / 180.0)


def lat_km_to_deg(dy_km):
    """Convert km in latitude direction to degrees."""
    return dy_km / (EARTH_RADIUS_KM * np.pi / 180.0)


def read_nc_field(dataset, field_path):
    """Read a netCDF field and convert masked values to NaN."""
    arr = dataset[field_path][:]
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    return np.asarray(arr, dtype=float)


def parse_band_names(band_names_attr):
    if isinstance(band_names_attr, bytes):
        band_names_attr = band_names_attr.decode("utf-8")
    return [item.strip() for item in str(band_names_attr).split(",")]


def get_band_index(variable, band_name):
    band_names = parse_band_names(variable.getncattr("band_names"))
    return band_names.index(str(band_name))


def read_and_scale_band(variable, band_index, scales_attr, offsets_attr):
    """Read one MODIS band and apply scale/offset."""
    data = variable[band_index, :, :]
    data = np.asarray(data, dtype=float)
    if np.ma.isMaskedArray(data):
        data = data.filled(np.nan)


    fill_value = variable.getncattr("_FillValue") if "_FillValue" in variable.ncattrs() else None
    if fill_value is not None:
        data[data == fill_value] = np.nan

    scales = np.asarray(variable.getncattr(scales_attr), dtype=float)
    offsets = np.asarray(variable.getncattr(offsets_attr), dtype=float)

    data = (data - offsets[band_index]) * scales[band_index]
    return data


def radiance2tb(radiance, wavelength_um):
    """Convert spectral radiance to brightness temperature using Planck's law."""
    wavelength_m = wavelength_um * 1e-6

    # Convert W m-2 sr-1 um-1 to W m-2 sr-1 m-1
    b = radiance * 1e6

    tb = np.full_like(radiance, np.nan, dtype=float)
    mask = np.isfinite(b) & (b > 0)

    if np.any(mask):
        c1 = 2.0 * h * c**2
        c2 = h * c / k
        tb[mask] = c2 / (
            wavelength_m * np.log(1.0 + c1 / (wavelength_m**5 * b[mask]))
        )

    return tb


def resize_2d(array, target_shape):
    """Resample a 2D array to a target shape using RegularGridInterpolator."""
    if array.shape == target_shape:
        return array

    src_y = np.arange(array.shape[0], dtype=float)
    src_x = np.arange(array.shape[1], dtype=float)

    tgt_y = np.linspace(0, array.shape[0] - 1, target_shape[0])
    tgt_x = np.linspace(0, array.shape[1] - 1, target_shape[1])

    interpolator = RegularGridInterpolator(
        (src_y, src_x),
        array,
        bounds_error=False,
        fill_value=None,
    )

    tgt_yy, tgt_xx = np.meshgrid(tgt_y, tgt_x, indexing="ij")
    resized = interpolator((tgt_yy, tgt_xx))

    return resized


def normalize_longitude_if_dateline_crossed(lon):
    """Normalize longitude only when the swath crosses the dateline."""
    lon_normalized = lon.copy()
    finite_lon = lon_normalized[np.isfinite(lon_normalized)]

    if finite_lon.size == 0:
        return lon_normalized

    if np.nanmax(finite_lon) - np.nanmin(finite_lon) > 180.0:
        lon_normalized[lon_normalized < 0.0] += 360.0

    return lon_normalized


def load_myd021_file_list(input_dir):
    file_list = sorted(glob(os.path.join(input_dir, "*.nc")))
    if not file_list:
        raise ValueError(f"No nc files found in: {input_dir}")
    return file_list


def build_rotated_square_grid(center_lon, center_lat, side_km, resolution_m, angle_deg):
    """
    Build a regular grid covering a rotated square region.

    The square is centered at (center_lon, center_lat), with side length side_km.
    The local square grid is rotated clockwise by angle_deg.
    """
    n_cells = int(side_km * 1000.0 / resolution_m)
    half_side_km = side_km / 2.0

    local_x = np.linspace(-half_side_km, half_side_km, n_cells)
    local_y = np.linspace(-half_side_km, half_side_km, n_cells)
    local_xx, local_yy = np.meshgrid(local_x, local_y)

    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    rot_x = local_xx * cos_a + local_yy * sin_a
    rot_y = -local_xx * sin_a + local_yy * cos_a

    lon_offset = lon_km_to_deg(rot_x, center_lat)
    lat_offset = lat_km_to_deg(rot_y)

    grid_lon = center_lon + lon_offset
    grid_lat = center_lat + lat_offset

    return grid_lon, grid_lat, n_cells, n_cells


def get_square_corners_from_grid(grid_lon, grid_lat):
    """
    Get the four corners of the rotated target grid.

    Order:
    top-left, top-right, bottom-right, bottom-left.
    """
    corners_lon = np.array([
        grid_lon[0, 0],
        grid_lon[0, -1],
        grid_lon[-1, -1],
        grid_lon[-1, 0],
    ])

    corners_lat = np.array([
        grid_lat[0, 0],
        grid_lat[0, -1],
        grid_lat[-1, -1],
        grid_lat[-1, 0],
    ])

    return corners_lon, corners_lat


def quick_check_square_in_swath(square_lon, square_lat, swath_lon, swath_lat):
    """
    Fast pre-check before interpolation.

    It checks whether the four corners of the rotated target square are inside
    the approximate quadrilateral formed by the four swath corners.

    This is much faster than interpolating first and then checking NaNs.
    """
    finite_mask = np.isfinite(swath_lon) & np.isfinite(swath_lat)
    if not np.any(finite_mask):
        return False

    # Fast bounding-box rejection
    swath_lon_min = np.nanmin(swath_lon)
    swath_lon_max = np.nanmax(swath_lon)
    swath_lat_min = np.nanmin(swath_lat)
    swath_lat_max = np.nanmax(swath_lat)

    if (
        np.nanmin(square_lon) < swath_lon_min
        or np.nanmax(square_lon) > swath_lon_max
        or np.nanmin(square_lat) < swath_lat_min
        or np.nanmax(square_lat) > swath_lat_max
    ):
        return False

    n_rows, n_cols = swath_lon.shape

    # Swath outer quadrilateral:
    # top-left, top-right, bottom-right, bottom-left
    swath_poly_lon = np.array([
        swath_lon[0, 0],
        swath_lon[0, n_cols - 1],
        swath_lon[n_rows - 1, n_cols - 1],
        swath_lon[n_rows - 1, 0],
    ])

    swath_poly_lat = np.array([
        swath_lat[0, 0],
        swath_lat[0, n_cols - 1],
        swath_lat[n_rows - 1, n_cols - 1],
        swath_lat[n_rows - 1, 0],
    ])

    if not np.all(np.isfinite(swath_poly_lon)) or not np.all(np.isfinite(swath_poly_lat)):
        return False

    swath_path = Path(np.column_stack([swath_poly_lon, swath_poly_lat]))
    square_points = np.column_stack([square_lon, square_lat])

    inside = swath_path.contains_points(square_points, radius=1e-9)
    return bool(np.all(inside))


def resample_to_grid(data, src_lon, src_lat, tgt_lon, tgt_lat, margin_deg=0.30):
    """
    Resample swath data onto target grid using LinearNDInterpolator.

    Speed-up:
    Only source pixels near the target-grid bounding box are used.
    """
    src_lon = np.asarray(src_lon, dtype=float)
    src_lat = np.asarray(src_lat, dtype=float)
    data = np.asarray(data, dtype=float)

    lon_min = np.nanmin(tgt_lon) - margin_deg
    lon_max = np.nanmax(tgt_lon) + margin_deg
    lat_min = np.nanmin(tgt_lat) - margin_deg
    lat_max = np.nanmax(tgt_lat) + margin_deg

    near_mask = (
        np.isfinite(src_lon)
        & np.isfinite(src_lat)
        & np.isfinite(data)
        & (src_lon >= lon_min)
        & (src_lon <= lon_max)
        & (src_lat >= lat_min)
        & (src_lat <= lat_max)
    )

    n_valid = np.count_nonzero(near_mask)
    if n_valid < 10:
        return np.full(tgt_lon.shape, np.nan), np.full(tgt_lon.shape, False)

    src_points = np.column_stack((src_lon[near_mask], src_lat[near_mask]))
    src_values = data[near_mask]

    tgt_points = np.column_stack((tgt_lon.ravel(), tgt_lat.ravel()))

    try:
        interpolator = LinearNDInterpolator(src_points, src_values, fill_value=np.nan)
        gridded = interpolator(tgt_points).reshape(tgt_lon.shape)
    except Exception as exc:
        print(f"Interpolation failed: {exc}")
        return np.full(tgt_lon.shape, np.nan), np.full(tgt_lon.shape, False)

    valid_mask = np.isfinite(gridded)
    return gridded, valid_mask


def plot_data_matrix(data, title, cmap, cbar_label, output_path, vmin=None, vmax=None):
    """Plot the data matrix without axes."""
    finite_mask = np.isfinite(data)
    if not np.any(finite_mask):
        raise ValueError(f"{title} contains no valid data after masking.")

    if vmin is None or vmax is None:
        vmin = np.nanpercentile(data, 2)
        vmax = np.nanpercentile(data, 98)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    im = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        interpolation="none",
        rasterized=True,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    ax.set_title(title)
    ax.axis("off")

    fig.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    mod_file_lst = load_myd021_file_list(INPUT_DIR)
    files_to_process = mod_file_lst[FILE_START:FILE_END]

    plotted_count = 0
    skipped_count = 0
    quick_skipped_count = 0

    grid_lon, grid_lat, n_rows, n_cols = build_rotated_square_grid(
        CENTER_LON,
        CENTER_LAT,
        SIDE_LENGTH_KM,
        RESOLUTION_M,
        ROTATION_ANGLE_DEG,
    )

    print(f"Target grid dimensions: {n_rows} x {n_cols}")

    square_corners_lon, square_corners_lat = get_square_corners_from_grid(
        grid_lon,
        grid_lat,
    )

    print("Target square corners:")
    for i in range(4):
        print(
            f"  corner {i + 1}: "
            f"lon={square_corners_lon[i]:.4f}, lat={square_corners_lat[i]:.4f}"
        )

    for hdf_file in files_to_process:
        dataset = None

        try:
            dataset = Dataset(hdf_file, "r")
        except OSError:
            skipped_count += 1
            print(f"Skipped unreadable file: {hdf_file}")
            continue

        try:
            # Step 1: read only geolocation first
            lat = read_nc_field(dataset, LAT_PATH)
            lon = read_nc_field(dataset, LON_PATH)
            lon = normalize_longitude_if_dateline_crossed(lon)

            # Step 2: fast pre-check before reading bands and interpolation
            if not quick_check_square_in_swath(
                square_corners_lon,
                square_corners_lat,
                lon,
                lat,
            ):
                skipped_count += 1
                quick_skipped_count += 1
                print(f"Skipped quickly, target square outside swath: {hdf_file}")
                continue

            # Step 3: read and scale reflectance band 7, 2.1 um
            ref_var = dataset[REFSB_500_PATH]
            ref_index = get_band_index(ref_var, 7)
            ref_21 = read_and_scale_band(
                ref_var,
                ref_index,
                "reflectance_scales",
                "reflectance_offsets",
            )

            # Step 4: read and scale emissive bands, then calculate BT11 - BT3.7
            emissive_var = dataset[EMISSIVE_PATH]

            bt11_index = get_band_index(emissive_var, 31)
            bt37_index = get_band_index(emissive_var, 20)

            rad_11 = read_and_scale_band(
                emissive_var,
                bt11_index,
                "radiance_scales",
                "radiance_offsets",
            )

            rad_37 = read_and_scale_band(
                emissive_var,
                bt37_index,
                "radiance_scales",
                "radiance_offsets",
            )

            tb_11 = radiance2tb(rad_11, 11.0)
            tb_37 = radiance2tb(rad_37, 3.7)
            tb_diff = tb_11 - tb_37

            # Step 5: resize geolocation to match each data field
            lat_ref = resize_2d(lat, ref_21.shape)
            lon_ref = resize_2d(lon, ref_21.shape)

            if tb_diff.shape == ref_21.shape:
                lat_tb = lat_ref
                lon_tb = lon_ref
            else:
                lat_tb = resize_2d(lat, tb_diff.shape)
                lon_tb = resize_2d(lon, tb_diff.shape)

            # Step 6: interpolate only after quick pre-check
            ref_21_grid, ref_valid = resample_to_grid(
                ref_21,
                lon_ref,
                lat_ref,
                grid_lon,
                grid_lat,
                margin_deg=CROP_MARGIN_DEG,
            )

            tb_diff_grid, tb_valid = resample_to_grid(
                tb_diff,
                lon_tb,
                lat_tb,
                grid_lon,
                grid_lat,
                margin_deg=CROP_MARGIN_DEG,
            )

            # Step 7: final strict check for gaps
            if not np.all(ref_valid) or not np.all(tb_valid):
                skipped_count += 1
                print(f"Skipped after interpolation, gaps in square region: {hdf_file}")
                continue

            stem = os.path.splitext(os.path.basename(hdf_file))[0]

            ref_output_path = os.path.join(OUT_DIR, f"{stem}_ref_2.1um.png")
            plot_data_matrix(
                ref_21_grid,
                title="2.1 um Reflectance",
                cmap="jet",
                cbar_label="Reflectance",
                output_path=ref_output_path,
            )

            tb_output_path = os.path.join(OUT_DIR, f"{stem}_tb11_minus_tb3.7.png")
            plot_data_matrix(
                tb_diff_grid,
                title="Brightness Temperature Difference (11 um - 3.7 um)",
                cmap="RdBu_r",
                cbar_label="Brightness Temperature Difference (K)",
                output_path=tb_output_path,
            )

            plotted_count += 1
            print(f"Saved figures for: {hdf_file}")

        finally:
            if dataset is not None:
                dataset.close()

    print(
        f"Finished. plotted={plotted_count}, "
        f"skipped={skipped_count}, "
        f"quick_skipped={quick_skipped_count}"
    )


if __name__ == "__main__":
    main()