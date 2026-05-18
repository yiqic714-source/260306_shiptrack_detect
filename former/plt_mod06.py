import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from pyhdf.SD import SD, SDC


YEAR = 2020
MONTH = 8
FNAME_ENDING = "S"
VAR_NAME = "Cloud_Effective_Radius"
OUT_DIR = "/home/chenyiqi/260306_fig_rotation/figs"
TARGET_LON_RANGE = (-85.0, -70.0)
TARGET_LAT_RANGE = (-35.0, -10.0)


def read_and_mask_mod_variable(hdf, var_name):
	sds = hdf.select(var_name)
	data = sds[:].astype(float)

	attrs = sds.attributes()
	fill_value = attrs.get("_FillValue")
	scale_factor = attrs.get("scale_factor")
	offset = attrs.get("add_offset")

	if fill_value is not None:
		data[data == fill_value] = np.nan
	if offset is not None:
		data = data - offset
	if scale_factor is not None:
		data = data * scale_factor

	return data


def resize_2d(array, target_shape):
	if array.shape == target_shape:
		return array

	src_y = np.arange(array.shape[0], dtype=float)
	src_x = np.arange(array.shape[1], dtype=float)
	tgt_y = np.linspace(0, array.shape[0] - 1, target_shape[0])
	tgt_x = np.linspace(0, array.shape[1] - 1, target_shape[1])

	interp_x = np.empty((array.shape[0], target_shape[1]), dtype=float)
	for row_idx in range(array.shape[0]):
		interp_x[row_idx, :] = np.interp(tgt_x, src_x, array[row_idx, :])

	interp_y = np.empty(target_shape, dtype=float)
	for col_idx in range(target_shape[1]):
		interp_y[:, col_idx] = np.interp(tgt_y, src_y, interp_x[:, col_idx])

	return interp_y


def normalize_longitude_if_dateline_crossed(lon):
	lon_normalized = lon.copy()
	finite_lon = lon_normalized[np.isfinite(lon_normalized)]
	if finite_lon.size == 0:
		return lon_normalized

	if np.nanmax(finite_lon) - np.nanmin(finite_lon) > 180:
		lon_normalized[lon_normalized < 0] += 360

	return lon_normalized


def has_pixels_in_target_region(lat, lon, lon_range, lat_range):
	lon_wrapped = ((lon + 180.0) % 360.0) - 180.0
	mask = (
		np.isfinite(lat)
		& np.isfinite(lon_wrapped)
		& (lon_wrapped >= lon_range[0])
		& (lon_wrapped <= lon_range[1])
		& (lat >= lat_range[0])
		& (lat <= lat_range[1])
	)
	return bool(np.any(mask))


def load_mod_file_list(year, month, fname_ending):
	mod_pkl = f"/data/chenyiqi/251028_albedo_cot/mod06/MOD06_files_{year}{month:02d}{fname_ending}_lon_m180_0.pkl"
	with open(mod_pkl, "rb") as f:
		mod_file_lst = pickle.load(f)["terra"]

	if not mod_file_lst:
		raise ValueError(f"No terra files found in: {mod_pkl}")

	default_month_dir = f"/data/chenyiqi/251028_albedo_cot/mod06/{year}{month:02d}{fname_ending}"
	resolved_files = []
	for file_name in mod_file_lst:
		file_path = file_name if os.path.isabs(file_name) else os.path.join(default_month_dir, file_name)
		resolved_files.append(file_path)

	return resolved_files


def plot_spatial_distribution(lat, lon, data, attrs, output_path):
	finite_mask = np.isfinite(data)
	if not np.any(finite_mask):
		raise ValueError(f"{VAR_NAME} contains no valid data after masking.")

	valid_range = attrs.get("valid_range")
	if valid_range is not None:
		scale_factor = attrs.get("scale_factor", 1.0)
		offset = attrs.get("add_offset", 0.0)
		vmin = (float(valid_range[0]) - offset) * scale_factor
		vmax = (float(valid_range[1]) - offset) * scale_factor
	else:
		vmin = np.nanpercentile(data, 2)
		vmax = np.nanpercentile(data, 98)

	fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
	mesh = ax.pcolormesh(
		lon,
		lat,
		data,
		shading="auto",
		cmap="viridis",
		vmin=vmin,
		vmax=vmax,
		rasterized=True,
	)
	cbar = fig.colorbar(mesh, ax=ax)
	cbar.set_label(f"{VAR_NAME} ({attrs.get('units', '')})")

	ax.set_title(f"{VAR_NAME} Spatial Distribution")
	ax.set_xlabel("Longitude")
	ax.set_ylabel("Latitude")
	ax.set_xlim(np.nanmin(lon), np.nanmax(lon))
	ax.set_ylim(np.nanmin(lat), np.nanmax(lat))
	fig.tight_layout()
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)


def main():
	os.makedirs(OUT_DIR, exist_ok=True)
	mod_file_lst = load_mod_file_list(YEAR, MONTH, FNAME_ENDING)
	plotted_count = 0
	skipped_count = 0

	for hdf_file in mod_file_lst[0:200]:
		hdf = SD(hdf_file, SDC.READ)
		try:
			lat = hdf.select('Latitude')[:]
			lat[lat==-999] = np.nan
			lon = hdf.select('Longitude')[:]
			lon[lon==-999] = np.nan
			if not has_pixels_in_target_region(lat, lon, TARGET_LON_RANGE, TARGET_LAT_RANGE):
				skipped_count += 1
				continue

			lon = normalize_longitude_if_dateline_crossed(lon)

			var_sds = hdf.select(VAR_NAME)
			var_attrs = var_sds.attributes()
			data = read_and_mask_mod_variable(hdf, VAR_NAME)

			lat_resized = resize_2d(lat, data.shape)
			lon_resized = resize_2d(lon, data.shape)

			stem = os.path.splitext(os.path.basename(hdf_file))[0]
			output_path = os.path.join(OUT_DIR, f"{stem}_{VAR_NAME}_spatial_distribution.png")
			plot_spatial_distribution(lat_resized, lon_resized, data, var_attrs, output_path)
			plotted_count += 1
		finally:
			hdf.end()

		print(f"Saved figure to: {output_path}")

	print(f"Finished. plotted={plotted_count}, skipped={skipped_count}")


if __name__ == "__main__":
	main()
