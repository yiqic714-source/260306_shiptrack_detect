import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from pyhdf.SD import SD, SDC
from scipy.ndimage import zoom
from scipy.interpolate import griddata
from scipy.signal import detrend
from math import radians, cos

# 物理常数
h = 6.62607015e-34
c = 2.99792458e8
k = 1.380649e-23
EARTH_RADIUS = 6371.0

# 配置参数
CONFIG = {
    "hdf_file": "/home/chenyiqi/260306_fig_rotation/MYD021KM.A2018218.1920.061.2018219152212.pscs_000502452615.hdf",
    "out_dir": "/home/chenyiqi/260306_fig_rotation/figs",
    "lon_range": (-82.5, -72.5),
    "lat_range": (-27.5, -17.5),
    "fig_size": (8, 6),
    "dpi": 300,
    "marker_points": [(-75.5, -21), (-74.8, -22.25)],
    "buffer_km": 75,
    "cross_axis_n": 150,
    "along_axis_n": 30,
    "analysis_window_km": 13,
    "point_exame_window_km": 45
}

# 基础函数
def init_env():
    os.makedirs(CONFIG["out_dir"], exist_ok=True)

def deg2km(lon_deg, lat_deg, ref_lat):
    lat_km = lat_deg * (2 * np.pi * EARTH_RADIUS) / 360.0
    lon_km = lon_deg * (2 * np.pi * EARTH_RADIUS * cos(radians(ref_lat))) / 360.0
    return lon_km, lat_km

def km2deg(dlon_km, dlat_km, ref_lat):
    lat_deg = dlat_km * 360.0 / (2 * np.pi * EARTH_RADIUS)
    lon_deg = dlon_km * 360.0 / (2 * np.pi * EARTH_RADIUS * cos(radians(ref_lat)))
    return lon_deg, lat_deg

def get_axis_geometry(p1, p2):
    lon1, lat1 = p1
    lon2, lat2 = p2
    ref_lat = (lat1 + lat2)/2
    dx_km, dy_km = deg2km(lon2 - lon1, lat2 - lat1, ref_lat)
    length_km = np.hypot(dx_km, dy_km)
    ux, uy = dx_km/length_km, dy_km/length_km
    vx, vy = -uy, ux
    return ref_lat, length_km, ux, uy, vx, vy

def build_rotated_grid(p1, p2, buffer_km, cross_axis_n=120, along_axis_n=30):
    ref_lat, length_km, ux, uy, vx, vy = get_axis_geometry(p1, p2)
    cross_edges = np.linspace(-buffer_km, buffer_km, cross_axis_n+1)
    cross_centers = 0.5*(cross_edges[:-1]+cross_edges[1:])
    along_edges = np.linspace(0, length_km, along_axis_n+1)
    along_centers = 0.5*(along_edges[:-1]+along_edges[1:])
    A, C = np.meshgrid(along_centers, cross_centers, indexing='ij')
    dx_km = A*ux + C*vx
    dy_km = A*uy + C*vy
    dlon, dlat = km2deg(dx_km, dy_km, ref_lat)
    lon_grid = p1[0]+dlon
    lat_grid = p1[1]+dlat
    return lon_grid, lat_grid, along_edges, cross_edges, along_centers, cross_centers, length_km

def read_modis_data(hdf, var_name):
    sds = hdf.select(var_name)
    data = sds[:].astype(float)
    attrs = sds.attributes()
    
    fv = attrs.get('_FillValue')
    if fv is not None:
        data[data == fv] = np.nan
    
    if 'radiance_scales' in attrs and 'radiance_offsets' in attrs:
        scales = np.array(attrs['radiance_scales'], float)
        offsets = np.array(attrs['radiance_offsets'], float)
        
        scales = scales.ravel() if scales.ndim > 0 else np.array([scales])
        offsets = offsets.ravel() if offsets.ndim > 0 else np.array([offsets])
        
        for i in range(data.shape[0]):
            scale = scales[i] if i < len(scales) else scales[0]
            offset = offsets[i] if i < len(offsets) else offsets[0]
            data[i] = (data[i] - offset) * scale
    
    return data

def radiance2tb(radiance, wavelength_um):
    wavelength_m = wavelength_um * 1e-6
    B = radiance * 1e6
    tb = np.full_like(radiance, np.nan)
    mask = np.isfinite(B) & (B > 0)
    if np.any(mask):
        c1 = 2 * h * c**2
        c2 = h * c / k
        tb[mask] = c2 / (wavelength_m * np.log(1 + c1 / (wavelength_m**5 * B[mask])))
    return tb

def expand_latlon_to_data_shape(lat, lon, data_shape):
    """修复维度缩放错误：之前把zoom_y写成了zoom_x，导致维度不匹配"""
    if lat.shape == data_shape and lon.shape == data_shape:
        return lat, lon
    # 正确计算y/x方向的缩放因子
    zoom_y = data_shape[0] / lat.shape[0]
    zoom_x = data_shape[1] / lon.shape[1]
    # 正确使用各自的缩放因子
    lat_big = zoom(lat, (zoom_y, zoom_x), order=1)
    lon_big = zoom(lon, (zoom_y, zoom_x), order=1)
    return lat_big, lon_big

def interpolate_to_rotated_grid(data, lat, lon, p1, p2, buffer_km, cross_axis_n=120, along_axis_n=30):
    lat_big, lon_big = expand_latlon_to_data_shape(lat, lon, data.shape)
    lon_grid, lat_grid, along_edges, cross_edges, along_centers, cross_centers, length_km = \
        build_rotated_grid(p1, p2, buffer_km, cross_axis_n, along_axis_n)

    lon_min, lon_max = np.nanmin(lon_grid)-0.2, np.nanmax(lon_grid)+0.2
    lat_min, lat_max = np.nanmin(lat_grid)-0.2, np.nanmax(lat_grid)+0.2
    mask = (np.isfinite(data) & np.isfinite(lat_big) & np.isfinite(lon_big) &
            (lon_big >= lon_min) & (lon_big <= lon_max) &
            (lat_big >= lat_min) & (lat_big <= lat_max))

    src_points = np.column_stack((lon_big[mask], lat_big[mask]))
    src_values = data[mask]
    tgt_points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    interp_values = griddata(src_points, src_values, tgt_points, method='linear')
    nan_mask = ~np.isfinite(interp_values)
    if np.any(nan_mask):
        interp_values[nan_mask] = griddata(src_points, src_values, tgt_points[nan_mask], method='nearest')
    data_rot = interp_values.reshape(lon_grid.shape)
    return data_rot, lon_grid, lat_grid, along_edges, cross_edges, cross_centers, length_km

def find_top_points(avg_list, cross_centers, window_size=13, min_distance=8, n_top=4):
    N = len(cross_centers)
    neg_corr_all = np.full(N, np.nan)
    
    # 提前对所有avg_1d做整条数据的去线性趋势处理
    detrended_avg_list = []
    for avg_1d in avg_list:
        # 对整条1维数组去线性趋势，保留有限值的结构
        mask_all = np.isfinite(avg_1d)
        if mask_all.sum() > 0:
            # 先复制原数组，避免修改输入
            avg_1d_detrend = avg_1d.copy()
            # 只对有限值部分去趋势
            avg_1d_detrend[mask_all] = detrend(avg_1d[mask_all], type='linear')
            detrended_avg_list.append(avg_1d_detrend)
        else:
            # 全是NaN的情况直接保留
            detrended_avg_list.append(avg_1d)
    
    for i in range(CONFIG["buffer_km"] - CONFIG["point_exame_window_km"], 
                   CONFIG["buffer_km"] + CONFIG["point_exame_window_km"]):
        local_corrs = []
        # 使用提前去趋势后的数组
        for avg_1d in detrended_avg_list:
            left = avg_1d[i - window_size:i]
            right = avg_1d[i + 1:i + 1 + window_size]
            mask = np.isfinite(left) & np.isfinite(right)
            if mask.sum() >= 5:
                xl = left[mask]
                xr = right[mask]
                # 移除窗口内的去趋势操作，直接计算相关系数
                corr = np.corrcoef(xl, xr)[0, 1]
                if np.isfinite(corr):
                    local_corrs.append(corr)
        if local_corrs:
            neg_corr_all[i] = np.mean(local_corrs)
    
    valid_idx = np.where(np.isfinite(neg_corr_all))[0]
    sorted_idx = valid_idx[np.argsort(neg_corr_all[valid_idx])]
    top_idx = []
    top_corrs = []
    for idx in sorted_idx:
        if all(abs(idx - exist) > min_distance for exist in top_idx):
            top_idx.append(idx)
            top_corrs.append(neg_corr_all[idx])
        if len(top_idx) == n_top:
            break
    return top_idx, top_corrs, neg_corr_all

def calculate_derivatives(data, x_coords):
    first_deriv = np.gradient(data, x_coords)
    second_deriv = np.gradient(first_deriv, x_coords)
    return first_deriv, second_deriv

def analyze_top_points(avg_list, data_rot_list, cross_centers, top_idx, top_corrs, window_km=13):
    curve_names = ["Tb 11-3.7", "Ref 2.1μm", "Ref 0.86μm"]
    results = {}
    feature_vectors = {}
    
    # 先计算每个点的基础特征
    for i, idx in enumerate(top_idx, 1):
        pos_km = cross_centers[idx]
        corr = top_corrs[i - 1]

        results[f"Top{i}"] = {
            "position_km": pos_km,
            "index": idx,
            "symmetry_corr": corr,
            "curves": {},
            "feature_vector": []
        }

        # 1D 窗口：用于均值/标准差/导数统计
        win_mask = (cross_centers >= pos_km - window_km) & (cross_centers <= pos_km + window_km)
        
        for curve_idx, (orig_data, rot2d, name) in enumerate(zip(avg_list, data_rot_list, curve_names)):
            win_data = orig_data[win_mask]
            mean_val = np.nanmean(win_data)
            std_val = np.nanstd(win_data)

            fd, sd = calculate_derivatives(orig_data, cross_centers)
            win_fd = fd[win_mask]
            win_sd = sd[win_mask]

            fd_mean = np.nanmean(win_fd)
            fd_std = np.nanstd(win_fd)
            sd_mean = np.nanmean(win_sd)

            # 新增：2D window 区域标准差
            # 这里按“全部 along-axis × cross-axis ±window_km”的带状2D区域计算
            win_2d = rot2d[:, win_mask]
            window2d_std = np.nanstd(win_2d)
            
            curve_stats = {
                "mean": mean_val,
                "std": std_val,
                "first_deriv_mean": fd_mean,
                "first_deriv_std": fd_std,
                "second_deriv_mean": sd_mean,
                "window2d_std": window2d_std
            }

            results[f"Top{i}"]["curves"][name] = curve_stats

            # 现在每个波段加入 6 个统计特征
            results[f"Top{i}"]["feature_vector"].extend([
                mean_val, std_val, fd_mean, fd_std, sd_mean, window2d_std
            ])

        # 再把 symmetry_corr 加进去
        results[f"Top{i}"]["feature_vector"].append(corr)

        feature_vectors[f"Top{i}"] = np.array(
            results[f"Top{i}"]["feature_vector"], dtype=np.float64
        )
    
    # 基于“所有前面特征”（18个波段特征 + symmetry_corr）计算平均欧氏距离
    top_names = [f"Top{i+1}" for i in range(len(top_idx))]
    for top_name in top_names:
        current_vec = feature_vectors[top_name]
        other_vecs = [feature_vectors[tn] for tn in top_names if tn != top_name]

        distances = []
        for vec in other_vecs:
            valid_mask = np.isfinite(current_vec) & np.isfinite(vec)
            if np.sum(valid_mask) > 0:
                dist = np.linalg.norm(current_vec[valid_mask] - vec[valid_mask])
                distances.append(dist)

        avg_distance = np.mean(distances) if distances else np.nan
        results[top_name]["avg_euclidean_distance"] = avg_distance
    
    return results

def save_results_to_csv(results, save_path):
    columns = [
        "Point_Name", "Position_km", "Index",

        "Tb11_3.7_mean", "Tb11_3.7_std", "Tb11_3.7_fd_mean", "Tb11_3.7_fd_std", "Tb11_3.7_sd_mean", "Tb11_3.7_2dstd",
        "Ref2.1_mean", "Ref2.1_std", "Ref2.1_fd_mean", "Ref2.1_fd_std", "Ref2.1_sd_mean", "Ref2.1_2dstd",
        "Ref0.86_mean", "Ref0.86_std", "Ref0.86_fd_mean", "Ref0.86_fd_std", "Ref0.86_sd_mean", "Ref0.86_2dstd",

        "Symmetry_Correlation_detrend",
        "Avg_Euclidean_Distance"
    ]

    rows = []
    for top_name, data in results.items():
        row = [
            top_name,
            round(data["position_km"], 4),
            data["index"],

            round(data["curves"]["Tb 11-3.7"]["mean"], 8),
            round(data["curves"]["Tb 11-3.7"]["std"], 8),
            round(data["curves"]["Tb 11-3.7"]["first_deriv_mean"], 8),
            round(data["curves"]["Tb 11-3.7"]["first_deriv_std"], 8),
            round(data["curves"]["Tb 11-3.7"]["second_deriv_mean"], 8),
            round(data["curves"]["Tb 11-3.7"]["window2d_std"], 8),

            round(data["curves"]["Ref 2.1μm"]["mean"], 8),
            round(data["curves"]["Ref 2.1μm"]["std"], 8),
            round(data["curves"]["Ref 2.1μm"]["first_deriv_mean"], 8),
            round(data["curves"]["Ref 2.1μm"]["first_deriv_std"], 8),
            round(data["curves"]["Ref 2.1μm"]["second_deriv_mean"], 8),
            round(data["curves"]["Ref 2.1μm"]["window2d_std"], 8),

            round(data["curves"]["Ref 0.86μm"]["mean"], 8),
            round(data["curves"]["Ref 0.86μm"]["std"], 8),
            round(data["curves"]["Ref 0.86μm"]["first_deriv_mean"], 8),
            round(data["curves"]["Ref 0.86μm"]["first_deriv_std"], 8),
            round(data["curves"]["Ref 0.86μm"]["second_deriv_mean"], 8),
            round(data["curves"]["Ref 0.86μm"]["window2d_std"], 8),

            round(data["symmetry_corr"], 8),
            round(data["avg_euclidean_distance"], 8) if np.isfinite(data["avg_euclidean_distance"]) else np.nan
        ]
        rows.append(row)

    with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        writer.writerows(rows)

    print("\n✅ 已为每个点、每个波段添加 2D window 标准差特征")
    print("✅ 欧氏距离已基于所有前面特征（18个波段统计特征 + symmetry_corr）计算")
    print(f"文件已保存到：{save_path}")


def plot_three_curves_with_top(avg_list, centered_avg_list, labels, cross_centers, title, cbar_label, save_name, top_idx):
    plt.figure(figsize=(10,6))
    colors = ['blue','green','orange']
    for centered_avg, label, color in zip(centered_avg_list, labels, colors):
        plt.plot(cross_centers, centered_avg, '-o', color=color, label=label, markersize=3)
    for i, idx in enumerate(top_idx,1):
        x = cross_centers[idx]
        plt.axvspan(x - CONFIG["analysis_window_km"], x + CONFIG["analysis_window_km"],
                    alpha=0.1, color='gray', label=f'Top{i} window' if i==1 else "")
        max_val = np.nanmax([centered_avg[idx] for centered_avg in centered_avg_list])
        for ca, c in zip(centered_avg_list, colors):
            plt.scatter(x, ca[idx], color='red', s=80, zorder=10, edgecolor='black')
        plt.text(x, max_val+0.05, f'Top{i}', color='red', fontsize=10, weight='bold', ha='center')
    plt.xlabel("Cross-axis distance (km)")
    plt.ylabel(f"{cbar_label} (Centered)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_name}.png", dpi=CONFIG["dpi"], bbox_inches='tight')
    plt.close()

def main():
    init_env()
    hdf = SD(CONFIG["hdf_file"], SDC.READ)
    lat = np.nan_to_num(hdf.select("Latitude")[:].astype(float), nan=0)
    lon = np.nan_to_num(hdf.select("Longitude")[:].astype(float), nan=0)

    emiss_data = read_modis_data(hdf, "EV_1KM_Emissive")
    tb_diff = radiance2tb(emiss_data[1], 11.0) - radiance2tb(emiss_data[0], 3.7)
    data_rot1, _, _, _, _, cross_centers, _ = interpolate_to_rotated_grid(
        tb_diff, lat, lon, *CONFIG["marker_points"], CONFIG["buffer_km"],
        CONFIG["cross_axis_n"], CONFIG["along_axis_n"])
    avg1 = np.nanmean(data_rot1, axis=0)
    centered_avg1 = avg1 - np.nanmean(avg1)

    ref_data = read_modis_data(hdf, "EV_500_Aggr1km_RefSB")[0]
    data_rot2, _, _, _, _, _, _ = interpolate_to_rotated_grid(
        ref_data, lat, lon, *CONFIG["marker_points"], CONFIG["buffer_km"],
        CONFIG["cross_axis_n"], CONFIG["along_axis_n"])
    avg2 = np.nanmean(data_rot2, axis=0)
    centered_avg2 = avg2 - np.nanmean(avg2)

    ref_data = read_modis_data(hdf, "EV_250_Aggr1km_RefSB")[1]
    data_rot3, _, _, _, _, _, _ = interpolate_to_rotated_grid(
        ref_data, lat, lon, *CONFIG["marker_points"], CONFIG["buffer_km"],
        CONFIG["cross_axis_n"], CONFIG["along_axis_n"])
    avg3 = np.nanmean(data_rot3, axis=0)
    centered_avg3 = avg3 - np.nanmean(avg3)
    
    avg_list = [avg1, avg2, avg3]
    centered_avg_list = [centered_avg1, centered_avg2, centered_avg3]
    top_idx, top_corrs, _ = find_top_points(avg_list, cross_centers, window_size=13, min_distance=10, n_top=3)

    plot_three_curves_with_top(
        avg_list, centered_avg_list,
        ["Tb 11-3.7","Ref 2.1μm","Ref 0.86μm"],
        cross_centers, "Along-axis average (Centered, Top3)",
        "Value", f"{CONFIG['out_dir']}/three_curves_symmetric_points", top_idx)

    data_rot_list = [data_rot1, data_rot2, data_rot3]
    results = analyze_top_points(
        avg_list,
        data_rot_list,
        cross_centers,
        top_idx,
        top_corrs,
        CONFIG["analysis_window_km"]
    )
    csv_path = f"{CONFIG['out_dir']}/top_points_analysis.csv"
    save_results_to_csv(results, csv_path)

    hdf.end()
    print("\ndone")

if __name__ == "__main__":
    main()