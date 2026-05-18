import os
import numpy as np
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
from scipy.ndimage import zoom
from scipy.interpolate import griddata
from math import radians, cos, sqrt

# ===================== 物理常数 =====================
h = 6.62607015e-34    # 普朗克常数 (J·s)
c = 2.99792458e8      # 光速 (m/s)
k = 1.380649e-23      # 玻尔兹曼常数 (J/K)

# ===================== 配置参数 =====================
CONFIG = {
    "hdf_file": "/home/chenyiqi/260306_fig_rotation/MYD021KM.A2018218.1920.061.2018219152212.pscs_000502452615.hdf",
    "out_dir": "/home/chenyiqi/260306_fig_rotation/figs",
    "lon_range": (-82.5, -72.5),
    "lat_range": (-27.5, -17.5),
    "fig_size": (8, 6),
    "dpi": 300,
    "marker_points": [(-75.5, -21), (-74.8, -22.25)],  # 两个端点
    "buffer_km": 60,          # 轴线左右各25 km
    "cross_axis_res_km": 1.0, # 垂直于轴方向的分辨率
    "cross_axis_n": 120,       # 垂直方向共50格
    "along_axis_n": 30        # 平行轴方向共30格
}

# 地球半径（km）
EARTH_RADIUS = 6371.0


# ===================== 基础函数 =====================
def init_env():
    """初始化环境：创建输出目录"""
    os.makedirs(CONFIG["out_dir"], exist_ok=True)


def deg2km(lon_deg, lat_deg, ref_lat):
    """
    经纬度差值 -> 公里
    """
    lat_km = lat_deg * (2 * np.pi * EARTH_RADIUS) / 360.0
    lon_km = lon_deg * (2 * np.pi * EARTH_RADIUS * cos(radians(ref_lat))) / 360.0
    return lon_km, lat_km


def km2deg(dlon_km, dlat_km, ref_lat):
    """
    公里 -> 经纬度差值
    """
    lat_deg = dlat_km * 360.0 / (2 * np.pi * EARTH_RADIUS)
    lon_deg = dlon_km * 360.0 / (2 * np.pi * EARTH_RADIUS * cos(radians(ref_lat)))
    return lon_deg, lat_deg


def get_axis_geometry(p1, p2):
    """
    计算轴线几何信息（全部在局地km平面里做）
    返回：
        ref_lat, length_km,
        ux, uy  : 沿轴方向单位向量
        vx, vy  : 垂直轴方向单位向量（左手侧）
    """
    lon1, lat1 = p1
    lon2, lat2 = p2

    ref_lat = (lat1 + lat2) / 2.0
    dx_km, dy_km = deg2km(lon2 - lon1, lat2 - lat1, ref_lat)
    length_km = np.hypot(dx_km, dy_km)

    ux = dx_km / length_km
    uy = dy_km / length_km

    # 垂直方向单位向量（逆时针旋转90°）
    vx = -uy
    vy = ux

    return ref_lat, length_km, ux, uy, vx, vy


def get_buffer_box(p1, p2, buffer_km):
    """
    计算缓冲矩形框四个顶点（顺时针）
    """
    ref_lat, _, _, _, vx, vy = get_axis_geometry(p1, p2)

    dlon_perp, dlat_perp = km2deg(buffer_km * vx, buffer_km * vy, ref_lat)

    lon1, lat1 = p1
    lon2, lat2 = p2

    p1_left  = (lon1 - dlon_perp, lat1 - dlat_perp)
    p2_left  = (lon2 - dlon_perp, lat2 - dlat_perp)
    p2_right = (lon2 + dlon_perp, lat2 + dlat_perp)
    p1_right = (lon1 + dlon_perp, lat1 + dlat_perp)

    return [p1_left, p2_left, p2_right, p1_right]


def build_rotated_grid(p1, p2, buffer_km, cross_axis_n=50, along_axis_n=30):
    """
    构建旋转后的规则网格

    cross-axis: 垂直于轴方向，[-buffer_km, +buffer_km]，共 cross_axis_n 格
    along-axis: 平行于轴方向，[0, length_km]，共 along_axis_n 格

    返回：
        lon_grid, lat_grid              : 目标网格中心点经纬度，shape=(along_axis_n, cross_axis_n)
        along_edges, cross_edges        : pcolormesh 用边界
        along_centers, cross_centers    : 网格中心坐标
        length_km                       : 轴线总长度
    """
    ref_lat, length_km, ux, uy, vx, vy = get_axis_geometry(p1, p2)

    # 垂直轴方向：50格，每格1km（这里刚好对应[-25,25]）
    cross_edges = np.linspace(-buffer_km, buffer_km, cross_axis_n + 1)
    cross_centers = 0.5 * (cross_edges[:-1] + cross_edges[1:])

    # 平行轴方向：总长度均分30格
    along_edges = np.linspace(0, length_km, along_axis_n + 1)
    along_centers = 0.5 * (along_edges[:-1] + along_edges[1:])

    # A: along-axis, C: cross-axis
    A, C = np.meshgrid(along_centers, cross_centers, indexing='ij')

    # 在局地km坐标系中表示
    dx_km = A * ux + C * vx
    dy_km = A * uy + C * vy

    # 转回经纬度
    dlon, dlat = km2deg(dx_km, dy_km, ref_lat)
    lon_grid = p1[0] + dlon
    lat_grid = p1[1] + dlat

    return lon_grid, lat_grid, along_edges, cross_edges, along_centers, cross_centers, length_km


def read_modis_data(hdf, var_name):
    """通用MODIS数据读取+定标函数"""
    sds = hdf.select(var_name)
    data = sds[:].astype(float)
    attrs = sds.attributes()

    # 填充值处理
    fv = attrs.get('_FillValue')
    if fv is not None:
        data[data == fv] = np.nan

    # 辐射定标
    if 'radiance_scales' in attrs and 'radiance_offsets' in attrs:
        scales = np.array(attrs['radiance_scales'], float)
        offsets = np.array(attrs['radiance_offsets'], float)
        for i in range(data.shape[0]):
            scale = scales if scales.ndim == 0 else scales[i] if i < len(scales) else scales[0]
            offset = offsets if offsets.ndim == 0 else offsets[i] if i < len(offsets) else offsets[0]
            data[i] = (data[i] - offset) * scale
    return data


def radiance2tb(radiance, wavelength_um):
    """辐射亮度转亮温"""
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
    """
    如果经纬度分辨率低于数据分辨率，则插值放大到与data同形状
    """
    if lat.shape == data_shape and lon.shape == data_shape:
        return lat, lon

    zoom_y = data_shape[0] / lat.shape[0]
    zoom_x = data_shape[1] / lon.shape[1]

    lat_big = zoom(lat, (zoom_y, zoom_x), order=1)
    lon_big = zoom(lon, (zoom_y, zoom_x), order=1)
    return lat_big, lon_big


# ===================== 原图绘制 =====================
def plot_modis_img(data, lat, lon, title, cmap, cbar_label, save_name):
    """原始经纬度图"""
    lat_big, lon_big = expand_latlon_to_data_shape(lat, lon, data.shape)

    plt.figure(figsize=CONFIG["fig_size"])
    plt.pcolormesh(lon_big, lat_big, data, shading='auto', cmap=cmap)
    plt.colorbar(label=cbar_label)

    # 绘制缓冲框
    p1, p2 = CONFIG["marker_points"]
    box_vertices = get_buffer_box(p1, p2, CONFIG["buffer_km"])
    box_lons = [v[0] for v in box_vertices] + [box_vertices[0][0]]
    box_lats = [v[1] for v in box_vertices] + [box_vertices[0][1]]
    plt.plot(box_lons, box_lats, 'y--', linewidth=2, alpha=0.8,
             label=f"{CONFIG['buffer_km']} km Buffer")

    # 标注两点
    for idx, (lon_p, lat_p) in enumerate(CONFIG["marker_points"], 1):
        plt.scatter(lon_p, lat_p, color='red', marker='*', s=200,
                    edgecolor='white', linewidth=1.5, zorder=10)
        plt.text(lon_p + 0.1, lat_p + 0.1, f"Point {idx}",
                 color='white', fontsize=10, weight='bold', zorder=10)

    # 轴线
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'w-', linewidth=1.5, alpha=0.8)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.xlim(CONFIG["lon_range"])
    plt.ylim(CONFIG["lat_range"])
    plt.title(title)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{CONFIG['out_dir']}/{save_name}.png", dpi=CONFIG["dpi"])
    plt.close()


# ===================== 旋转网格插值 =====================
def interpolate_to_rotated_grid(data, lat, lon, p1, p2, buffer_km,
                                cross_axis_n=50, along_axis_n=30):
    """
    将原始变量插值到旋转网格
    返回：
        data_rot, along_edges, cross_edges, along_centers, cross_centers, length_km
    """
    lat_big, lon_big = expand_latlon_to_data_shape(lat, lon, data.shape)

    lon_grid, lat_grid, along_edges, cross_edges, along_centers, cross_centers, length_km = \
        build_rotated_grid(p1, p2, buffer_km, cross_axis_n, along_axis_n)

    # 为了提高效率，只取目标框附近的数据
    lon_min = np.nanmin(lon_grid) - 0.2
    lon_max = np.nanmax(lon_grid) + 0.2
    lat_min = np.nanmin(lat_grid) - 0.2
    lat_max = np.nanmax(lat_grid) + 0.2

    mask = (
        np.isfinite(data) &
        np.isfinite(lat_big) &
        np.isfinite(lon_big) &
        (lon_big >= lon_min) & (lon_big <= lon_max) &
        (lat_big >= lat_min) & (lat_big <= lat_max)
    )

    src_points = np.column_stack((lon_big[mask], lat_big[mask]))
    src_values = data[mask]

    tgt_points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

    # 先线性插值
    interp_values = griddata(src_points, src_values, tgt_points, method='linear')

    # 再用最近邻补NaN
    nan_mask = ~np.isfinite(interp_values)
    if np.any(nan_mask):
        interp_values[nan_mask] = griddata(
            src_points, src_values, tgt_points[nan_mask], method='nearest'
        )

    data_rot = interp_values.reshape(lon_grid.shape)

    return data_rot, along_edges, cross_edges, along_centers, cross_centers, length_km


def plot_rotated_grid(data_rot, along_edges, cross_edges, title, cmap, cbar_label, save_name):
    """
    绘制旋转后规则网格图
    x轴：垂直于轴方向距离(km)
    y轴：沿轴方向距离(km)
    """
    plt.figure(figsize=CONFIG["fig_size"])
    plt.pcolormesh(cross_edges, along_edges, data_rot, shading='auto', cmap=cmap)
    plt.colorbar(label=cbar_label)

    plt.xlabel("Cross-axis distance (km)")
    plt.ylabel("Along-axis distance (km)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['out_dir']}/{save_name}.png", dpi=CONFIG["dpi"])
    plt.close()

def plot_along_axis_avg(data_rot, cross_centers, title, cmap, cbar_label, save_name):
    """
    沿纵轴（沿轴方向）做平均，得到1D数组，并标出左右15格负相关最强点
    不允许标记相邻点，保证三个Top点不挨着
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 平均 -> 1D
    avg_1d = np.nanmean(data_rot, axis=0)
    N = len(avg_1d)
    window_size = 13  # 左右格数

    neg_corr = np.full(N, np.nan)

    # 计算左右15格的负相关
    for i in range(window_size, N - window_size):
        left = avg_1d[i - window_size:i]
        right = avg_1d[i + 1:i + 1 + window_size]
        mask = np.isfinite(left) & np.isfinite(right)
        if mask.sum() >= 5:
            corr = np.corrcoef(left[mask], right[mask])[0,1]
            neg_corr[i] = corr

    # 找负相关最强的前三个点，且不相邻
    sorted_idx = np.argsort(neg_corr)  # 从小到大（最负相关优先）
    top3_idx = []
    for idx in sorted_idx:
        if np.isnan(neg_corr[idx]):
            continue
        # 检查是否与已有Top点距离过近
        if all(abs(idx - exist) > 3 for exist in top3_idx):
            top3_idx.append(idx)
        if len(top3_idx) == 2:
            break

    top3_values = avg_1d[top3_idx]
    top3_positions = cross_centers[top3_idx]

    # 绘图
    plt.figure(figsize=CONFIG["fig_size"])
    plt.plot(cross_centers, avg_1d, '-o', color='blue', label='Along-axis average')
    plt.xlabel("Cross-axis distance (km)")
    plt.ylabel(cbar_label)
    plt.title(f"{title} (Along-axis average)")
    plt.grid(True)

    # 标记三个Top点
    for i, pos, val in zip(range(1,4), top3_positions, top3_values):
        plt.scatter(pos, val, color='red', s=100, zorder=10)
        plt.text(pos, val, f"Top{i}", color='red', fontsize=10, weight='bold', 
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"{CONFIG['out_dir']}/{save_name}_along_axis_avg.png", dpi=CONFIG["dpi"])
    plt.close()

    print(f"{save_name}: saved along-axis average profile, length={len(avg_1d)}, top3 positions = {top3_positions}, corr = {neg_corr[top3_idx]}")
    
def process_and_plot(data, lat, lon, title, cmap, cbar_label, save_name):
    """
    同时输出：
    1. 原始经纬度图
    2. 旋转规则网格插值图
    3. 沿轴平均1D图
    """
    # 原图
    plot_modis_img(data, lat, lon, title, cmap, cbar_label, save_name)

    # 旋转网格插值
    data_rot, along_edges, cross_edges, along_centers, cross_centers, length_km = \
        interpolate_to_rotated_grid(
            data, lat, lon,
            CONFIG["marker_points"][0], CONFIG["marker_points"][1],
            CONFIG["buffer_km"],
            CONFIG["cross_axis_n"], CONFIG["along_axis_n"]
        )

    # 新图：旋转网格
    plot_rotated_grid(
        data_rot, along_edges, cross_edges,
        title=f"{title} (Rotated grid: {CONFIG['cross_axis_n']} × {CONFIG['along_axis_n']})",
        cmap=cmap,
        cbar_label=cbar_label,
        save_name=f"{save_name}_rotated_grid"
    )

    # 新增图：沿纵轴平均
    plot_along_axis_avg(data_rot, cross_centers, title, cmap, cbar_label, save_name)

    print(f"{save_name}: rotated-grid shape = {data_rot.shape}, axis length = {length_km:.2f} km")


# ===================== 主流程 =====================
def main():
    init_env()
    hdf = SD(CONFIG["hdf_file"], SDC.READ)

    # 读取经纬度
    lat = np.nan_to_num(hdf.select("Latitude")[:].astype(float), nan=0)
    lon = np.nan_to_num(hdf.select("Longitude")[:].astype(float), nan=0)

    # 1. 亮温差（11μm - 3.7μm）
    emiss_data = read_modis_data(hdf, "EV_1KM_Emissive")
    tb_diff = radiance2tb(emiss_data[1], 11.0) - radiance2tb(emiss_data[0], 3.7)

    process_and_plot(
        tb_diff, lat, lon,
        title="Brightness Temperature Difference (11μm - 3.7μm)",
        cmap="RdBu_r",
        cbar_label="Brightness Temperature Difference (K)",
        save_name="11μm_minus_3.7μm_Tb_diff"
    )

    # 2. 反射波段
    ref_bands = [
        ("EV_500_Aggr1km_RefSB", 0, "2.1 μm", "2.1_μm"),
        ("EV_250_Aggr1km_RefSB", 0, "0.65 μm", "0.65_μm"),
        ("EV_250_Aggr1km_RefSB", 1, "0.86 μm", "0.86_μm")
    ]

    for var_name, band_idx, title, save_name in ref_bands:
        ref_data = read_modis_data(hdf, var_name)[band_idx]
        process_and_plot(
            ref_data, lat, lon,
            title=title,
            cmap="jet",
            cbar_label="Radiance (W/m²/μm/sr)",
            save_name=save_name
        )

    hdf.end()
    print("done - 原始4张 + 旋转网格插值4张，共8张图")


if __name__ == "__main__":
    main()