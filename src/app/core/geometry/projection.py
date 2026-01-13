import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

def apply_base_mesh_shape(vertices_2d, base_mesh_data, canvas_size):
    """
    2Dメッシュの頂点に対し、base_mesh_data(x,y -> z)を使って
    Z座標を補間し、ベースとなる3D形状を作成する。
    """
    if base_mesh_data is None or len(base_mesh_data) == 0:
        return np.zeros(len(vertices_2d))

    # 2Dキャンバスサイズ (固定値 1200x900 と仮定)
    w, h = canvas_size # (width, height)
    if w == 0 or h == 0: return np.zeros(len(vertices_2d))
        
    # 頂点座標を 0.0~1.0 に正規化
    norm_v2d = vertices_2d.copy()
    norm_v2d[:, 0] /= w
        
    # Viewportからのデータ(projected_data)は、extract_selection内で
    # 「1.0 - (p[1]/h)」でY反転されている。
    # ここでの vertices_2d は左上原点のピクセル座標なので、同様に反転させて合わせる。
    norm_v2d[:, 1] = 1.0 - (norm_v2d[:, 1] / h)
        
    # 補間器の作成 (X, Y -> Z)
    points = base_mesh_data[:, :2] # X, Y (0.0~1.0)
    values = base_mesh_data[:, 2]  # Z
        
    # 線形補間
    interpolator = LinearNDInterpolator(points, values)
    z_base = interpolator(norm_v2d)
        
    # 線形補間でNaNになった場所（＝輪郭の外側）は、0.0（変形なし）にする
    # これにより「点線の外側は変形しない」が実現できる
    z_base[np.isnan(z_base)] = 0.0
            
    return z_base

def map_2d_mesh_to_3d_surface(vertices_2d, base_mesh_data, canvas_size):
    """
    2Dメッシュの頂点を、base_mesh_data(x, y -> z)を用いて3D空間にマッピングする。
    """
    if base_mesh_data is None or len(base_mesh_data) == 0:
        # ベースデータがない場合は平坦なZ=0平面を返す
        return np.zeros(len(vertices_2d))

    w, h = canvas_size
    
    # キャンバスサイズで正規化 (0.0 ~ 1.0)
    norm_v2d = vertices_2d.copy()
    norm_v2d[:, 0] /= w
    norm_v2d[:, 1] = 1.0 - (norm_v2d[:, 1] / h) # Y軸反転

    # 補間器の作成 (X, Y -> Z)
    points = base_mesh_data[:, :2]
    values = base_mesh_data[:, 2]
    
    # 線形補間でZ値を求める
    interpolator = LinearNDInterpolator(points, values)
    z_mapped = interpolator(norm_v2d)
    
    # 輪郭外(NaN)は、一番近い有効な値で埋める (Nearest)
    if np.any(np.isnan(z_mapped)):
        nearest = NearestNDInterpolator(points, values)
        nan_mask = np.isnan(z_mapped)
        z_mapped[nan_mask] = nearest(norm_v2d[nan_mask])
        
    return z_mapped

def map_2d_mesh_to_3d_surface_with_normals(vertices_2d, base_mesh_data, canvas_size):
    """
    2Dメッシュ頂点に対応する「3D座標」と「法線」を補間して返す
    """
    if base_mesh_data is None or len(base_mesh_data) == 0:
        # データがない場合は平坦な面(Z=0, Normal=Z軸)を返す
        n = len(vertices_2d)
        return np.hstack([vertices_2d, np.zeros((n, 1))]), np.tile([0,0,1], (n, 1))

    w, h = canvas_size
    
    # 2D座標の正規化
    norm_v2d = vertices_2d.copy()
    norm_v2d[:, 0] /= w
    norm_v2d[:, 1] = 1.0 - (norm_v2d[:, 1] / h)

    # 補間データの準備
    # base_mesh_data: [u, v, x, y, z, nx, ny, nz]
    points = base_mesh_data[:, :2]     # UV
    attributes = base_mesh_data[:, 2:] # Pos(3) + Normal(3)
    
    # 線形補間
    interpolator = LinearNDInterpolator(points, attributes)
    interpolated = interpolator(norm_v2d)
    
    # 欠損値(NaN)の穴埋め（最近傍補間）
    if np.any(np.isnan(interpolated)):
        nearest = NearestNDInterpolator(points, attributes)
        nan_mask = np.isnan(interpolated[:, 0]) # XがNaNなら全部NaN
        interpolated[nan_mask] = nearest(norm_v2d[nan_mask])
        
    # データの分離
    positions = interpolated[:, 0:3]
    normals = interpolated[:, 3:6]
    
    # [重要] 補間によって法線の長さが変わってしまうため、再正規化(長さ1にする)
    norms_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norms_len[norms_len < 1e-6] = 1.0 # ゼロ除算防止
    normals /= norms_len
    
    return positions, normals