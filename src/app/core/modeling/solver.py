import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

def compute_parametric_heights(n, verts, fixed_point_strokes, orig_defs, tree):
    """
    変形ストロークごとに「固定点との交差点」を探し、
    そこを基準(高さ0)としてストロークに沿った距離で高さを決定する。
    """
    fixed_h = {}
    fixed_s = set()

    # 固定点ストロークの全頂点をまとめたKDTreeを作成
    all_fixed_points = []
    if fixed_point_strokes:
        for s in fixed_point_strokes:
            all_fixed_points.extend(s)
    
    fixed_tree = None
    if all_fixed_points:
        fixed_tree = cKDTree(all_fixed_points)

    # 各Deformationストロークについて個別に処理
    for item in orig_defs:
        if not isinstance(item, dict) or 'points' not in item:
            continue
            
        pts = item['points'] # ストロークの点群 (順序あり)
        pm = item['params']
        mag = pm['magnitude']
        prof = pm['profile']

        inf_val = pm.get('influence', 50.0)
        radius = max(1.0, inf_val * 1.0)
        
        if len(pts) < 2: continue

        # ストロークに対応するメッシュ頂点のインデックスを取得
        #_, v_idxs = tree.query(pts)
        
        # --- ストロークに沿った累積距離 (Arc Length) を計算 ---
        # これにより、メッシュ形状に関係なく「描いた線の長さ」で制御できる
        segment_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum_dist = np.insert(np.cumsum(segment_lens), 0, 0)
        total_length = cum_dist[-1]
        
        # --- 基準点（距離0地点）の決定 ---
        intersect_dist_on_stroke = 0.0 # デフォルトは始点(0.0)
        has_intersection = False

        if fixed_tree:
            # ストローク上の各点と、固定点群との距離を計算
            dists_to_fixed, _ = fixed_tree.query(pts)
            
            # 最も近い点が「交差（近接）」しているか判定 (閾値: 10.0px)
            min_idx = np.argmin(dists_to_fixed)
            """
            min_val = dists_to_fixed[min_idx]
            
            if min_val < 10.0:
                has_intersection = True
                # 交差点での累積距離を基準にする
                intersect_dist_on_stroke = cum_dist[min_idx]
            """
            
            if dists_to_fixed[min_idx] < 10.0:
                has_intersection = True
                intersect_dist_on_stroke = cum_dist[min_idx]
        
        stroke_factors = np.zeros(len(pts))
        
        # --- 変形強度 t (0.0 ~ 1.0) の計算 ---
        #t_values = np.zeros(len(pts))

        if has_intersection:
            # === Case A: 固定点あり ===
            # 交差点からの距離が遠いほど強く変形 (0 -> 1)
            dist_from_intersect = np.abs(cum_dist - intersect_dist_on_stroke)
            max_d = np.max(dist_from_intersect)
            max_d = max(max_d, 1e-4)
            stroke_factors = dist_from_intersect / max_d
            """
            if max_d < 1e-4: max_d = 1.0
            
            t_values = dist_from_intersect / max_d
            """
            
        else:
            # === Case B: 固定点なし ===
            # ストロークの中心に近いほど強く変形 (1 -> 0)
            center_dist = total_length / 2.0
            dist_from_center = np.abs(cum_dist - center_dist)
            
            # 中心から端までの距離
            max_d = total_length / 2.0
            max_d = max(max_d, 1e-4)
            stroke_factors = 1.0 - (dist_from_center / max_d)

            """
            if max_d < 1e-4: max_d = 1.0
            
            # 中心で t=1.0, 端で t=0.0 になるように計算
            t_values = 1.0 - (dist_from_center / max_d)
            """
        
        # 値を 0~1 にクリップ
        #t_values = np.clip(t_values, 0.0, 1.0)
        stroke_factors = np.clip(stroke_factors, 0.0, 1.0)

        indices_list = tree.query_ball_point(pts, r=radius)
        temp_heights = {}
        
        """
        # --- 高さの適用 ---
        for i, v_idx in enumerate(v_idxs):
            t = t_values[i]
            
            # 高さ計算式: h = Magnitude * (t ^ Profile)
            h = mag if prof == 0.0 else mag * (t ** prof)
            
            # 重複時の優先処理 (絶対値が大きい方を採用)
            if v_idx in fixed_h:
                if abs(h) > abs(fixed_h[v_idx]):
                    fixed_h[v_idx] = h
            else:
                fixed_h[v_idx] = h
                fixed_s.add(v_idx)
        """

        for i, pt_idx_list in enumerate(indices_list):
            if not pt_idx_list: continue
            
            # ストローク上の点 P_i
            p_i = pts[i]
            # その点の縦方向係数 (Profile用)
            longitudinal_factor = stroke_factors[i]
            # Profile適用 (縦方向の勾配)
            longitudinal_val = longitudinal_factor ** prof if prof > 0 else (1.0 if longitudinal_factor > 0 else 0)

            # 影響範囲内の各頂点について
            for v_idx in pt_idx_list:
                # 頂点座標 V
                v_pos = verts[v_idx]
                
                # 横方向の距離 d (Cross-sectional)
                d = np.linalg.norm(v_pos - p_i)
                
                # 横方向の減衰 (中心=1, 端=0)
                # 単純な線形減衰: 1 - (d / radius)
                if d > radius: continue
                cross_factor = 1.0 - (d / radius)
                
                # 最終的な高さ係数 = 縦係数 * 横係数
                combined_factor = longitudinal_val * cross_factor
                
                # 高さ計算
                h = mag * combined_factor
                
                # 同じ頂点が複数のストローク点から影響を受ける場合、絶対値が大きい方を採用
                if v_idx in temp_heights:
                    if abs(h) > abs(temp_heights[v_idx]):
                        temp_heights[v_idx] = h
                else:
                    temp_heights[v_idx] = h

        # 結果を fixed_h に統合
        for v_idx, h in temp_heights.items():
            if v_idx in fixed_h:
                if abs(h) > abs(fixed_h[v_idx]):
                    fixed_h[v_idx] = h
            else:
                fixed_h[v_idx] = h
                fixed_s.add(v_idx)
                
    return fixed_h, fixed_s

def solve_laplace(n, tris, fixed_h, fixed_s):
    """Dirichlet境界条件付きラプラス方程式ソルバー"""
    A = lil_matrix((n, n))
    b = np.zeros(n)
    adj = [set() for _ in range(n)]

    for t in tris:
        for i in range(3): 
            adj[t[i]].update([t[(i+1)%3], t[(i+2)%3]])
    
    for i in range(n):
        if i in fixed_s:
            A[i, i] = 1.0
            b[i] = fixed_h[i]
        else:
            deg = len(adj[i])
            A[i, i] = deg
            for nb in adj[i]:
                A[i, nb] = -1.0
            b[i] = 0.0

    return spsolve(A.tocsc(), b)