import numpy as np
import triangle as tr
import pyvista as pv
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from app.core.geometry.polygon import clean_polygon, resample_stroke
from app.core.geometry.topology import extract_largest_polygon, snap_lines
from app.core.geometry.projection import apply_base_mesh_shape, map_2d_mesh_to_3d_surface_with_normals
from app.core.modeling.solver import compute_parametric_heights, solve_laplace

class MeshGenerator:
    def __init__(self):
        pass

    def robust_mesh_generation_unified(self, boundary, strokes_dict, hole_markers=None, fine_mesh=False):
        """
        全ストロークをまとめて交差解決し、単一のセグメントリストとしてTriangleに渡す
        """
        # 1. 全ストロークをLineStringとして収集
        all_lines = []
        
        poly = np.asarray(boundary, dtype=np.float32)
        poly = clean_polygon(poly)
        if len(poly) >= 3:
            all_lines.append(LineString(poly))

        for key, lines in strokes_dict.items():
            """
            if key != 'hole':
                continue
            """

            for line in lines:
                line = clean_polygon(line)
                if len(line) >= 2:
                    all_lines.append(LineString(line))

        if not all_lines: raise RuntimeError("有効なストロークがありません。")

        all_lines = snap_lines(all_lines, tolerance=15.0)

        # 2. 交差解決
        try:
            merged_geom = unary_union(all_lines)
        except Exception as e:
            raise RuntimeError(f"交差解決エラー: {e}")

        # 3. セグメント化
        final_segments_geom = []
        if merged_geom.geom_type == 'LineString':
            final_segments_geom.append(merged_geom)
        elif merged_geom.geom_type == 'MultiLineString':
            final_segments_geom.extend(merged_geom.geoms)
        elif merged_geom.geom_type == 'GeometryCollection':
            for g in merged_geom.geoms:
                if g.geom_type in ['LineString', 'LinearRing']:
                    final_segments_geom.append(g)

        # 4. 頂点とセグメント配列の作成
        vertex_map = {} 
        vertices = []
        segments = []

        def get_v_idx(pt):
            key = (round(pt[0], 4), round(pt[1], 4))
            if key not in vertex_map:
                vertex_map[key] = len(vertices)
                vertices.append([pt[0], pt[1]])
            return vertex_map[key]

        for geom in final_segments_geom:
            coords = list(geom.coords)
            if len(coords) < 2: continue
            for i in range(len(coords) - 1):
                idx1 = get_v_idx(coords[i])
                idx2 = get_v_idx(coords[i+1])
                if idx1 != idx2:
                    segments.append([idx1, idx2])

        if not vertices or not segments:
            raise RuntimeError("メッシュ生成用のセグメントが生成できませんでした。")

        data = {
            'vertices': np.array(vertices, dtype=np.float32),
            'segments': np.array(segments, dtype=int)
        }
        if hole_markers:
            data['holes'] = hole_markers

        # 5. メッシュ生成 (リトライロジック)
        options_to_try = ['pq30a500D', 'pq10a500D', 'pa500D', 'pD']
        mesh = None
        errors = []
        for opts in options_to_try:
            try:
                mesh = tr.triangulate(data, opts)
                if 'triangles' in mesh and len(mesh['triangles']) > 0: break
            except Exception as e: errors.append(f"{opts}: {e}"); continue

        if mesh is None:
            raise RuntimeError(f"メッシュ生成失敗:\n" + "\n".join(errors))
            
        return np.asarray(mesh['vertices']), np.asarray(mesh['triangles'], dtype=int)
    
    def generate(self, sketch_data, base_mesh_data=None, canvas_size=(1200, 900), is_fitting=False):
        strokes = sketch_data.get_strokes()
        
        # 境界線処理 (最大ポリゴン抽出)
        boundary_fixed = strokes.get('boundary_fixed', [])
        boundary_free = strokes.get('boundary_free', [])
        boundary_pts = boundary_fixed + boundary_free
        
        if not boundary_pts: raise RuntimeError("Boundary not found")
        boundary = extract_largest_polygon(boundary_pts)
        if boundary is None: raise RuntimeError("Invalid Boundary")

        # 2. Holeマーカー
        hole_markers = []
        if strokes['hole']:
            for s in strokes['hole']:
                s = clean_polygon(s)
                if len(s) < 3: continue
                try:
                    if not np.allclose(s[0], s[-1]): s = np.vstack((s, s[0]))
                    poly_geom = Polygon(s)
                    if not poly_geom.is_valid: poly_geom = poly_geom.buffer(0)
                    if not poly_geom.is_empty:
                        rep_pt = poly_geom.representative_point()
                        hole_markers.append([rep_pt.x, rep_pt.y])
                except:
                    hole_markers.append(np.mean(s[:-1], axis=0).tolist())

        # 3. ストローク収集 & リサンプリング
        mesh_strokes = {}
        # Fixed Point
        if 'fixed_point' in strokes and strokes['fixed_point']:
            mesh_strokes['fixed_point'] = [resample_stroke(clean_polygon(s), 10.0) for s in strokes['fixed_point']]
        
        # Deformation
        if strokes['deformation']:
            def_list = []
            for item in strokes['deformation']:
                if isinstance(item, dict) and 'points' in item: pts = item['points']
                else: pts = item
                def_list.append(resample_stroke(clean_polygon(pts), 10.0))
            mesh_strokes['deformation'] = def_list
            
        # Hole (壁として登録)
        if strokes['hole']:
            mesh_strokes['hole'] = [clean_polygon(s) for s in strokes['hole']]

        # 4. メッシュ生成
        vertices_2d, triangles = self.robust_mesh_generation_unified(
            boundary, mesh_strokes, hole_markers=hole_markers, fine_mesh=is_fitting
        )
        num_vertices = len(vertices_2d)

        # === ベース形状計算 (Noneチェック追加) ===
        base_positions = None
        base_normals = None
        z_base_scalar = None

        if is_fitting:
            base_positions, base_normals = map_2d_mesh_to_3d_surface_with_normals(
                vertices_2d, base_mesh_data, canvas_size
            )
        else:
            z_base_scalar = apply_base_mesh_shape(vertices_2d, base_mesh_data, canvas_size)
            
            # 安全対策: 万が一 None が返ってきたら 0埋め
            if z_base_scalar is None:
                z_base_scalar = np.zeros(num_vertices)

        # 6. 変形量計算
        tree = cKDTree(vertices_2d)
        fixed_indices = []
        if 'fixed_point' in mesh_strokes:
            for s in mesh_strokes['fixed_point']:
                _, idx = tree.query(s); fixed_indices.extend(np.unique(idx))
        fixed_indices = np.unique(fixed_indices)

        _, b_idxs = tree.query(boundary)
        boundary_indices = np.unique(b_idxs)
        
        fixed_deltas, fixed_set = compute_parametric_heights(
            num_vertices, vertices_2d,
            mesh_strokes.get('fixed_point', []),
            strokes['deformation'], tree
        )

        # 準備: 変形ストロークのLineStringリスト
        def_lines = []
        if 'deformation' in mesh_strokes:
            for s in mesh_strokes['deformation']:
                if len(s) >= 2: def_lines.append(LineString(s))

        # 各固定点ストロークについて判定
        if 'fixed_point' in mesh_strokes:
            for s in mesh_strokes['fixed_point']:
                if len(s) < 2: continue
                fixed_line = LineString(s)
                
                # 交差判定 (intersects)
                is_intersecting = False
                for d_line in def_lines:
                    if fixed_line.intersects(d_line):
                        is_intersecting = True
                        break
                
                # [判定] 交差していない場合 -> 全点を固定
                if not is_intersecting:
                    _, idxs = tree.query(s)
                    for idx in np.unique(idxs):
                        fixed_deltas[idx] = 0.0
                        fixed_set.add(idx)

        # === 輪郭の固定処理 (Fixed / Free) ===
        _, b_idxs = tree.query(boundary)
        boundary_indices = np.unique(b_idxs)

        # 固定輪郭 (Boundary Fixed) の頂点を特定して固定
        if boundary_fixed:
            # 固定輪郭ストロークをまとめる
            fixed_stroke_pts = np.vstack(boundary_fixed)
            if len(fixed_stroke_pts) > 0:
                fixed_stroke_tree = cKDTree(fixed_stroke_pts)
                # 輪郭頂点のうち、固定ストロークに近いものを抽出
                dists, _ = fixed_stroke_tree.query(vertices_2d[boundary_indices])
                
                # 近い点(2.0px以内)を固定セットに追加
                fixed_boundary_indices = boundary_indices[dists < 2.0]
                for idx in fixed_boundary_indices:
                    if idx not in fixed_set:
                        fixed_deltas[idx] = 0.0
                        fixed_set.add(idx)

        # 7. ソルバー実行
        delta_scalar = solve_laplace(num_vertices, triangles, fixed_deltas, fixed_set)
        delta_scalar = np.nan_to_num(delta_scalar, nan=0.0)

        # === 最終座標の決定 ===
        if is_fitting:
            displacement = base_normals * delta_scalar.reshape(-1, 1)
            final_vertices = base_positions + displacement
        else:
            # 通常モード
            # ここで z_base_scalar が None だとエラーになるが、上記対策で回避
            final_z = z_base_scalar + delta_scalar
            final_vertices = np.hstack((vertices_2d, final_z.reshape(-1, 1)))
            final_vertices[:, 0] *= -1 # 左右反転

        # メッシュ作成
        faces = np.hstack((np.full((len(triangles), 1), 3), triangles))
        faces = faces.flatten()
        mesh = pv.PolyData(final_vertices, faces)

        # 頂点や面がない場合は処理をスキップ
        if mesh.n_points == 0 or mesh.n_cells == 0:
            return mesh

        # 重複頂点のマージ (裂け防止)
        #mesh.clean(inplace=True)

        if not is_fitting:
            mesh.flip_faces(inplace=True)
            
        try:
            mesh.compute_normals(inplace=True, consistent_normals=True, auto_orient_normals=True)
            mesh.smooth(n_iter=100, relaxation_factor=0.1, inplace=True)
        except Exception as e:
            print(f"Post-processing warning: {e}")
            
        return mesh