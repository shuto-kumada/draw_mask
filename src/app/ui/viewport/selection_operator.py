import numpy as np
import vtk
import pyvista as pv
import triangle as tr
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from app.core.geometry.polygon import clean_polygon

class SelectionOperator:

    @staticmethod
    def _adaptive_subdivide(mesh, level=2, min_area=None):
        """
        指定した面積(min_area)より大きいセルのみを再分割する。
        これにより、既に細かいメッシュは分割されず、粗い部分だけが滑らかになる。
        """
        if not mesh or mesh.n_cells == 0: return mesh
        # PolyData保証
        if not isinstance(mesh, pv.PolyData): mesh = mesh.extract_surface()
        
        # 三角形化 (必須)
        if not mesh.is_all_triangles:
            mesh = mesh.triangulate()
        
        # 面積計算
        sizes = mesh.compute_cell_sizes()
        area_array = sizes["Area"]
        
        # 閾値が指定されていない場合、バウンディングボックス対角線の0.5%相当の面積とする
        # (比較的粗めだが、変形には十分な密度)
        if min_area is None:
            min_area = (mesh.length * 0.005) ** 2

        # 大きいセルのマスク
        large_mask = area_array > min_area
        
        # 分割不要ならそのまま返す (ここが「もともと細かい場合」の対策)
        if not np.any(large_mask):
            # print("Mesh is already fine enough.")
            return mesh
            
        # 全て分割対象なら一括処理
        if np.all(large_mask):
             return mesh.subdivide(level, subfilter='linear')
        
        # 分離: 大きいセル(分割対象)と小さいセル(維持対象)
        large_cells = mesh.extract_cells(large_mask).extract_surface()
        small_cells = mesh.extract_cells(~large_mask).extract_surface()
        
        # 大きい方だけ再分割
        refined_large = large_cells.triangulate().subdivide(level, subfilter='linear')
        
        # 結合してクリーニング
        merged = (refined_large + small_cells).clean(tolerance=1e-5)
        
        return merged
    
    # === 共通フィルタ ===
    @staticmethod
    def _filter_facing_and_closest(mesh, camera_vector, center_point):
        """
        1. カメラ視線と逆向き（こちらを向いている）面だけを残す (Backface Culling)
        2. 残ったパーツのうち、基準点(center_point)に最も近い連結成分を抽出する
        """
        if mesh.n_points == 0: return None
        
        # ポリゴン化
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface()

        # 法線計算
        if mesh.point_normals is None:
            mesh.compute_normals(inplace=True, auto_orient_normals=True)

        normals = mesh.point_normals
        if normals is None:
             # point_normalsがない場合はcell_normalsで代用
             mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
             normals = mesh.cell_normals
             
        if normals is None: return None
        
        # カメラの視線ベクトル
        view_vec = np.array(camera_vector)
        norm = np.linalg.norm(view_vec)
        if norm > 0: view_vec = view_vec / norm
        
        # 内積計算 (dot < 0 なら対面している=表)
        dots = np.einsum('ij,j->i', normals, view_vec)
        mask = dots < 0.0 
        
        # 条件を満たすセルだけ抽出
        facing_mesh = mesh.extract_cells(mask).extract_surface()
        
        if facing_mesh.n_points == 0: return None

        # --- 連結成分分解と距離選別 ---
        facing_mesh.clean(inplace=True)
        bodies = facing_mesh.split_bodies()
        
        if bodies.n_blocks == 0: return None
        if bodies.n_blocks == 1: 
            body = bodies[0]
            if not isinstance(body, pv.PolyData): body = body.extract_surface()
            return body

        # 最も近いボディを選択
        closest_body = None
        min_dist = float('inf')
        ref_p = np.array(center_point)
        
        for body in bodies:
            body_center = np.array(body.center)
            dist = np.linalg.norm(body_center - ref_p)
            if dist < min_dist:
                min_dist = dist
                closest_body = body
        
        if closest_body and not isinstance(closest_body, pv.PolyData):
            closest_body = closest_body.extract_surface()
        
        if closest_body:
             try:
                closest_body.compute_normals(inplace=True, auto_orient_normals=True)
             except: pass
                
        return closest_body
    
    # === 切り取り (Traceモード) ===
    @staticmethod
    def extract_selection(mesh, trace_points_2d, plotter):
        """
        2Dトレース線を使ってメッシュを切り抜き、投影データを作成する
        """
        if not mesh or len(trace_points_2d) < 3:
            return None, None, None

        renderer = plotter.renderer
        ratio = plotter.devicePixelRatioF()
        win_h = plotter.height() * ratio
        w = plotter.width()
        h = plotter.height()

        # 1. 2Dスクリーン座標から3Dループを作成 (ニアプレーン上)
        loop_points_3d = vtk.vtkPoints()
        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToDisplay()

        # 重心計算用
        center_accum = np.array([0.0, 0.0, 0.0])
        count = 0

        for qt_x, qt_y in trace_points_2d:
            vtk_x = qt_x * ratio
            vtk_y = win_h - (qt_y * ratio)

            coordinate.SetValue(vtk_x, vtk_y, 0.0)
            world_pt = coordinate.GetComputedWorldValue(renderer)
            loop_points_3d.InsertNextPoint(world_pt)

            center_accum += np.array(world_pt)
            count += 1
        
        # スケッチの重心（＝カメラのすぐ手前にある点）
        center_point = center_accum / count if count > 0 else (0,0,0)

        # 2. 選択ループ定義 (カメラ視線方向へ伸びる筒)
        selection_loop = vtk.vtkImplicitSelectionLoop()
        selection_loop.SetLoop(loop_points_3d)
        
        camera = renderer.GetActiveCamera()
        view_normal = np.array(camera.GetDirectionOfProjection())
        selection_loop.SetNormal(view_normal)

        # 3. クリップ実行
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(mesh)
        clipper.SetClipFunction(selection_loop)
        clipper.InsideOutOff() # 内側を残す
        clipper.SetValue(0.0)
        clipper.Update()
        
        clipped = pv.wrap(clipper.GetOutput())
        extracted = SelectionOperator._filter_facing_and_closest(clipped, view_normal, center_point)

        # さらに念のため連結成分でノイズ除去 (一番大きい塊を残す)
        if extracted is None or extracted.n_points == 0:
             return None, None, None

        # 投影データ作成 (Canvas連携用)
        projected_data = [] 
        coord_world = vtk.vtkCoordinate()
        coord_world.SetCoordinateSystemToWorld()
        
        for p in extracted.points:
            coord_world.SetValue(p)
            x_pix, y_pix = coord_world.GetComputedDoubleDisplayValue(renderer)
            x_norm = x_pix / w
            y_norm = 1.0 - (y_pix / h)
            z_world = p[2] 
            projected_data.append([x_norm, y_norm, z_world])

        # 2D輪郭正規化
        normalized_2d = []
        for p in trace_points_2d:
            normalized_2d.append([p[0] / w, 1.0 - (p[1] / h)])
            
        return extracted, normalized_2d, np.array(projected_data, dtype=np.float32)
    
    # === ベース形状取得 (Fitモード) ===
    @staticmethod
    def get_base_shape_from_sketch(mesh, sketch_points, plotter, canvas_size):
        """
        2Dスケッチ領域に対応する3Dメッシュの表面形状データ(projected_data)を取得する
        """
        if not mesh or len(sketch_points) < 3:
            return None
        
        # 法線がない場合は計算しておく
        if mesh.point_normals is None:
            mesh.compute_normals(inplace=True, auto_orient_normals=True)

        renderer = plotter.renderer
        ratio = plotter.devicePixelRatioF()
        win_h = plotter.height() * ratio
        w = plotter.width()
        h = plotter.height()
        
        cw, ch = canvas_size
        # キャンバス座標をビューポート座標へ変換するスケール
        scale_x = w / cw
        scale_y = h / ch

        # ループ作成
        loop_points_3d = vtk.vtkPoints()
        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToDisplay()

        center_accum = np.array([0.0, 0.0, 0.0])
        count = 0

        for sx, sy in sketch_points:
            # Canvas -> Viewport Screen
            qt_x = sx * scale_x
            qt_y = sy * scale_y
            
            # Qt(Top-Left) -> VTK(Bottom-Left)
            vtk_x = qt_x * ratio
            vtk_y = win_h - (qt_y * ratio)

            coordinate.SetValue(vtk_x, vtk_y, 0.0)
            world_pt = coordinate.GetComputedWorldValue(renderer)
            loop_points_3d.InsertNextPoint(world_pt)

            center_accum += np.array(world_pt)
            count += 1
        
        center_point = center_accum / count if count > 0 else (0,0,0)

        # クリップ (型抜き)
        selection_loop = vtk.vtkImplicitSelectionLoop()
        selection_loop.SetLoop(loop_points_3d)
        camera = renderer.GetActiveCamera()
        view_normal = np.array(camera.GetDirectionOfProjection())
        selection_loop.SetNormal(view_normal)

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(mesh)
        clipper.SetClipFunction(selection_loop)
        clipper.InsideOutOff()
        clipper.SetValue(0.0)
        clipper.Update()

        clipped = pv.wrap(clipper.GetOutput())
        extracted = SelectionOperator._filter_facing_and_closest(clipped, view_normal, center_point)

        if extracted is None or extracted.n_points == 0: return None
        
        # === ここで確実にPolyDataにする ===
        if not isinstance(extracted, pv.PolyData):
            extracted = extracted.extract_surface()
        
        # 法線を再計算 (抽出過程で失われることがあるため)
        extracted.compute_normals(inplace=True, auto_orient_normals=True)

        # 投影データ作成 (Generatorへ渡す用)
        projected_data = [] 
        coord_world = vtk.vtkCoordinate()
        coord_world.SetCoordinateSystemToWorld()
        
        for i, p in enumerate(extracted.points):
            coord_world.SetValue(p)
            x_pix, y_pix = coord_world.GetComputedDoubleDisplayValue(renderer)
            
            x_norm = x_pix / w
            y_norm = 1.0 - (y_pix / h)
            
            # 法線取得
            normal = extracted.point_normals[i]
            
            # 全データを1行にまとめる
            projected_data.append([
                x_norm, y_norm,       # 0, 1: UV
                p[0], p[1], p[2],     # 2, 3, 4: Position
                normal[0], normal[1], normal[2] # 5, 6, 7: Normal
            ])
            
        return np.array(projected_data, dtype=np.float32)
    
    # === 表面切り抜き (ヘルパー) ===
    @staticmethod
    def create_surface_from_sketch(mesh, sketch_points, plotter, canvas_size):
        """
        2Dスケッチ座標から3Dメッシュを切り抜いて返す（高品位パッチ生成の下請け）
        """
        if not mesh or len(sketch_points) < 3: return None
        renderer = plotter.renderer
        ratio = plotter.devicePixelRatioF()
        win_h = plotter.height() * ratio
        w = plotter.width(); cw, ch = canvas_size
        scale_x = w / cw; scale_y = plotter.height() / ch

        loop_points_3d = vtk.vtkPoints()
        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToDisplay()

        center_accum = np.array([0.0, 0.0, 0.0])
        count = 0
        for sx, sy in sketch_points:
            qt_x = sx * scale_x; qt_y = sy * scale_y
            vtk_x = qt_x * ratio; vtk_y = win_h - (qt_y * ratio)
            coordinate.SetValue(vtk_x, vtk_y, 0.0)
            world_pt = coordinate.GetComputedWorldValue(renderer)
            loop_points_3d.InsertNextPoint(world_pt)
            center_accum += np.array(world_pt)
            count += 1
        center_point = center_accum / count if count > 0 else (0,0,0)

        selection_loop = vtk.vtkImplicitSelectionLoop()
        selection_loop.SetLoop(loop_points_3d)
        camera = renderer.GetActiveCamera()
        view_normal = np.array(camera.GetDirectionOfProjection())
        selection_loop.SetNormal(view_normal)

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(mesh)
        clipper.SetClipFunction(selection_loop)
        clipper.InsideOutOn()
        clipper.SetValue(0.0)
        clipper.Update()
        
        clipped = pv.wrap(clipper.GetOutput())
        extracted = SelectionOperator._filter_facing_and_closest(clipped, view_normal, center_point)
        return extracted

    # === 高品質パッチ生成 (Triangle生成 -> 投影 -> 再分割) ===
    @staticmethod
    def create_high_quality_patch(mesh, sketch_points, hole_strokes, plotter, canvas_size):
        """
        スケッチ輪郭内を2Dでメッシュ分割し、レイキャストでオブジェクト表面に投影。
        さらに再分割(Subdivide)して滑らかにする。
        """
        if not mesh or len(sketch_points) < 3:
            return None

        # 1. 2D輪郭の準備 (Shapelyでクリーニング)
        try:
            poly_geom = Polygon(sketch_points)
            if not poly_geom.is_valid: poly_geom = poly_geom.buffer(0)
            if poly_geom.is_empty: return None
            
            # 最大のポリゴンを採用
            if poly_geom.geom_type == 'MultiPolygon':
                poly_geom = max(poly_geom.geoms, key=lambda p: p.area)
                
            poly_pts = np.array(poly_geom.exterior.coords, dtype=np.float32)
            poly_pts = clean_polygon(poly_pts, min_edge_len=1.0)
            # 始点と終点の重複削除
            if len(poly_pts) > 0 and np.allclose(poly_pts[0], poly_pts[-1]):
                poly_pts = poly_pts[:-1]
        except:
            return None

        if len(poly_pts) < 3: return None

        # 2. Triangleで初期メッシュ生成 (少し粗めでOK、後でSubdivideする)
        n_pts = len(poly_pts)
        segments = np.vstack([np.arange(n_pts), (np.arange(n_pts) + 1) % n_pts]).T.tolist()

        vertices = poly_pts.tolist()
        hole_markers = []

        # Hole処理
        if hole_strokes:
            for h_stroke in hole_strokes:
                h_stroke = clean_polygon(np.array(h_stroke, dtype=np.float32))
                if len(h_stroke) < 3: continue
                
                # 穴マーカー計算
                try:
                    # Shapelyで代表点取得
                    h_poly = Polygon(h_stroke)
                    if not h_poly.is_valid: h_poly = h_poly.buffer(0)
                    if h_poly.is_empty: continue
                    rep_pt = h_poly.representative_point()
                    hole_markers.append([rep_pt.x, rep_pt.y])
                except: continue

                # 頂点・セグメント追加
                start_idx = len(vertices)
                if np.allclose(h_stroke[0], h_stroke[-1]):
                    pts = h_stroke[:-1]
                else:
                    pts = h_stroke

                n_h = len(pts)
                vertices.extend(pts.tolist())
                
                h_segs = np.vstack([
                    np.arange(start_idx, start_idx + n_h),
                    np.r_[np.arange(start_idx + 1, start_idx + n_h), start_idx]
                ]).T.tolist()
                segments.extend(h_segs)

        data = dict(vertices=poly_pts, segments=segments)
        if hole_markers:
            data['holes'] = hole_markers

        try:
            # pq20a200: 最小角20度、最大面積20px
            tri_data = tr.triangulate(data, 'pq20a200')
        except:
            try: 
                if hole_markers:
                    del data['holes']
                    tri_data = tr.triangulate(data, 'pq20a200')
                else:
                    tri_data = tr.triangulate(data, 'p')
            except: pass

        """
        if tri_data is None or 'vertices' not in tri_data or 'triangles' not in tri_data:
            print("Triangle mesh generation failed.")
            return None
        """
            
        vertices_2d = tri_data['vertices']
        faces_2d = tri_data['triangles']

        if len(faces_2d) == 0: return None

        # 3. 3D表面への投影 (Ray Cast)
        renderer = plotter.renderer
        ratio = plotter.devicePixelRatioF()
        win_h = plotter.height() * ratio
        
        cw, ch = canvas_size
        scale_x = plotter.width() / cw
        scale_y = plotter.height() / ch

        projected_points_3d = []
        valid_indices = []
        
        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToDisplay()
        camera_pos = np.array(renderer.GetActiveCamera().GetPosition())

        # ロケータ構築
        locator = vtk.vtkOBBTree()
        locator.SetDataSet(mesh)
        locator.BuildLocator()
        
        points_vtk = vtk.vtkPoints()

        for i, (sx, sy) in enumerate(vertices_2d):
            qt_x = sx * scale_x
            qt_y = sy * scale_y
            vtk_x = qt_x * ratio
            vtk_y = win_h - (qt_y * ratio)
            
            coordinate.SetValue(vtk_x, vtk_y, 0.0)
            p_near = np.array(coordinate.GetComputedWorldValue(renderer))
            
            coordinate.SetValue(vtk_x, vtk_y, 1.0)
            p_far = np.array(coordinate.GetComputedWorldValue(renderer))
            
            # レイキャスト
            code = locator.IntersectWithLine(p_near, p_far, points_vtk, None)
            
            if code != 0:
                # 最もカメラに近い交点を採用
                min_dist = float('inf')
                best_pt = None
                n_int = points_vtk.GetNumberOfPoints()
                for k in range(n_int):
                    pt = np.array(points_vtk.GetPoint(k))
                    dist = np.linalg.norm(pt - camera_pos)
                    if dist < min_dist:
                        min_dist = dist
                        best_pt = pt
                
                if best_pt is not None:
                    projected_points_3d.append(best_pt)
                    valid_indices.append(i)

        # 4. メッシュ構築と再分割
        if len(projected_points_3d) < 3: return None
            
        # インデックス再マッピング
        idx_map = {old: new for new, old in enumerate(valid_indices)}
        valid_faces = []
        for face in faces_2d:
            if face[0] in idx_map and face[1] in idx_map and face[2] in idx_map:
                new_face = [3, idx_map[face[0]], idx_map[face[1]], idx_map[face[2]]]
                valid_faces.extend(new_face)
        
        if not valid_faces: return None

        # 初期パッチ (PyVista)
        patch_mesh = pv.PolyData(np.array(projected_points_3d), valid_faces)
        
        # === 再分割と吸着 ===
        try:
            # 1. 細分化 (Linear分割)
            #subdivided = patch_mesh.subdivide(2, subfilter='linear')
            limit_area = (patch_mesh.length * 0.01) ** 2
            subdivided = SelectionOperator._adaptive_subdivide(patch_mesh, level=2, min_area=limit_area)
            
            # 2. 吸着 (Projection)
            # 増えた頂点は平面上にあるため、再度元のメッシュ表面に吸着させる
            # これには vtkCellLocator (FindClosestPoint) を使うのが高速かつ正確    
            surf_locator = vtk.vtkCellLocator()
            surf_locator.SetDataSet(mesh)
            surf_locator.BuildLocator()
            
            new_pts = []
            sub_pts = subdivided.points
            
            for p in sub_pts:
                closest = [0.0, 0.0, 0.0]
                cell_id = vtk.reference(0)
                sub_id = vtk.reference(0)
                dist = vtk.reference(0.0)
                surf_locator.FindClosestPoint(p, closest, cell_id, sub_id, dist)
                new_pts.append(closest)
                
            subdivided.points = np.array(new_pts)
            
            # 法線計算
            subdivided.compute_normals(inplace=True, auto_orient_normals=True)

            return subdivided       
        except Exception as e:
            print(f"Subdivision/Project failed: {e}")
            # 失敗したら初期パッチを返す
            patch_mesh.compute_normals(inplace=True, auto_orient_normals=True)

            return patch_mesh

    # === オブジェクト直接変形 ===
    @staticmethod
    def warp_mesh_by_strokes(mesh, sketch_points, fixed_strokes, params, plotter, canvas_size, do_subdivide=True):
        """
        スケッチのストロークに基づいて、メッシュの頂点を法線方向に移動させる
        """
        if not mesh or len(sketch_points) < 2:
            return None

        # 1. 2Dスケッチを3D表面上の点群に変換
        sketch_points = SelectionOperator._resample_points(sketch_points, interval=2.0)
        renderer = plotter.renderer
        ratio = plotter.devicePixelRatioF()
        win_h = plotter.height() * ratio
        w = plotter.width()
        h = plotter.height()
        cw, ch = canvas_size
        scale_x = w / cw
        scale_y = h / ch

        stroke_3d_points = []
        
        # ピック処理で表面の点を取得
        for sx, sy in sketch_points:
            qt_x = sx * scale_x
            qt_y = sy * scale_y
            vtk_x = qt_x * ratio
            vtk_y = win_h - (qt_y * ratio)
            
            plotter.picker.Pick(vtk_x, vtk_y, 0, renderer)
            pos = plotter.picker.GetPickPosition()
            stroke_3d_points.append(pos)

        if not stroke_3d_points: return None
        stroke_3d_points = np.array(stroke_3d_points)

        # 固定点ストロークの3D投影
        fixed_3d_tree = None
        if fixed_strokes:
            all_fixed_3d = []
            for f_stroke in fixed_strokes:
                f_stroke = SelectionOperator._resample_points(f_stroke, interval=2.0)
                for fx, fy in f_stroke:
                    qt_x = fx * scale_x; qt_y = fy * scale_y
                    vtk_x = qt_x * ratio; vtk_y = win_h - (qt_y * ratio)
                    plotter.picker.Pick(vtk_x, vtk_y, 0, renderer)
                    pos = plotter.picker.GetPickPosition()
                    # (0,0,0) 除外チェック入れても良い
                    all_fixed_3d.append(pos)
            if all_fixed_3d:
                fixed_3d_tree = cKDTree(all_fixed_3d)
        
        # ストロークの累積距離 (Arc Length) を計算
        segment_lens = np.linalg.norm(np.diff(stroke_3d_points, axis=0), axis=1)
        cum_dist = np.insert(np.cumsum(segment_lens), 0, 0)
        total_length = cum_dist[-1]

        # 交差点の検出
        intersect_dist_on_stroke = 0.0
        has_intersection = False

        if fixed_3d_tree:
            # ストローク上の点と固定点群との距離
            dists_to_fixed, _ = fixed_3d_tree.query(stroke_3d_points)
            min_idx = np.argmin(dists_to_fixed)
            
            # 近接判定 (モデルサイズの1%程度まで近づけば交差とみなす)
            threshold = mesh.length * 0.01 
            if dists_to_fixed[min_idx] < threshold:
                has_intersection = True
                intersect_dist_on_stroke = cum_dist[min_idx]

        # プロファイル計算用のパラメータ
        mag = params['magnitude'] * 0.01
        prof = params['profile']
        
        # ストローク上の各点における「変形の強さ(0.0~1.0)」を事前計算
        # これが "Generate Objectのような" スロープを作ります
        stroke_factors = np.zeros(len(stroke_3d_points))

        if has_intersection:
            # Case A: 固定点あり (交差点で0、離れるほど強くなる)
            dist_from_intersect = np.abs(cum_dist - intersect_dist_on_stroke)
            max_d = np.max(dist_from_intersect)
            if max_d < 1e-6: max_d = 1.0
            stroke_factors = dist_from_intersect / max_d
        else:
            # Case B: 固定点なし (中心で最大、端で0)
            center_dist = total_length / 2.0
            dist_from_center = np.abs(cum_dist - center_dist)
            max_d = total_length / 2.0
            if max_d < 1e-6: max_d = 1.0
            stroke_factors = 1.0 - (dist_from_center / max_d)
            
        stroke_factors = np.clip(stroke_factors, 0.0, 1.0)

        length = mesh.length
        inf_radius = (params['influence'] / 100.0) * (length * 0.2)
        if inf_radius < 1e-6: inf_radius = 1e-6

        # 2. メッシュ頂点の検索用ツリー作成
        # 変形対象のメッシュをコピー
        warped_mesh = mesh.copy()

        if do_subdivide:
            warped_mesh = SelectionOperator._local_subdivide_around_stroke(warped_mesh, stroke_3d_points, inf_radius)
            # Fit Object モード: 局所的な三角形化と再分割を行う
            if not warped_mesh.is_all_triangles:
                warped_mesh = warped_mesh.triangulate()
            
            limit_area = (warped_mesh.length * 0.01) ** 2
            #inf_radius = (params['influence'] / 100.0) * (warped_mesh.length * 0.2)
            warped_mesh = SelectionOperator._adaptive_subdivide(warped_mesh, level=1, min_area=limit_area)
        else:
            # Trace Fit モード: メッシュ構造（四角形など）を維持したまま頂点のみ動かす
            pass

        """
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(warped_mesh)
        cleaner.SetTolerance(0.005) # 0.5% の距離内なら結合
        cleaner.PointMergingOn()
        cleaner.Update()
        warped_mesh = pv.wrap(cleaner.GetOutput())
        """
        warped_mesh.clean(inplace=True)

        """
        try:
            warped_mesh = warped_mesh.subdivide(2, subfilter='linear')
        except Exception as e:
            print(f"Mesh subdivision failed: {e}")
        """
        
        # 法線計算
        if warped_mesh.point_normals is None:
            warped_mesh.compute_normals(inplace=True, auto_orient_normals=True)
            
        vertices = warped_mesh.points
        normals = warped_mesh.point_normals

        # カメラ位置と視線ベクトルを取得 (裏側変形防止)
        camera = renderer.GetActiveCamera()
        cam_pos = np.array(camera.GetPosition())
        
        # ストローク点群のKDTree
        stroke_tree = cKDTree(stroke_3d_points)
        mesh_tree = cKDTree(vertices)
        
        # パラメータ
        #mag = params['magnitude'] * 0.01 # スケール調整
        #prof = params['profile']
        # Influenceはピクセル単位ではなく「3D空間距離」として扱う必要がある
        # 簡易的にメッシュのバウンディングボックスからスケールを推定
        length = mesh.length # 対角線の長さ
        inf_radius = (params['influence'] / 100.0) * (length * 0.2) # 最大でモデルサイズの20%程度

        if inf_radius < 1e-6: inf_radius = 1e-6

        # 3. 影響範囲内の頂点を探して変形
        # 全頂点に対して「ストロークまでの最短距離」を計算するのは重いので、
        # ストロークの各点から半径インフルエンス内の頂点を探す
        
        # 影響を受ける頂点のインデックス集合
        # query_ball_point はリストのリストを返す
        indices_list = stroke_tree.query_ball_point(vertices, r=inf_radius)
        
        # 変形量の計算
        # 各頂点について、ストロークまでの距離 d を求め、変位 h を加算する
        # 重複して計算しないよう、一度距離マップを作るのが理想
        
        # 高速化: query_ball_pointの結果（インデックス）がある点だけ処理
        active_indices = [i for i, idxs in enumerate(indices_list) if idxs]
        
        if not active_indices:
            return warped_mesh
        
        # 法線チェック用の基準法線
        #mesh_tree = cKDTree(vertices)
        _, nearest_indices = mesh_tree.query(stroke_3d_points)
        stroke_normals = normals[nearest_indices]

        # 対象頂点の座標
        active_verts = vertices[active_indices]
        
        # ストロークまでの距離を計算
        dists, neighbor_indices = stroke_tree.query(active_verts)
        
        # 変位計算と適用
        for k, idx in enumerate(active_indices):
            d = dists[k]
            target_n = normals[idx]
            nearest_stroke_idx = neighbor_indices[k]
            
            # 2. 法線チェック
            # 変形対象の頂点法線 (target_n) と、ストロークの法線 (ref_n) の内積をとる
            ref_n = stroke_normals[neighbor_indices[k]]
            dot1 = np.dot(target_n, ref_n)

            to_cam = cam_pos - vertices[idx]
            dot2 = np.dot(target_n, to_cam)

            if dot1 < 0.0 or dot2 < 0.0: continue

            """
            # 固定点付近のマスク処理
            mask_factor = 1.0
            if fixed_3d_tree:
                # 頂点が固定点に近い場合、変形量を減らす
                f_dist, _ = fixed_3d_tree.query(vertices[idx])
                # 固定点から一定距離内は動かさない、あるいは減衰させる
                # ここでは influence radius の半分くらいで減衰させてみる
                fix_threshold = inf_radius * 0.5
                if f_dist < fix_threshold:
                    # 距離0で0, 閾値で1になるような係数
                    mask_factor = f_dist / fix_threshold
                    mask_factor = np.clip(mask_factor, 0.0, 1.0)
                    mask_factor = mask_factor ** 2 # 滑らかに
            
            t = d / inf_radius
            if t > 1.0: t = 1.0
            decay = (1.0 - t) ** prof if prof > 0 else 1.0
            displacement = mag * decay * mask_factor
            vertices[idx] += normals[idx] * displacement
            """
            
            # 1. 縦方向の係数 (Longitudinal Factor)
            # ストローク上の位置に応じた強さを取得 (これでスロープができる)
            longitudinal_factor = stroke_factors[nearest_stroke_idx]
            
            # 2. 横方向の係数 (Cross-sectional Decay)
            # ストローク中心から離れるほど弱くなる
            t_cross = d / inf_radius
            if t_cross > 1.0: t_cross = 1.0
            cross_decay = (1.0 - t_cross)
            
            # 3. プロファイルの適用 (Profile Curve)
            # Generate Objectでは h = mag * (t ^ prof)
            # ここでは 縦×横 の係数に対してプロファイルをかける
            combined_factor = longitudinal_factor * cross_decay
            
            final_factor = combined_factor ** prof if prof > 0 else 1.0
            
            # 最終変形量
            displacement = mag * final_factor
            
            vertices[idx] += normals[idx] * displacement

        # 4. メッシュ更新
        warped_mesh.points = vertices

        #"""
        try:
            warped_mesh.smooth_taubin(n_iter=50, pass_band=0.05, inplace=True)
        except: pass
        #"""

        """
        if active_indices:
            try:
                # (1) 変形に関わった頂点を含む領域だけを抽出
                # extract_points は指定した頂点を使用しているセルを抽出します
                # これにより「変形領域」+「その隣接セル(境界)」が取れます
                sub_mesh = warped_mesh.extract_points(active_indices, include_cells=True)
                
                # PolyDataであることを保証
                if not isinstance(sub_mesh, pv.PolyData):
                    sub_mesh = sub_mesh.extract_surface()
                
                # (2) 元の頂点IDを保持しているか確認 (PyVistaは通常 vtkOriginalPointIds を持ちます)
                if 'vtkOriginalPointIds' in sub_mesh.point_data:
                    orig_ids = sub_mesh.point_data['vtkOriginalPointIds']
                    
                    # (3) 抽出領域のみスムージング
                    # boundary_smoothing=False にすることで、
                    # 切り出したパッチの「縁（＝変形していない領域との境界）」を固定します。
                    sub_mesh.smooth_taubin(n_iter=50, pass_band=0.05, inplace=True, boundary_smoothing=False)
                    
                    # (4) 滑らかになった座標を元のメッシュに書き戻す
                    # 境界は固定されているため、元のメッシュと隙間なく繋がります
                    warped_mesh.points[orig_ids] = sub_mesh.points
                    
            except Exception as e:
                print(f"Local smoothing failed: {e}")
                pass
        """

        warped_mesh.compute_normals(inplace=True, feature_angle=0, auto_orient_normals=False)
        
        return warped_mesh
        
    # === 2D点群のリサンプリング (高密度化) ===
    @staticmethod
    def _resample_points(points, interval=2.0):
        """
        点と点の間隔を指定したピクセル数(interval)以下になるように補間して増やす。
        これにより、マウスを速く動かしても線が途切れなくなる。
        """
        if len(points) < 2: return np.array(points)
        
        points = np.array(points)
        new_points = [points[0]]
        
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            dist = np.linalg.norm(p2 - p1)
            
            if dist > interval:
                # 間隔が広い場合は埋める
                num_steps = int(dist / interval)
                vec = p2 - p1
                for j in range(1, num_steps + 1):
                    # 線形補間
                    new_pt = p1 + vec * (j / (num_steps + 1))
                    new_points.append(new_pt)
            
            new_points.append(p2)
            
        return np.array(new_points)
    
    # === 指定領域のメッシュ抽出 ===
    @staticmethod
    def extract_region_for_preview(mesh, outline_2d, plotter):
        """
        2D輪郭でメッシュを切り抜き、その部分のメッシュを返す（赤色表示用）。
        """
        if not mesh or len(outline_2d) < 3: return None

        renderer = plotter.renderer
        ratio = plotter.devicePixelRatioF()
        win_h = plotter.height() * ratio

        # 2D輪郭から3Dループ作成
        loop_points_3d = vtk.vtkPoints()
        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToDisplay()

        for x, y in outline_2d:
            vtk_x = x * ratio
            vtk_y = win_h - (y * ratio)
            coordinate.SetValue(vtk_x, vtk_y, 0.0)
            world_pt = coordinate.GetComputedWorldValue(renderer)
            loop_points_3d.InsertNextPoint(world_pt)
            
        # 選択ループ
        selection_loop = vtk.vtkImplicitSelectionLoop()
        selection_loop.SetLoop(loop_points_3d)
        
        camera = renderer.GetActiveCamera()
        view_normal = np.array(camera.GetDirectionOfProjection())
        selection_loop.SetNormal(view_normal)

        # クリップ実行
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(mesh)
        clipper.SetClipFunction(selection_loop)
        clipper.InsideOutOff() # 内側を残す
        clipper.Update()
        
        clipped = pv.wrap(clipper.GetOutput())
        if clipped.n_points == 0: return None
        
        # フィルタリング (手前側の表面のみ抽出)
        camera_pos = camera.GetPosition()

        # 法線判定
        clipped.compute_normals(cell_normals=True, point_normals=False, inplace=True)
        centers = clipped.cell_centers().points
        view_vecs = np.array(camera_pos) - centers
        dots = np.einsum('ij,ij->i', clipped.cell_normals, view_vecs)
        
        facing_mesh = clipped.extract_cells(dots > 0).extract_surface()
        if facing_mesh.n_points == 0: return None
        
        # 連結成分に分解して、2Dの重心に最も近いものを選択
        bodies = facing_mesh.split_bodies()
        if bodies.n_blocks == 0: return None
        
        # 2D重心（ワールド座標）
        loop_center = np.mean([loop_points_3d.GetPoint(i) for i in range(loop_points_3d.GetNumberOfPoints())], axis=0)
        
        best_body = bodies[0]
        min_dist = float('inf')
        for body in bodies:
            d = np.linalg.norm(np.array(body.center) - loop_center)
            if d < min_dist:
                min_dist = d
                best_body = body
                
        return best_body
    
    @staticmethod
    def subdivide_region(main_mesh, patch_mesh):
        """
        指定された領域（patch_mesh）とその周辺のメッシュを再分割し、
        造形用の高い解像度を持たせる。
        """
        if not main_mesh or not patch_mesh:
            return main_mesh

        print("Subdividing selected region...")
        
        tree = cKDTree(patch_mesh.points)
        dists, _ = tree.query(main_mesh.points)
        
        # 非常に近い距離にある頂点を持つセルを「パッチ領域」とする
        # (閾値はメッシュの平均エッジ長程度)
        threshold = patch_mesh.length * 0.15
        inside_mask = dists < threshold

        # PolyDataであることを保証
        if not isinstance(main_mesh, pv.PolyData):
            main_mesh = main_mesh.extract_surface()
        
        # 頂点マスクをセルマスクに変換
        inside_cells = []
        for i in range(main_mesh.n_cells):
            v_ids = main_mesh.get_cell(i).point_ids
            if any(inside_mask[v_id] for v_id in v_ids):
                inside_cells.append(i)
        
        if not inside_cells: 
            print("No cells found in range.")
            return main_mesh

        # 3. 指定範囲だけを抽出して処理
        target_part = main_mesh.extract_cells(inside_cells)
        outside_part = main_mesh.extract_cells([i for i in range(main_mesh.n_cells) if i not in inside_cells])

        # === 指定範囲のみ三角形化して再分割 ===
        if not isinstance(target_part, pv.PolyData): target_part = target_part.extract_surface()
        #refined_target = target_part.triangulate().subdivide(2, subfilter='linear')
        
        # 閾値: オブジェクト全体の 0.2% 程度 (局所なので少し細かく)
        limit_area = (main_mesh.length * 0.005) ** 2
        refined_target = SelectionOperator._adaptive_subdivide(target_part, level=2, min_area=limit_area)

        # 4. マージ(再結合)
        if not isinstance(outside_part, pv.PolyData): outside_part = outside_part.extract_surface()
        merged = outside_part + refined_target
        final_mesh = merged.clean(tolerance=1e-4)

        if not isinstance(final_mesh, pv.PolyData):
            final_mesh = final_mesh.extract_surface()

        # === スムージングで滑らかにする ===
        # 再分割によってカクカクした面が目立つのを防ぐため、軽くスムージングをかける
        # (形状が変わりすぎないよう、反復回数は少なめに)
        try:
            final_mesh.smooth_taubin(n_iter=50, pass_band=0.1, inplace=True)
        except:
            pass
        
        final_mesh.compute_normals(inplace=True, feature_angle=0, auto_orient_normals=False)
        
        print(f"Local Refinement complete: {main_mesh.n_cells} -> {final_mesh.n_cells} faces")
        return final_mesh
    
    @staticmethod
    def _local_subdivide_around_stroke(mesh, stroke_points, influence_radius):
        """
        ストローク周辺のメッシュ密度が足りない場合、局所的に再分割を行う。
        (ROI/BoundingBoxを使って確実にセルを捉える改良版)
        """
        if not mesh or mesh.n_cells == 0:
            return mesh

        # 1. 分割対象のセルを特定する (Bounding Box検索)
        # 影響範囲(ROI)に含まれるセルを広く取る
        margin = influence_radius * 2.0
        p_min = np.min(stroke_points, axis=0) - margin
        p_max = np.max(stroke_points, axis=0) + margin
        
        # セル中心がBOX内にあるものを高速フィルタリング
        centers = mesh.cell_centers().points
        mask = (
            (centers[:,0] >= p_min[0]) & (centers[:,0] <= p_max[0]) &
            (centers[:,1] >= p_min[1]) & (centers[:,1] <= p_max[1]) &
            (centers[:,2] >= p_min[2]) & (centers[:,2] <= p_max[2])
        )
        target_ids = np.where(mask)[0]
        
        # もしBOX検索で取れなかった場合(ポリゴンが巨大すぎる場合)への保険:
        # ストローク点に最も近いセルを追加
        if len(target_ids) == 0:
            try:
                closest_ids = np.unique(mesh.find_closest_cell(stroke_points))
                target_ids = np.union1d(target_ids, closest_ids)
            except: pass

        if len(target_ids) == 0: return mesh

        # 2. 現在の密度を確認
        check_region = mesh.extract_cells(target_ids)
        if check_region.n_points == 0: return mesh

        sizes = check_region.compute_cell_sizes()
        avg_area = np.mean(sizes['Area'])
        current_spacing = np.sqrt(avg_area) # 簡易的なエッジ長推定
        
        target_spacing = influence_radius * 0.3 # 影響半径の30%以下の細かさを目指す
        if target_spacing <= 1e-6: target_spacing = 1e-6
        
        if current_spacing <= target_spacing:
            return mesh
            
        # 必要な分割レベルを計算
        ratio = current_spacing / target_spacing
        level = int(np.ceil(np.log2(ratio)))
        level = np.clip(level, 1, 3) # 最大3回まで
        
        # 3. 再分割実行
        try:
            # 分割対象とそれ以外を分ける
            all_ids = np.arange(mesh.n_cells)
            rest_ids = np.setdiff1d(all_ids, target_ids)
            
            selection = mesh.extract_cells(target_ids).extract_surface()
            rest = mesh.extract_cells(rest_ids).extract_surface()
            
            # Linear分割 (形状を変えずに頂点を増やす)
            refined = selection.triangulate().subdivide(level, subfilter='linear')
            
            # 結合 (Cleanで頂点をマージ)
            merged = (rest + refined).clean(tolerance=mesh.length * 0.0001)
            merged.compute_normals(inplace=True, auto_orient_normals=True)
            return merged
            
        except Exception as e:
            print(f"Local subdivision failed: {e}")
            return mesh