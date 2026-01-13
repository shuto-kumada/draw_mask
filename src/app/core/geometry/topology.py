import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union, linemerge, polygonize
from app.core.geometry.polygon import clean_polygon

def extract_largest_polygon(stroke_list):
    """
    複数のストロークから形成される「閉じた領域」を検出し、
    最大面積のものを輪郭として返す。
    """
    if not stroke_list: return None

    # 1. ShapelyのLineStringに変換
    lines = []
    for s in stroke_list:
        if len(s) >= 2: lines.append(LineString(s))
    if not lines: return None

    lines = snap_lines(lines)

    # 2. 交差部分で線を分割（ノード化）
    try:
        noded = unary_union(lines)
        polygons = list(polygonize(noded))
    except:
        return None

    # ポリゴンが見つかった場合（閉じた領域がある場合）
    if polygons:
        # 面積が最大のものを採用
        best_poly = max(polygons, key=lambda p: p.area)
        return np.array(best_poly.exterior.coords, dtype=np.float32)
        
    # ポリゴンが見つからない場合（線が閉じていない場合）
    # 従来の「無理やり繋ぐ」方式にフォールバック
    return unify_boundary_strokes_fallback(stroke_list)

def unify_boundary_strokes_fallback(stroke_list):
    # 1本しかない場合はそれを返す
    if len(stroke_list) == 1:
        return stroke_list[0]

    # 処理用リスト（コピーを作成）
    segments = [s for s in stroke_list if len(s) > 1]
    if not segments: return None

    # 結果の点群：最初のセグメントから開始
    merged_points = segments.pop(0).tolist()

    # 全てのセグメントが尽きるまで繋ぎ合わせる
    while segments:
        # 現在の末尾
        current_tail = np.array(merged_points[-1])
            
        best_idx = -1
        min_dist = float('inf')
        should_reverse = False

        # 残りのセグメントの中から、現在の末尾に一番近い始点（または終点）を探す
        for i, seg in enumerate(segments):
            head = seg[0]
            tail = seg[-1]

             # 始点との距離
            d_head = np.linalg.norm(head - current_tail)
            if d_head < min_dist:
                min_dist = d_head
                best_idx = i
                should_reverse = False
                
            # 終点との距離（逆向きに繋ぐ方が近い場合）
            d_tail = np.linalg.norm(tail - current_tail)
            if d_tail < min_dist:
                min_dist = d_tail
                best_idx = i
                should_reverse = True
            
        # 最も近いセグメントを採用
        if best_idx != -1:
            next_seg = segments.pop(best_idx)
            if should_reverse:
                next_seg = next_seg[::-1] # 逆順にする
                
            # 結合（最初の点は重複するのでスキップしても良いが、念のため繋ぐ）
            merged_points.extend(next_seg.tolist())
        else:
            # 繋げられるものがない（離れすぎている）場合
            # とりあえず次のものを強制的に追加して線を引く
            merged_points.extend(segments.pop(0).tolist())

    # Numpy配列化
    merged = np.array(merged_points, dtype=np.float32)

    if np.linalg.norm(merged[0] - merged[-1]) > 1e-4:
        merged = np.vstack((merged, merged[0]))

    return merged

def resolve_all_intersections(strokes_dict):
    """
    全ての種類のストロークをまとめてShapelyに渡し、交差点で分割（ノード化）する。
    分割後、元の属性（Ridge, Valleyなど）を復元して返す。
    """
    # 1. 全ストロークを LineString 化し、元の属性を紐付ける
    # structure: [ (LineString, 'ridge'), (LineString, 'valley'), ... ]
    all_lines_with_type = []
    
    for t, stroke_list in strokes_dict.items():
        for s in stroke_list:
            if len(s) >= 2:
                all_lines_with_type.append((LineString(s), t))
        
    if not all_lines_with_type:
        return {}

    # 2. 全ラインをまとめて unary_union (これで交差点が頂点として分割される)
    raw_lines = [item[0] for item in all_lines_with_type]
    try:
        noded_geom = unary_union(raw_lines)
    except Exception as e:
        print(f"Union failed: {e}")
        return strokes_dict # 失敗したらそのまま返す

    # 分割後の線分リストを取得
    split_lines = []
    if noded_geom.geom_type == 'LineString':
        split_lines.append(noded_geom)
    elif noded_geom.geom_type == 'MultiLineString':
        split_lines.extend(noded_geom.geoms)
    elif noded_geom.geom_type == 'GeometryCollection':
        for g in noded_geom.geoms:
            if g.geom_type in ['LineString', 'LinearRing']:
                split_lines.append(g)

    # 3. 分割された各線分に、元の属性を割り当てる
    # 方法: 線分の中点が、元のどの線の近くにあったかで判定
    resolved_strokes = {k: [] for k in strokes_dict.keys()}
        
    for split_line in split_lines:
        # 線分の中点
        mid_pt = split_line.interpolate(0.5, normalized=True)
            
        best_type = None
        min_dist = 1.0 # 判定閾値 (ピクセル)
            
        # 元の線すべてと距離比較 (少し重いが確実)
        # 高速化のためにバッファを使う手もあるが、本数が少なければ距離で十分
        for orig_line, t in all_lines_with_type:
            dist = orig_line.distance(mid_pt)
            if dist < min_dist:
                min_dist = dist
                best_type = t
            
        # 属性が見つかればリストに追加
        if best_type:
            pts = np.array(split_line.coords, dtype=np.float32)
            resolved_strokes[best_type].append(pts)
        else:
            # 該当なしの場合（通常ありえないが）、デフォルトでboundaryかcontourへ
            pts = np.array(split_line.coords, dtype=np.float32)
            if 'contour' in resolved_strokes: resolved_strokes['contour'].append(pts)

    return resolved_strokes

def snap_lines(lines, tolerance=15.0):
    """
    ShapelyのLineStringリストを受け取り、端点が近いもの同士を吸着させて返す
    """
    if not lines: return []
    
    # 座標リストに変換して操作
    lines_coords = [list(line.coords) for line in lines]
    any_snapped = True
    iteration = 0
    
    # 反復的にスナップ（くっついたことでさらに近くの点とくっつく場合があるため）
    while any_snapped and iteration < 3:
        any_snapped = False
        iteration += 1
        
        # 比較用に一時的なLineStringを作成
        current_shapely_lines = [LineString(coords) for coords in lines_coords if len(coords) >= 2]
        if not current_shapely_lines: break

        for i in range(len(lines_coords)):
            stroke_points = lines_coords[i]
            if len(stroke_points) < 2: continue
            
            # 始点と終点のみチェック
            indices_to_check = [0, -1]
            
            for pt_idx in indices_to_check:
                current_pt_tuple = stroke_points[pt_idx]
                current_pt = Point(current_pt_tuple)
                
                best_candidate_pt = None
                min_dist = tolerance

                for j, target_line in enumerate(current_shapely_lines):
                    if i == j: continue # 自分自身とは比較しない
                    
                    dist = target_line.distance(current_pt)
                    if dist < min_dist:
                        # 線上の最も近い点を計算 (project -> interpolate)
                        proj_dist = target_line.project(current_pt)
                        nearest_pt = target_line.interpolate(proj_dist)
                        min_dist = dist
                        best_candidate_pt = (nearest_pt.x, nearest_pt.y)
                
                if best_candidate_pt:
                    # 座標更新
                    if stroke_points[pt_idx] != best_candidate_pt:
                        stroke_points[pt_idx] = best_candidate_pt
                        any_snapped = True

    # LineStringに戻して返す
    snapped_lines = []
    for coords in lines_coords:
        if len(coords) >= 2:
            snapped_lines.append(LineString(coords))
            
    return snapped_lines