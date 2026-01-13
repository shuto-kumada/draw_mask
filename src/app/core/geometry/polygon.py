import numpy as np

def resample_stroke(pts, interval=2.0):
    """
    ストロークの点密度を高める（リサンプリング）
    interval: 点と点の間隔（ピクセル）。小さいほど高密度になり変形が強くなる。
    """
    if len(pts) < 2:
        return pts
            
    new_pts = [pts[0]]
    for i in range(len(pts) - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        dist = np.linalg.norm(p2 - p1)
            
        # 間隔が広い場合、点を挿入する
        if dist > interval:
            num_steps = int(dist / interval)
            vec = p2 - p1
            for j in range(1, num_steps + 1):
                new_pts.append(p1 + vec * (j / (num_steps + 1)))
            
        new_pts.append(p2)
            
    return np.array(new_pts, dtype=np.float32)

def clean_polygon(pts, min_edge_len=1.0):
    # 差分計算（微小な移動を除去）
    if len(pts) == 0: return pts
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate(([True], diffs > 1e-4))
    pts = pts[keep]
        
    if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) < min_edge_len:
        pts = pts[:-1]
            
    cleaned = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - cleaned[-1]) >= min_edge_len:
            cleaned.append(p)
    pts = np.array(cleaned)
        
    if len(pts) < 3:
        return pts

    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack((pts, pts[0]))

    return pts