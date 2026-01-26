import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import unary_union, linemerge, polygonize

class SketchData:
    def __init__(self):
        # ストロークの種類ごとに座標リストを保存する辞書
        # キー: ストロークの種類, 値: [ [x, y], [x, y], ... ] のリストのリスト
        self.strokes = {
            'boundary_fixed': [],    # 輪郭(固定)
            'boundary_free': [],     # 輪郭(可動)
            'fixed_point': [],       # 固定点
            'hole': [],              # 穴
            'deformation': []        # 変形用 (パラメータ付き)
        }

        self.history = []  # アンドゥ用に履歴を保存するリスト

    def add_stroke(self, stroke_type, points, params=None):
       """
       params: {
           'magnitude': float, # 曲がり具合 (-100 ~ 100)
           'profile': float,   # 曲がり方 (指数: 0=直線/平坦, 1=線形, 2=曲線)
           'influence': float  # 影響範囲 (今回は簡易的にメッシュの剛性に反映などを想定)
       }
       """
       if stroke_type in self.strokes:
            # numpy配列に変換して格納 (計算で使いやすくするため)
            pts_array = np.array(points, dtype=np.float32)
            # 重複点除去（Shapelyのエラー防止）
            if len(pts_array) > 1:
                # deformation のみパラメータと一緒に保存
                if stroke_type == 'deformation':
                    if params is None: 
                        params = {'magnitude': 50, 'profile': 1.0, 'influence': 1.0}
                    added_obj = {
                        'points': pts_array,
                        'params': params
                    }
                    self.strokes[stroke_type].append(added_obj)
                else:
                    # boundary, base_line, hole は座標のみ
                    added_obj = pts_array
                    self.strokes[stroke_type].append(added_obj)
            
            self.history.append({
                    'type': stroke_type,
                    'data': added_obj,
                    # 描画再現用に生の点群も保持しておく
                    'points': points, 
                    'params': params
                })
    
    def undo(self):
        """
        最後に描いたストロークを取り消す
        戻り値: True=成功, False=履歴なし
        """
        if not self.history:
            return False
            
        # 1. 履歴から最後の操作を取り出す
        last_action = self.history.pop()
        s_type = last_action['type']
        target_obj = last_action['data']
        
        # 2. 現在のストロークデータから、そのオブジェクトを削除する
        if s_type in self.strokes:
            target_list = self.strokes[s_type]
            
            # 消しゴム等でリストが再生成されている場合、オブジェクトIDが変わっている可能性があるため
            # try-except で安全に削除を試みる
            try:
                # リスト内に同一オブジェクトがあれば削除
                if target_obj in target_list:
                    target_list.remove(target_obj)
                else:
                    # オブジェクトが見つからない場合（消しゴムで分割された後など）
                    # 厳密なUndoは難しいが、ここではエラーにせず「履歴からは消えた」こととする
                    pass
            except ValueError:
                pass
                
        return True

    def clear(self):
        for key in self.strokes:
            self.strokes[key] = []
        self.history = []
    
    def get_strokes(self):
        return self.strokes
    
    def erase_strokes(self, eraser_points, radius=18.0):
        """
        eraser_points: 消しゴムの軌跡 [(x,y), ...]
        radius: 消しゴムの半径 (幅36pxなら半径18)
        """
        if not eraser_points:
            return

        # 消しゴムの点群で検索ツリーを作成 (高速化のため)
        eraser_tree = cKDTree(eraser_points)

        new_strokes_structure = {k: [] for k in self.strokes}

        # 1. 座標のみを持つタイプ (Hole含む)
        simple_types = ['boundary_fixed', 'boundary_free', 'fixed_point', 'hole']
        for t in simple_types:
            for stroke in self.strokes[t]:
                if len(stroke) == 0: continue
                dists, _ = eraser_tree.query(stroke)
                current_segment = []
                for i, p in enumerate(stroke):
                    if dists[i] > radius:
                        current_segment.append(p)
                    else:
                        if len(current_segment) > 1:
                            new_strokes_structure[t].append(np.array(current_segment, dtype=np.float32))
                        current_segment = []
                if len(current_segment) > 1:
                    new_strokes_structure[t].append(np.array(current_segment, dtype=np.float32))

        # 2. パラメータを持つタイプ (Deformation)
        for item in self.strokes['deformation']:
            stroke = item['points']
            params = item['params']
            if len(stroke) == 0: continue
            dists, _ = eraser_tree.query(stroke)
            current_segment = []
            for i, p in enumerate(stroke):
                if dists[i] > radius:
                    current_segment.append(p)
                else:
                    if len(current_segment) > 1:
                        new_strokes_structure['deformation'].append({
                            'points': np.array(current_segment, dtype=np.float32),
                            'params': params
                        })
                    current_segment = []
            if len(current_segment) > 1:
                new_strokes_structure['deformation'].append({
                    'points': np.array(current_segment, dtype=np.float32),
                    'params': params
                })

        self.strokes = new_strokes_structure
        print("Erase processing complete.")