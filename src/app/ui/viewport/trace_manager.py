import numpy as np
import pyvista as pv
import vtk
from PyQt6.QtCore import Qt, QEvent
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, unary_union, polygonize
from app.core.geometry.topology import snap_lines

class TraceManager:
    def __init__(self, viewport):
        self.viewport = viewport # 親ウィジェットへの参照
        self.plotter = viewport.plotter
        
        self.is_active = False
        self.is_drawing = False
        
        # 現在描画中のストローク
        self.current_points_2d = [] # スクリーン座標
        self.current_points_3d = [] # ワールド座標(表示用)
        
        # 確定済みストロークのリスト
        self.strokes_2d = [] 
        self.strokes_3d = []
        
        self.actors = [] # 描画用アクターのリスト

        self.current_actor = None

    def start_trace(self):
        self.is_active = True
        self.clear()

    def stop_trace(self):
        self.is_active = False
        self.clear()

    def clear(self):
        """スケッチのリセット"""
        self.is_drawing = False
        self.current_points_2d = []
        self.current_points_3d = []
        self.strokes_2d = []
        self.strokes_3d = []
        
        # アクターの削除
        for actor in self.actors:
            self.plotter.remove_actor(actor)
        self.actors = []

        if self.current_actor:
            self.plotter.remove_actor(self.current_actor)
            self.current_actor = None
        
        self.viewport._safe_render()

    def handle_event(self, event):
        """ViewportのeventFilterから呼ばれる"""
        if not self.is_active: return False

        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                if self.viewport.current_mesh: # メッシュがある時だけ
                    self.is_drawing = True
                    self.current_points_2d = []
                    self.current_points_3d = []
                    self._add_point(event.pos())
            return True # イベント消費 (カメラ操作ブロック)

        elif event.type() == QEvent.Type.MouseMove:
            if self.is_drawing:
                self._add_point(event.pos())
            return True

        elif event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
                self.is_drawing = False
                self._finish_stroke()
            return True
        
        elif event.type() == QEvent.Type.Wheel:
            return True # ズームブロック

        return False

    def _add_point(self, qt_pos):
        ratio = self.plotter.devicePixelRatioF()
        qt_x = qt_pos.x() * ratio
        qt_y = qt_pos.y() * ratio
        win_h = self.plotter.height() * ratio
        vtk_x = qt_x
        vtk_y = win_h - qt_y

        self.current_points_2d.append([qt_pos.x(), qt_pos.y()])

        # カメラ手前の空間座標を計算 (スクリーン描画風)
        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToDisplay()
        coordinate.SetValue(vtk_x, vtk_y, 0.0)
        world_pt = coordinate.GetComputedWorldValue(self.plotter.renderer)
        self.current_points_3d.append(world_pt)

        self._update_current_actor()

    def _update_current_actor(self):
        # 現在描いている線の表示更新
        # 既存の current_actor を削除して再追加するのは重いが、PyVistaの簡易実装としては一般的
        # 本格的には vtkPolyData を直接更新する方が良い
        if len(self.current_points_3d) < 2: return

        # 既存の「現在描画中」アクターを探して削除
        # (リストの最後に追加することにする)
        if self.current_actor:
            self.plotter.remove_actor(self.current_actor)
            self.current_actor = None
        
        points = np.array(self.current_points_3d)
        line_mesh = pv.lines_from_points(points)
        
        self.current_actor = self.plotter.add_mesh(
            line_mesh, color="black", line_width=3,
            render_lines_as_tubes=False, reset_camera=False,
            name="current_stroke"
        )
        
        # 最前面表示設定
        mapper = self.current_actor.GetMapper()
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-6.0, -6.0)
        
        self.viewport._safe_render()
    
    def _finish_stroke(self):
        """ストローク終了時の処理。確定リストに移す"""
        if len(self.current_points_2d) > 2:
            # 1筆書きで閉じた場合の自動クローズ処理
            start = np.array(self.current_points_2d[0])
            end = np.array(self.current_points_2d[-1])
            if np.linalg.norm(start - end) < 20.0: # 20px以内なら閉じる
                self.current_points_2d.append(self.current_points_2d[0])
                # 3D側も閉じる(見た目のため)
                self.current_points_3d.append(self.current_points_3d[0])
                # 表示を更新して閉じた状態を見せる
                self._update_current_actor()

            self.strokes_2d.append(self.current_points_2d)
            self.strokes_3d.append(self.current_points_3d)
            
            if self.current_actor:
                self.actors.append(self.current_actor)
                self.current_actor = None
        
        self.current_points_2d = []
        self.current_points_3d = []
        
        # 閉領域判定をトライ
        self.check_closed_region()

    def check_closed_region(self):
        """
        描かれたストローク群が閉じた領域を形成しているか判定する。
        閉じていれば、領域確定処理（シグナル発火やコールバック）を行う。
        """
        if not self.strokes_2d: return

        # Shapelyを使って結合・ポリゴン化を試みる
        try:
            lines = []
            for stroke in self.strokes_2d:
                if len(stroke) >= 2:
                    lines.append(LineString(stroke))
            
            snapped_lines = snap_lines(lines, tolerance=20.0)

            # 結合
            merged = linemerge(unary_union(snapped_lines))

            # 線分結合 (LineMerge)
            if merged.geom_type == 'MultiLineString':
                merged = linemerge(merged)
            
            # ポリゴン化
            polygons = list(polygonize(merged))
            
            if polygons:
                print("Closed region detected!")
                # 最大のポリゴンを採用
                poly = max(polygons, key=lambda p: p.area)
                
                # 外周座標を取得
                outline_2d = list(poly.exterior.coords)
                
                # Viewport側に通知して、領域確認モードへ移行
                self.viewport.on_trace_completed(outline_2d)
                
        except Exception as e:
            # まだ閉じていない、または交差が複雑すぎる
            pass