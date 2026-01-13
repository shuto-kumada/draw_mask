from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPen, QPixmap, QColor, QMouseEvent, QCursor, QPolygonF
from PyQt6.QtCore import Qt, QPoint, QPointF
from app.data.sketch_data import SketchData

class SketchCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.sketch_data = SketchData()
        self.current_stroke_points = []
        self.current_params = {'magnitude': 50.0, 'profile': 1.0, 'influence': 1.0}

        # ガイド（切り抜き輪郭）用
        self.guide_polygon = None

        # ベースメッシュの投影データ (Generator用)
        self.base_mesh_data = None

        self.initialize_canvas()

    def initialize_canvas(self):
        self.pixmap = QPixmap(1200, 900)
        self.pixmap.fill(Qt.GlobalColor.white)
        self.last_point = QPoint()
        
        self.current_tool_type = 'boundary_fixed' 
        self.set_tool('boundary_fixed')

    def set_external_boundary(self, normalized_points):
        """
        Traceモード等で作られた輪郭点群(0.0~1.0)を受け取り、
        boundary_fixedストロークとして登録して描画する。
        """
        if not normalized_points: return

        # 1. データのリセット
        self.reset_canvas()
        
        # 2. 座標変換 (正規化 -> ピクセル)
        w = self.width() if self.width() > 0 else 1200
        h = self.height() if self.height() > 0 else 900
        
        pixel_points = []
        for p in normalized_points:
            # pは [x_norm, y_norm] (yは下がプラスのQt座標系であることを期待)
            # Viewportからの出力は y_norm = 1.0 - (y_vtk / win_h) となっていればOK
            px = p[0] * w
            py = p[1] * h
            pixel_points.append([px, py])
            
        # 3. ストロークとして登録
        # Boundary Fixed として登録する
        self.sketch_data.add_stroke('boundary_fixed', pixel_points)
        
        # 4. 描画 (黒線)
        painter = QPainter(self.pixmap)
        pen = QPen(Qt.GlobalColor.black, 3, 
                   Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        
        poly = QPolygonF([QPointF(pt[0], pt[1]) for pt in pixel_points])
        
        # 閉じた線として描画
        painter.drawPolyline(poly)
        # 始点と終点を結ぶ
        if len(pixel_points) > 2:
            painter.drawLine(QPointF(*pixel_points[-1]), QPointF(*pixel_points[0]))
            
        painter.end()
        self.update()
        print("External boundary set.")
    
    def update_params(self, magnitude, profile, influence):
        self.current_params['magnitude'] = magnitude
        self.current_params['profile'] = profile
        self.current_params['influence'] = influence
        if self.current_tool_type == 'deformation':
            self._update_deformation_pen()
    
    def _update_deformation_pen(self):
        mag = self.current_params['magnitude']
        # 正:赤, 負:青, 0:緑
        alpha = min(255, int(abs(mag) * 2.0 + 55))
        if mag > 5:
            self.current_color = QColor(255, 0, 0, alpha) # 凸
        elif mag < -5:
            self.current_color = QColor(0, 0, 255, alpha) # 凹
        else:
            self.current_color = QColor(0, 255, 0, 200)   # 平坦
            
        self.current_width = 4
        self.current_cap = Qt.PenCapStyle.RoundCap

    def set_tool(self, tool_type):
        """ツール切り替え"""
        self.current_tool_type = tool_type
        
        if tool_type == 'deformation':
            self._update_deformation_pen()
            self.setCursor(Qt.CursorShape.ArrowCursor)
        
        elif tool_type == 'eraser':
            self.current_color = Qt.GlobalColor.white
            self.current_width = 36
            self.current_cap = Qt.PenCapStyle.SquareCap

            # 正方形カーソル
            cursor_size = self.current_width
            pixmap = QPixmap(cursor_size, cursor_size)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            pen = QPen(Qt.GlobalColor.black)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawRect(0, 0, cursor_size - 1, cursor_size - 1)
            painter.end()
            self.setCursor(QCursor(pixmap, cursor_size // 2, cursor_size // 2))

        else:
            # 各ツールの固定色
            if tool_type == 'boundary_fixed':
                self.current_color = Qt.GlobalColor.black
                self.current_width = 3
            elif tool_type == 'boundary_free':
                self.current_color = QColor(0, 0, 139)
                self.current_width = 3
            elif tool_type == 'fixed_point':
                self.current_color = QColor(200, 200, 200)
                self.current_width = 3
            elif tool_type == 'hole':  # Hole設定を維持
                self.current_color = QColor(150, 150, 150)
                self.current_width = 3
            
            self.current_cap = Qt.PenCapStyle.RoundCap
            self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def set_guide_outline(self, normalized_points):
        """
        3Dビューから受け取った輪郭点(0.0~1.0)をキャンバスサイズに合わせて保持
        """
        if not normalized_points:
            self.guide_polygon = None
        else:
            w = self.width()
            h = self.height()
            # 座標変換 (0.0-1.0 -> 0-Width/Height)
            # 3DのY座標は上が+だが、Qtは下が+なのでYを反転させる必要がある場合が多いが、
            # Viewportからのデータ形式による。ここではそのまま変換する。
            points = [QPointF(p[0] * w, p[1] * h) for p in normalized_points]
            self.guide_polygon = QPolygonF(points)
        
        self.update() # 再描画
    
    def set_base_mesh_data(self, data):
        """Viewportから受け取った投影データ [[x,y,z], ...] を保持"""
        self.base_mesh_data = data

    def reset_canvas(self):
        """全てをクリアして初期状態に戻す"""
        # データクリア
        self.sketch_data.clear()
        self.current_stroke_points = []
        self.base_mesh_data = None
        self.guide_polygon = None
        
        # 画面クリア
        self.pixmap.fill(Qt.GlobalColor.white)
        self.update()
        print("Canvas Reset")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap)

        if self.guide_polygon:
            # 薄いグレーの点線で描画
            pen = QPen(QColor(150, 150, 150, 100), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolygon(self.guide_polygon)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = event.position().toPoint()
            self.current_stroke_points = []
            self.current_stroke_points.append((self.last_point.x(), self.last_point.y()))

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.MouseButton.LeftButton:
            painter = QPainter(self.pixmap)
            pen = QPen(self.current_color, self.current_width, 
                       Qt.PenStyle.SolidLine, self.current_cap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)

            current_point = event.position().toPoint()
            painter.drawLine(self.last_point, current_point)
            painter.end()

            self.last_point = current_point
            self.update()
            
            self.current_stroke_points.append((current_point.x(), current_point.y()))

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            
            # 消しゴムの場合: 削除処理を実行
            if self.current_tool_type == 'eraser':
                if len(self.current_stroke_points) > 0:
                    # 消しゴムの半径 (太さの半分)
                    radius = self.current_width / 2.0
                    self.sketch_data.erase_strokes(self.current_stroke_points, radius=radius)
            
            # 通常ツールの場合: ストロークを追加
            elif len(self.current_stroke_points) > 2:
                self.sketch_data.add_stroke(
                    self.current_tool_type, 
                    self.current_stroke_points,
                    params=self.current_params.copy() # 現在のスライダー値をコピーして渡す
                )
                print(f"Added stroke: {self.current_tool_type}")
            
            self.current_stroke_points = []
    
    def resizeEvent(self, event):
        super().resizeEvent(event)