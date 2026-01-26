import os
import pyvista as pv
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QButtonGroup, 
    QMessageBox, QScrollArea, QFileDialog, QStackedWidget, QFrame, QApplication, QSizePolicy
)
from PyQt6.QtCore import Qt
from app.canvas.sketch_canvas import SketchCanvas
from app.core.modeling.mesh_generator import MeshGenerator
from app.core.geometry.topology import extract_largest_polygon
from app.ui.viewport.viewport_3d_widget import Viewport3D
from app.ui.sidebars.sidebar_sketch import SketchSidebar
from app.ui.sidebars.sidebar_object import ObjectSidebar

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Draw Mask")

        self.resize(1280, 800)

        self.setStyleSheet("""
            QPushButton { font-size: 12px; padding: 8px; margin: 2px; text-align: left; }
            QPushButton:checked { background-color: #3daee9; color: white; border: 2px solid #2da0db; }
            
            QPushButton.mode_btn { 
                font-weight: bold; font-size: 14px; text-align: center;
                background-color: #ddd; border: 1px solid #bbb;
                border-radius: 4px;
            }
            QPushButton.mode_btn:checked {
                background-color: #555; color: white; border: 1px solid #333;
            }
                           
            QPushButton#fit_btn {
                background-color: #FFD700; color: black; font-weight: bold; font-size: 14px; 
                margin-top: 15px; text-align: center;
                border: 1px solid #b08d00; border-radius: 4px; padding: 8px;
            }
            QPushButton#fit_btn:hover { background-color: #FFC107; }
                           
            QPushButton#run_btn { 
                background-color: #ff6b6b; color: white; font-weight: bold; font-size: 14px; 
                margin-top: 15px; text-align: center;
            }
            QPushButton#run_btn:hover { background-color: #ff5252; }
                           
            QPushButton#save_btn { 
                background-color: #e0e0e0; margin-top: 5px; border: 1px solid #999; text-align: center;
            }              
            QPushButton#save_btn:hover { background-color: #d0d0d0; }
                           
            QLabel { font-weight: bold; margin-top: 10px; }
            QSlider { margin-bottom: 15px; }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_widget.setLayout(main_layout)

        self.canvas = SketchCanvas()
        self.viewport = Viewport3D()
        self.viewport.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.viewport.trace_confirmed.connect(self.on_trace_selection_confirmed)

        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_widget.setFixedWidth(280)
        
        mode_layout = QHBoxLayout()
        self.mode_group = QButtonGroup(self)

        self.btn_mode_sketch = QPushButton("Sketch Mode")
        self.btn_mode_sketch.setCheckable(True)
        self.btn_mode_sketch.setChecked(True)
        self.btn_mode_sketch.setProperty("class", "mode_btn")

        self.btn_mode_3d = QPushButton("3D Object Mode")
        self.btn_mode_3d.setCheckable(True)
        self.btn_mode_3d.setProperty("class", "mode_btn")

        self.mode_group.addButton(self.btn_mode_sketch, 0)
        self.mode_group.addButton(self.btn_mode_3d, 1)
        self.mode_group.idClicked.connect(self.on_mode_changed)

        mode_layout.addWidget(self.btn_mode_sketch)
        mode_layout.addWidget(self.btn_mode_3d)

        sidebar_layout.addLayout(mode_layout)
        sidebar_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine))

        self.sidebar_stack = QStackedWidget()
        self.sketch_sidebar = SketchSidebar()
        self.sidebar_stack.addWidget(self.sketch_sidebar)
        
        # シグナル
        self.sketch_sidebar.tool_changed.connect(self.canvas.set_tool)
        self.sketch_sidebar.param_changed.connect(self.on_param_changed)
        self.sketch_sidebar.reset_clicked.connect(self.on_canvas_reset)
        self.sketch_sidebar.undo_clicked.connect(self.canvas.undo)
        self.sketch_sidebar.save_clicked.connect(self.on_save_sketch_clicked)
        self.sketch_sidebar.run_clicked.connect(self.on_run_clicked)
        self.sketch_sidebar.fit_clicked.connect(self.on_fit_to_object_clicked)
        self.sketch_sidebar.trace_fit_clicked.connect(self.on_trace_fit_clicked)

        self.object_sidebar = ObjectSidebar()
        self.sidebar_stack.addWidget(self.object_sidebar)
        
        self.object_sidebar.import_clicked.connect(self.on_import_clicked)
        self.object_sidebar.export_clicked.connect(self.on_export_obj_clicked)
        self.object_sidebar.mode_changed.connect(self.on_object_mode_changed)
        self.object_sidebar.toggle_edges.connect(self.viewport.toggle_edges)
        self.object_sidebar.toggle_edl.connect(self.viewport.toggle_edl)
        self.object_sidebar.toggle_gizmo.connect(self.viewport.toggle_gizmo)
        self.object_sidebar.smooth_clicked.connect(self.viewport.apply_global_smoothing)

        sidebar_layout.addWidget(self.sidebar_stack)
        main_layout.addWidget(sidebar_widget, 0)

        self.view_stack = QStackedWidget(); self.view_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll_area = QScrollArea(); scroll_area.setWidget(self.canvas); scroll_area.setWidgetResizable(True)
        self.view_stack.addWidget(scroll_area); self.view_stack.addWidget(self.viewport)
        main_layout.addWidget(self.view_stack, 1)

    # --- イベントハンドラ ---
    def on_mode_changed(self, index):
        self.sidebar_stack.setCurrentIndex(index)
        self.view_stack.setCurrentIndex(index)
    
    def on_param_changed(self):
        sb = self.sketch_sidebar
        m = sb.slider_mag.value()
        p = sb.slider_prof.value() / 10.0
        i = sb.slider_inf.value() / 10.0
        self.canvas.update_params(m, p, i)
    
    def on_canvas_reset(self):
        self.canvas.reset_canvas()
        self.sketch_sidebar.set_trace_fit_enabled(False)
    
    def on_save_sketch_clicked(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(base_dir, "output")
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", output_dir, "PNG (*.png)")
        if path: self.canvas.pixmap.save(path, "PNG")

    def on_run_clicked(self):
        try:
            print("Generating...")
            generator = MeshGenerator()
            w = self.canvas.width()
            h = self.canvas.height()
            mesh = generator.generate(
                self.canvas.sketch_data,
                base_mesh_data=self.canvas.base_mesh_data,
                canvas_size=(w, h),
                is_fitting=False
            )
            self.viewport.update_mesh(mesh)
            self.btn_mode_3d.setChecked(True)
            self.on_mode_changed(1)
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Generation failed:\n{str(e)}")

    def on_fit_to_object_clicked(self):
        """スケッチ輪郭を3Dオブジェクトに張り付ける"""
        
        # 1. 現在のスケッチから輪郭を取得
        strokes = self.canvas.sketch_data.get_strokes()

        boundary_fixed = strokes.get('boundary_fixed', [])
        boundary_free = strokes.get('boundary_free', [])
        boundary_pts = boundary_fixed + boundary_free

        deformations = strokes.get('deformation', [])

        holes = strokes.get('hole', [])
        fixed_points = strokes.get('fixed_point', [])
        
        if not boundary_pts:
            QMessageBox.warning(self, "Warning", "輪郭線(Boundary)を描いてください。")
            return

        # 2. 閉じたポリゴンとして整理
        polygon = extract_largest_polygon(boundary_pts)
        
        if polygon is None:
            QMessageBox.warning(self, "Warning", "有効な閉じた輪郭が見つかりません。")
            return

        # 3. 3Dモードに切り替えて実行 (オブジェクトが表示されている必要がある)
        self.btn_mode_3d.setChecked(True)
        self.on_mode_changed(1)

        # 描画更新
        QApplication.processEvents()
        self.viewport.plotter.render()
        
        # 4. ビューポートに依頼
        w = self.canvas.width()
        h = self.canvas.height()

        if deformations:
            print("Deforming base object...")
            for item in deformations:
                points = item['points']
                params = item['params']
                self.viewport.apply_deformation_to_mesh(points, fixed_points, params, (w, h), True)

        success = self.viewport.fit_sketch_to_object(polygon, holes, (w, h))
        
        if not success:
             QMessageBox.warning(self, "Failed", "オブジェクトへの投影に失敗しました。")

    def on_import_clicked(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        input_dir = os.path.join(base_dir, "input")
        if not os.path.exists(input_dir): os.makedirs(input_dir)
        path, _ = QFileDialog.getOpenFileName(self, "Import Object", input_dir, "3D Files (*.obj *.stl *.ply *.vtk)")
        if path:
            try:
                self.viewport.load_file(path)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")

    def on_export_obj_clicked(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(base_dir, "output")
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        path, _ = QFileDialog.getSaveFileName(self, "Export Object", output_dir, "OBJ Files (*.obj);;STL Files (*.stl);;PLY Files (*.ply)")
        if path:
            try:
                self.viewport.save_file(path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file:\n{e}")

    def on_object_mode_changed(self, is_trace):
        """サイドバーからのモード切替"""
        if is_trace:
            # Traceモード開始
            self.viewport.set_trace_mode(True)
        else:
            # Viewモードへ
            self.viewport.set_trace_mode(False)

    def on_trace_selection_confirmed(self, selected_mesh, outline_2d):
        """
        Traceモードで領域指定が完了し、Yesが押されたときの処理
        """
        print("Trace selection confirmed in MainWindow.")
        
        # 1. 2D輪郭を正規化してスケッチキャンバスにセット
        # Viewportのサイズで正規化 (0.0~1.0)
        w = self.viewport.width()
        h = self.viewport.height()
        normalized_outline = []
        for p in outline_2d:
            # ViewportのTraceManagerが保持する座標はスクリーン座標
            # ここでは outline_2d がそのままスクリーン座標(px)で来ていると仮定
            # (SelectionOperatorでの計算に使われているものと同じ)
            nx = p[0] / w
            ny = p[1] / h
            normalized_outline.append([nx, ny])

        # キャンバスにセット (Boundary Fixedとして登録)
        self.canvas.set_external_boundary(normalized_outline)
        
        # 2. モードをスケッチモードに切り替え
        self.btn_mode_sketch.setChecked(True)
        self.on_mode_changed(0)
        
        # 完了したらViewモードに戻す
        self.object_sidebar.mode_view_btn.setChecked(True)
        self.viewport.set_trace_mode(False)

        self.sketch_sidebar.set_trace_fit_enabled(True)

    def on_trace_fit_clicked(self):
        """Traceで指定した視点・領域で貼り付けを実行"""
        strokes = self.canvas.sketch_data.get_strokes()
        boundary_pts = strokes.get('boundary_fixed', []) + strokes.get('boundary_free', [])
        deformations = strokes.get('deformation', [])
        holes = strokes.get('hole', [])
        fixed_points = strokes.get('fixed_point', [])
        
        if not boundary_pts: return

        polygon = extract_largest_polygon(boundary_pts)
        if polygon is None: return

        self.btn_mode_3d.setChecked(True)
        self.on_mode_changed(1)
        QApplication.processEvents()
        self.viewport.plotter.render()
        
        w = self.canvas.width(); h = self.canvas.height()

        # 1. Trace視点で変形
        if deformations:
            print("Deforming with Trace View...")
            for item in deformations:
                points = item['points']; params = item['params']
                self.viewport.apply_deformation_with_trace_view(points, fixed_points, params, (w, h), False)

        # 2. Trace視点でパッチ生成
        success = self.viewport.fit_trace_to_object(polygon, holes, (w, h))
        if not success: QMessageBox.warning(self, "Failed", "投影に失敗しました。")
    
    def on_mesh_refined(self, mesh):
        """再分割が完了した際の処理"""
        print("Mesh refined. Switching UI to View mode.")
        # サイドバーのボタン状態を更新
        self.object_sidebar.mode_view_btn.setChecked(True)
        self.object_sidebar.mode_trace_btn.setChecked(False)
        # モード自体はViewport側で set_trace_mode(False) されているのでUIの同期のみ