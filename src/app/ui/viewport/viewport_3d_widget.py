import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt, QEvent, pyqtSignal
from app.ui.viewport.overlay import ConfirmOverlay
from app.ui.viewport.trace_manager import TraceManager
from app.ui.viewport.selection_operator import SelectionOperator

class Viewport3D(QWidget):
    # シグナル定義
    trace_completed = pyqtSignal() # トレース完了・確認待ち状態になった
    trace_confirmed = pyqtSignal(object, list) # トレース確定 (切り抜きメッシュを渡す)

    # メッシュが更新されたことをMainWindowに知らせる
    mesh_refined = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # レイアウト設定
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout) #消しちゃダメ

        # PyVistaのQt埋め込み用ウィジェット
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter, 1)

        # 初期設定
        self.plotter.set_background("#333333") # 背景を少し暗くしてモデルを目立たせる

        # Eye-Dome Lighting (EDL)
        self.show_gizmo = False

        self.current_mesh = None # 現在表示中のメッシュデータを保持
        self.original_mesh = None 
        self.actor = None # 表示中のアクター（操作用）

        self.preview_actor = None # プレビュー表示用のアクター

        # Trace時のカメラ位置保存用
        self.trace_camera_pos = None

        # --- サブモジュールの初期化 ---
        # 1. トレース管理
        self.trace_manager = TraceManager(self)
        
        # 2. 確認オーバーレイ
        self.confirm_widget = ConfirmOverlay(self)

        # 内部でボタンイベントを処理し、シグナルに変換する
        self.btn_confirm = self.confirm_widget.btn_confirm
        self.btn_retry = self.confirm_widget.btn_retry

        self.btn_confirm.clicked.connect(self.on_yes_clicked)
        self.btn_retry.clicked.connect(self.on_no_clicked)

        # イベントフィルタ
        self.plotter.installEventFilter(self)

        # 確定用データ保持
        self.pending_region_mesh = None
        
        # リセットボタン (Traceモード時のみ表示したい)
        self.btn_trace_reset = QPushButton("Reset Trace", self)
        self.btn_trace_reset.setStyleSheet("background-color: #ffcccc; color: darkred; font-weight: bold; border-radius: 4px; padding: 5px;")
        self.btn_trace_reset.clicked.connect(self.on_trace_reset)
        self.btn_trace_reset.hide()
    
    # --- 基本機能 ---
    def _safe_render(self):
        """ウィンドウサイズが有効な場合のみ描画する"""
        if self.isVisible() and self.plotter.width() > 1 and self.plotter.height() > 1:
            self.plotter.render()
    
    def update_mesh(self, mesh, reset_camera=True, is_preview=False):
        """メッシュを受け取って表示を更新する"""
        self.current_mesh = mesh
        self.original_mesh = mesh

        # 新しいメッシュを読み込むときはプレビューを消す
        self.remove_preview()

        if self.actor:
            self.plotter.remove_actor(self.actor)
            self.actor = None
        
        if mesh:
            if not isinstance(mesh, pv.PolyData):
                mesh = mesh.extract_surface()

            # 常に手前に表示されるように法線を再計算しておく
            if mesh.point_normals is None:
                mesh.compute_normals(inplace=True, auto_orient_normals=True)

            # === 見た目の調整 ===
            # smooth_shading: 滑らかに表示
            # specular: 光沢の強さ
            # show_edges: ポリゴンのエッジを表示するかどうか (デフォルトはFalseで見やすく)
            self.actor = self.plotter.add_mesh(
                mesh, 
                color="lightblue",        # 明るい水色 (陰影が見やすい)
                smooth_shading=True,      # 滑らかシェーディング
                specular=0.5,             # 適度な光沢
                specular_power=15,        # ハイライトの鋭さ
                show_edges=False,         # ワイヤーフレームは最初はOFF
                edge_color="black",
                line_width=1.0
            )
            
            if reset_camera:
                self.plotter.view_xy()
                self.plotter.camera.up = (0, -1, -1)
                self.plotter.enable_parallel_projection()
                self.plotter.reset_camera()
        
        self._safe_render()

    def remove_preview(self):
        if self.preview_actor:
            self.plotter.remove_actor(self.preview_actor)
            self.preview_actor = None
        self._safe_render()

    # === Traceモード制御 ===
    def set_trace_mode(self, enabled):
        if enabled:
            if not self.current_mesh:
                print("No mesh to trace.")
                return
            
            # カメラ固定（イベントフィルタで制御）
            self.trace_manager.start_trace()
            self.setCursor(Qt.CursorShape.CrossCursor)
            
            # リセットボタン表示
            self.btn_trace_reset.show()
            self._update_ui_pos()
            
            # 既存のプレビュー消去
            self.remove_preview()
            
        else:
            self.trace_manager.stop_trace()
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.btn_trace_reset.hide()
            self.confirm_widget.hide()
            self.remove_preview()
    
    def clear_trace(self):
        self.trace_manager.clear()

    def on_trace_reset(self):
        """トレースをリセットしてやり直す"""
        self.trace_manager.clear()
        self.remove_preview()
        self.confirm_widget.hide()
        # Traceモードは継続

    def on_trace_completed(self, outline_2d):
        """
        TraceManagerで輪郭が閉じられた際に自動で呼ばれる。
        Fit Objectと同じ『高品質パッチ生成』方式で可視化を行う。
        """
        if not self.current_mesh: return
        self.pending_outline = outline_2d

        self.trace_camera_pos = self.plotter.camera_position

        # Fit Objectと同じロジック(create_high_quality_patch)を呼び出す
        # Traceモードなので穴(holes)は空リスト、canvas_sizeは現在のサイズを使用
        w, h = self.width(), self.height()
        
        region = SelectionOperator.create_high_quality_patch(
            self.current_mesh, 
            outline_2d, 
            [], # Traceモードでは現在穴ストロークは未定義
            self.plotter, 
            (w, h)
        )
        
        if region and region.n_points > 0:
            self.pending_region_mesh = region
            
            # ハイライト表示
            self.remove_preview()
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.preview_actor = self.plotter.add_mesh(
                region, 
                color="#FF0000",       # 赤色
                opacity=0.9, 
                smooth_shading=True, 
                name="trace_preview_patch"
            )
            
            # 最前面に表示するためのオフセット設定
            mapper = self.preview_actor.GetMapper()
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
            mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2.0, -2.0)

            self._safe_render()
            
            # 確認UI
            self.confirm_widget.show_message("この領域を決定しますか？")
            self.confirm_widget.update_position(self.rect())

            self.btn_trace_reset.hide()
        else:
            print("Trace projection failed. Please retry.")
            self.trace_manager.clear()

    # === 保存したTrace視点で変形を行うメソッド ===
    def apply_deformation_with_trace_view(self, sketch_points, fixed_strokes, params, canvas_size, do_subdivide=True):
        """
        Trace時のカメラ視点に一時的に戻してから変形計算を行い、その後元の視点に戻す。
        これにより、ユーザーが視点を変えても正しい位置に変形が適用される。
        """
        if not self.current_mesh: return False
        
        # 現在のカメラ状態をバックアップ
        current_camera = self.plotter.camera_position
        
        # Trace時のカメラ状態があれば復元
        if self.trace_camera_pos:
            self.plotter.camera_position = self.trace_camera_pos
            # 座標変換のためにレンダリング更新が必要
            # (描画更新はしないが、内部行列は更新する必要があるため)
            # self.plotter.renderer.ResetCameraClippingRange() # 必要に応じて
        
        # 変形実行 (内部でPickerやCoordinate変換が走る)
        result = self.apply_deformation_to_mesh(sketch_points, fixed_strokes, params, canvas_size, do_subdivide)
        
        # カメラを元の状態に戻す
        self.plotter.camera_position = current_camera
        self._safe_render()
        
        return result

    # === 確認ボタンハンドラ ===
    def on_yes_clicked(self):
        """Yes: 確定"""
        print("DEBUG: Yes button clicked!") # デバッグ用ログ
        
        if self.current_mesh and self.pending_region_mesh:
            refined = SelectionOperator.subdivide_region(self.current_mesh, self.pending_region_mesh)
            if refined:
                self.update_mesh(refined, reset_camera=False)
                self.mesh_refined.emit(refined)

        self.hide_confirmation()
        self.remove_preview()
        self.trace_manager.clear()
        self.set_trace_mode(False) # モード終了

        # MainWindowへ通知
        self.trace_confirmed.emit(self.pending_region_mesh, self.pending_outline)

    def on_no_clicked(self):
        """No: やり直し"""
        print("DEBUG: No button clicked!") # デバッグ用ログ
        
        self.hide_confirmation()
        self.remove_preview()
        self.trace_manager.clear()
        
        # 十字カーソルに戻して描き直しを可能にする
        self.setCursor(Qt.CursorShape.CrossCursor)

    # === UIレイアウト調整 ===
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_ui_pos()

    def _update_ui_pos(self):
        # リセットボタン: 左上
        self.btn_trace_reset.adjustSize()
        self.btn_trace_reset.move(20, 20)
        
        # 確認ウィジェット: 中央下
        if self.confirm_widget.isVisible():
            self.confirm_widget.update_position(self.rect())

    def hide_confirmation(self):
        self.confirm_widget.hide()
    
    # === イベントフィルタ ===
    def eventFilter(self, source, event):
        if source is self.plotter:
            # TraceManagerにイベントを渡す
            # 確認中は描画させないなどの制御も可能
            if not self.confirm_widget.isVisible():
                if self.trace_manager.handle_event(event):
                    return True
        return super().eventFilter(source, event)
    
    # --- 切り取り実行 (Operatorへ委譲) ---
    def extract_selection(self):
        # TraceManagerが持っている2D座標データを使う
        trace_pts = self.trace_manager.points_2d
        return SelectionOperator.extract_selection(self.current_mesh, trace_pts, self.plotter)
    
    # --- ファイルIO / 表示切替 (既存) ---
    def load_file(self, file_path):
        """ファイルを読み込んで表示"""
        try:
            mesh = pv.read(file_path)
            self.update_mesh(mesh)
            print(f"Loaded: {file_path}")
        except Exception as e:
            print(f"Load Error: {e}")
            raise e
        
    def save_file(self, file_path):
        """現在のメッシュを保存"""
        if self.current_mesh:
            self.current_mesh.save(file_path)
            print(f"Saved: {file_path}")
        else:
            raise RuntimeError("保存する3Dモデルがありません。")
        
    # === 表示切替用メソッド ===
    def toggle_edges(self, show):
        """ワイヤーフレーム（ポリゴン線）の表示切替"""
        if self.actor:
            prop = self.actor.GetProperty()
            if show:
                prop.EdgeVisibilityOn()
                prop.SetEdgeColor(0, 0, 0) # 黒線
            else:
                prop.EdgeVisibilityOff()
            self._safe_render()

    def toggle_edl(self, enable):
        # ウィンドウサイズが0の場合は処理しない（クラッシュ防止）
        if self.plotter.width() <= 1 or self.plotter.height() <= 1:
            return

        if enable:
            try:
                self.plotter.enable_eye_dome_lighting()
            except:
                pass
        else:
            self.plotter.disable_eye_dome_lighting()
        self._safe_render()
    
    def toggle_gizmo(self, show):
        if show:
            self.plotter.show_grid()
            self.plotter.add_axes()
        else:
            self.plotter.remove_bounds_axes() # グリッド削除
            self.plotter.hide_axes()          # 軸削除
        self._safe_render()

    def restore_original(self):
        if self.original_mesh:
            self.update_mesh(self.original_mesh, reset_camera=False, is_preview=False)

    def apply_deformation_to_mesh(self, sketch_points, fixed_strokes, params, canvas_size, do_subdivide=True):
        """スケッチとパラメータを使って、現在のメッシュを変形させる"""
        if not self.current_mesh:
            return False
        
        # 現在の視点をバックアップ
        current_camera = self.plotter.camera_position

        # Trace時の視点を復元 (これによって投影場所がスケッチ時と一致する)
        if self.trace_camera_pos:
            self.plotter.camera_position = self.trace_camera_pos
            # 行列計算のためにレンダリングはしないが内部更新が必要
            self.plotter.renderer.ResetCameraClippingRange()

        warped_mesh = SelectionOperator.warp_mesh_by_strokes(
            self.current_mesh, sketch_points, fixed_strokes, params, self.plotter, canvas_size, do_subdivide
        )

        # 元の視点に戻す
        self.plotter.camera_position = current_camera

        if warped_mesh:
            #self.current_mesh = warped_mesh

            # 描画更新 (reset_camera=False で視点を維持)
            self.update_mesh(warped_mesh, reset_camera=False)
            return True
            
        return False

    def fit_sketch_to_object(self, sketch_points, hole_strokes, canvas_size):
        """
        MeshGeneratorで生成されたメッシュを受け取り、強調表示する
        """
        if not self.current_mesh:
            return False

        # SelectionOperator.create_high_quality_patch を使用
        fitted_mesh = SelectionOperator.create_high_quality_patch(
            self.current_mesh, sketch_points, hole_strokes, self.plotter, canvas_size
        )

        if fitted_mesh:
            self.remove_preview()
            
            # オレンジ色で追加表示
            self.preview_actor = self.plotter.add_mesh(
                fitted_mesh,
                color="#FF6600",       # オレンジ
                opacity=1.0,
                smooth_shading=True,
                show_edges=False,
                name="fitted_sketch",
                specular=0.5,
                specular_power=15
            )
            
            # 最前面に表示設定
            mapper = self.preview_actor.GetMapper()
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
            mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2.0, -2.0)
            
            self._safe_render()
            return True
            
        return False
    
    def fit_trace_to_object(self, sketch_points, hole_strokes, canvas_size):
        if not self.current_mesh: return False
        
        # 1. 視点復元
        current_camera = self.plotter.camera_position
        if self.trace_camera_pos:
            self.plotter.camera_position = self.trace_camera_pos
        
        # 2. パッチ生成 (現在のプロッター設定で計算)
        fitted_mesh = SelectionOperator.create_high_quality_patch(
            self.current_mesh, sketch_points, hole_strokes, self.plotter, canvas_size
        )
        
        # 3. 視点を戻す
        self.plotter.camera_position = current_camera

        if fitted_mesh:
            self._show_patch(fitted_mesh)
            return True
        return False

    def _show_patch(self, mesh):
        self.remove_preview()
        
        # 法線計算
        try:
            if mesh.point_normals is None:
                mesh.compute_normals(inplace=True, auto_orient_normals=True)
        except: pass

        self.preview_actor = self.plotter.add_mesh(
            mesh, color="#FF6600", opacity=1.0, smooth_shading=True,
            show_edges=False, name="fitted_sketch", specular=0.5, specular_power=15
        )
        mapper = self.preview_actor.GetMapper()
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2.0, -2.0)
        self._safe_render()
    
    def extract_base_shape_from_sketch(self, sketch_points, canvas_size):
        """
        スケッチ輪郭に対応する現在のメッシュの形状データを取得する
        (SelectionOperatorに委譲)
        """
        if not self.current_mesh:
            return None

        return SelectionOperator.get_base_shape_from_sketch(
            self.current_mesh, sketch_points, self.plotter, canvas_size
        )