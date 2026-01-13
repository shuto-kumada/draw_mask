from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QButtonGroup, QLabel, QSlider, QHBoxLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
import os

class SketchSidebar(QWidget):
    # シグナル定義（MainWindowへイベントを伝える）
    tool_changed = pyqtSignal(str)
    param_changed = pyqtSignal()
    reset_clicked = pyqtSignal()
    save_clicked = pyqtSignal()
    run_clicked = pyqtSignal()
    fit_clicked = pyqtSignal()
    trace_fit_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # ツール
        self.tool_group = QButtonGroup(self)
        tools = [
            ("Boundary Fixed", "boundary_fixed"),
            ("Boundary Free", "boundary_free"),
            ("Deformation Pen", "deformation"), 
            ("Fixed Point", "fixed_point"),
            ("Hole", "hole"),
            ("Eraser", "eraser")
        ]
        layout.addWidget(QLabel("Tools"))
        for name, type_id in tools:
            btn = QPushButton(name)
            btn.setCheckable(True)
            if type_id == "boundary_fixed": btn.setChecked(True)
            # lambdaでシグナル発火
            btn.clicked.connect(lambda c, t=type_id: self.tool_changed.emit(t))
            self.tool_group.addButton(btn)
            layout.addWidget(btn)

        # パラメータ
        layout.addWidget(QLabel("Magnitude"))
        self.slider_mag = QSlider(Qt.Orientation.Horizontal)
        self.slider_mag.setRange(-100, 100); self.slider_mag.setValue(50)
        self.slider_mag.valueChanged.connect(self.param_changed.emit)
        layout.addWidget(self.slider_mag)

        layout.addWidget(QLabel("Profile"))
        self.slider_prof = QSlider(Qt.Orientation.Horizontal)
        self.slider_prof.setRange(0, 50); self.slider_prof.setValue(10)
        self.slider_prof.valueChanged.connect(self.param_changed.emit)
        layout.addWidget(self.slider_prof)

        layout.addWidget(QLabel("Influence"))
        self.slider_inf = QSlider(Qt.Orientation.Horizontal)
        self.slider_inf.setRange(1, 100); self.slider_inf.setValue(50)
        self.slider_inf.valueChanged.connect(self.param_changed.emit)
        layout.addWidget(self.slider_inf)
        
        layout.addStretch()
        
        # ボタン
        self.btn_reset = QPushButton("Reset Canvas")
        self.btn_reset.setStyleSheet("background-color: #ffcccc; color: darkred;")
        self.btn_reset.clicked.connect(self.reset_clicked.emit)
        layout.addWidget(self.btn_reset)

        self.btn_save = QPushButton("Save Sketch Image")
        self.btn_save.clicked.connect(self.save_clicked.emit)
        layout.addWidget(self.btn_save)

        self.btn_fit = QPushButton("Fit to Object >>")
        self.btn_fit.setStyleSheet("""
            QPushButton {
                background-color: #FFD700; color: black; font-weight: bold; 
                font-size: 14px; margin-top: 15px; text-align: center;
                border: 1px solid #b08d00; border-radius: 4px; padding: 8px;
            }
            QPushButton:hover { background-color: #FFC107; }
        """)
        self.btn_fit.clicked.connect(self.fit_clicked.emit)
        layout.addWidget(self.btn_fit)

        self.btn_trace_fit = QPushButton("Trace Fit >>")
        self.btn_trace_fit.setStyleSheet("""
            QPushButton {
                background-color: #FF5722; color: white; font-weight: bold; 
                font-size: 14px; margin-top: 5px; text-align: center;
                border: 1px solid #E64A19; border-radius: 4px; padding: 8px;
            }
            QPushButton:hover { background-color: #FF7043; }
            QPushButton:disabled { background-color: #555; color: #888; border: 1px solid #444; }
        """)
        self.btn_trace_fit.clicked.connect(self.trace_fit_clicked.emit)
        self.btn_trace_fit.setEnabled(False) # 初期状態は無効
        layout.addWidget(self.btn_trace_fit)

        self.btn_run = QPushButton("Generate 3D >>")
        self.btn_run.setObjectName("run_btn")
        self.btn_run.clicked.connect(self.run_clicked.emit)
        layout.addWidget(self.btn_run)

    def set_trace_fit_enabled(self, enabled):
        self.btn_trace_fit.setEnabled(enabled)