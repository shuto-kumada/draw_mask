from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QButtonGroup, QLabel, QCheckBox
)
from PyQt6.QtCore import pyqtSignal

class ObjectSidebar(QWidget):
    import_clicked = pyqtSignal()
    export_clicked = pyqtSignal()
    mode_changed = pyqtSignal(bool) # True=Trace, False=View
    cut_clicked = pyqtSignal()
    toggle_edges = pyqtSignal(bool)
    toggle_edl = pyqtSignal(bool)
    toggle_gizmo = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # ファイル
        layout.addWidget(QLabel("File Operations"))
        btn_import = QPushButton("Import Object File...")
        btn_import.clicked.connect(self.import_clicked.emit)
        layout.addWidget(btn_import)
        
        btn_export = QPushButton("Export Object File...")
        btn_export.clicked.connect(self.export_clicked.emit)
        layout.addWidget(btn_export)
        
        layout.addSpacing(20)
        
        # モード
        layout.addWidget(QLabel("Operation Mode"))
        self.mode_view_btn = QPushButton("View Mode (Rotate)")
        self.mode_view_btn.setCheckable(True); self.mode_view_btn.setChecked(True)
        self.mode_trace_btn = QPushButton("Trace Mode (Draw)")
        self.mode_trace_btn.setCheckable(True)
        
        self.op_group = QButtonGroup(self)
        self.op_group.addButton(self.mode_view_btn)
        self.op_group.addButton(self.mode_trace_btn)
        
        self.mode_view_btn.clicked.connect(lambda: self.mode_changed.emit(False))
        self.mode_trace_btn.clicked.connect(lambda: self.mode_changed.emit(True))
        
        layout.addWidget(self.mode_view_btn)
        layout.addWidget(self.mode_trace_btn)
        
        # 表示オプション
        layout.addSpacing(20)
        layout.addWidget(QLabel("View Options"))
        
        self.chk_edges = QCheckBox("Show Wireframe (Edges)")
        self.chk_edges.toggled.connect(self.toggle_edges.emit)
        layout.addWidget(self.chk_edges)
        
        self.chk_edl = QCheckBox("Enhanced Shading (EDL)")
        self.chk_edl.setChecked(False)
        self.chk_edl.toggled.connect(self.toggle_edl.emit)
        layout.addWidget(self.chk_edl)

        self.chk_gizmo = QCheckBox("Show Axes & Grid")
        self.chk_gizmo.setChecked(False)
        self.chk_gizmo.toggled.connect(self.toggle_gizmo.emit)
        layout.addWidget(self.chk_gizmo)

        layout.addStretch()