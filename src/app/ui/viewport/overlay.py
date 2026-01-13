from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt

class ConfirmOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.hide()
        
        self.setStyleSheet("""
            QWidget#OverlayContainer {
                background-color: rgba(30, 30, 30, 220);
                border-radius: 10px;
                border: 1px solid #555;
            }
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 5px;
            }
            QPushButton {
                color: white; 
                font-weight: bold; 
                font-size: 12px;
                border: none; 
                border-radius: 4px; 
                padding: 8px 16px;
                min-width: 80px;
            }
            QPushButton:hover { opacity: 0.8; }
            QPushButton#btn_yes { background-color: #4CAF50; }     /* 緑 */
            QPushButton#btn_no { background-color: #f44336; }      /* 赤 */
        """)

        # コンテナウィジェット（スタイル適用用）
        container = QWidget(self)
        container.setObjectName("OverlayContainer")
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(container)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # メッセージ
        self.lbl_message = QLabel("領域を確定しますか？")
        self.lbl_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_message)
        
        # ボタンレイアウト
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        
        self.btn_confirm = QPushButton("Yes")
        self.btn_confirm.setObjectName("btn_confirm")
        
        self.btn_retry = QPushButton("No (Retry)")
        self.btn_retry.setObjectName("btn_retry")

        btn_layout.addWidget(self.btn_confirm)
        btn_layout.addWidget(self.btn_retry)

        layout.addLayout(btn_layout)

    def show_message(self, text):
        self.lbl_message.setText(text)
        self.adjustSize()
        self.show()
        self.raise_()

    def update_position(self, parent_rect):
        """親ウィジェットの中央下部に配置"""
        self.adjustSize()
        w = self.width()
        h = self.height()
        # 中央下
        self.move((parent_rect.width() - w) // 2, parent_rect.height() - h - 30)