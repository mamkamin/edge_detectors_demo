from PySide6.QtCore import Qt
import torch
import sys
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

WIN_TITLE = "Edge detectors Demo"
WIN_WIDTH = 1200
WIN_HEIGHT = 700


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WIN_TITLE)
        self.resize(WIN_WIDTH, WIN_HEIGHT)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(
            QLabel(f"Cuda available: {torch.cuda.is_available()}"),
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        self.setCentralWidget(central)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
