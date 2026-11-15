from PySide6.QtCore import Qt
import torch
import sys
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QWidget,
)

WIN_TITLE = "Edge detectors Demo"
WIN_WIDTH = 1200
WIN_HEIGHT = 700


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WIN_TITLE)
        self.resize(WIN_WIDTH, WIN_HEIGHT)

        central = QWidget()
        grid = QGridLayout(central)

        label = QLabel(f"Cuda available: {torch.cuda.is_available()}")
        load_img_button = QPushButton("Load image")
        sobel_button = QPushButton("Sobel Operator")
        canny_button = QPushButton("Canny Operator")
        pidinet_button = QPushButton("PiDiNet Operator")

        grid.addWidget(label, 0, 0, 4, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(load_img_button, 0, 1)
        grid.addWidget(sobel_button, 1, 1)
        grid.addWidget(canny_button, 2, 1)
        grid.addWidget(pidinet_button, 3, 1)
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 1)

        self.setCentralWidget(central)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
