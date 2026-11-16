from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap
import torch
import sys
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
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

        self.image_label = QLabel("No Image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        load_img_button = QPushButton("Load image")
        load_img_button.clicked.connect(self.load_image)
        sobel_button = QPushButton("Sobel Operator")
        canny_button = QPushButton("Canny Operator")
        pidinet_button = QPushButton("PiDiNet Operator")

        grid.addWidget(
            self.image_label, 0, 0, 4, 1, alignment=Qt.AlignmentFlag.AlignCenter
        )
        grid.addWidget(load_img_button, 0, 1)
        grid.addWidget(sobel_button, 1, 1)
        grid.addWidget(canny_button, 2, 1)
        grid.addWidget(pidinet_button, 3, 1)
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 1)

        self.setCentralWidget(central)

    @Slot()
    def load_image(self):
        """
        Load image and show it in label
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All files (*)"
        )
        if not path:
            return
        pix = QPixmap(path)
        self.image_label.setPixmap(
            pix.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
