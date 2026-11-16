from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap, QImage
import cv2
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

        self.current_path = ""
        self.image_label = QLabel("No Image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(600, 400)

        load_img_button = QPushButton("Load image")
        load_img_button.clicked.connect(self.load_image)

        sobel_button = QPushButton("Sobel Operator")
        sobel_button.clicked.connect(self.apply_sobel)

        canny_button = QPushButton("Canny Operator")
        canny_button.clicked.connect(self.apply_canny)

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

    def show_image(self, img):
        """
        Convert image from cv Mat to QImage and render it
        """
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()
        self.image_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(self.image_label.size(), Qt.KeepAspectRatio)
        )

    @Slot()
    def apply_sobel(self):
        if not self.current_path:
            return
        gray = cv2.imread(self.current_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        mag = cv2.magnitude(gx, gy)
        edges = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        self.result = edges
        self.show_image(edges)

    @Slot()
    def apply_canny(self):
        if not self.current_path:
            return
        gray = cv2.imread(self.current_path, cv2.IMREAD_GRAYSCALE)

        if gray is None:
            return

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(
            blur, threshold1=50, threshold2=150, apertureSize=3, L2gradient=False
        )
        self.result = edges
        self.show_image(edges)

    @Slot()
    def apply_pidinet(self):
        pass

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
        self.current_path = path
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
