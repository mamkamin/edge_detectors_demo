from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap, QImage
import cv2
import numpy as np
import sys
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

WIN_TITLE = "Edge detectors Demo"
WIN_WIDTH = 1200
WIN_HEIGHT = 700
PANEL_MIN_SIZE = 300


def to_qimage(img):
    """
    Convert a cv Mat (grayscale or BGR) to QImage
    """
    if img.ndim == 2:
        h, w = img.shape
        return QImage(
            img.data, w, h, img.strides[0], QImage.Format.Format_Grayscale8
        ).copy()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()


def sobel_edges(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def canny_edges(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(
        blur, threshold1=50, threshold2=150, apertureSize=3, L2gradient=False
    )


class ImagePanel(QWidget):
    """
    A titled image area with a caption underneath (caption will hold the ms timing)
    """

    def __init__(self, title):
        super().__init__()
        self._pixmap = None

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold;")

        self.image_label = QLabel("No image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(PANEL_MIN_SIZE, PANEL_MIN_SIZE)
        # Ignored: the pixmap must not push the label bigger on every rescale
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        self.image_label.setStyleSheet("border: 1px solid #888;")

        self.caption_label = QLabel("")
        self.caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(self.caption_label)

    def set_image(self, img):
        self._pixmap = QPixmap.fromImage(to_qimage(img))
        self._rescale()

    def clear(self):
        self._pixmap = None
        self.image_label.clear()
        self.image_label.setText("No image")
        self.caption_label.setText("")

    def _rescale(self):
        if self._pixmap is None:
            return
        self.image_label.setPixmap(
            self._pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WIN_TITLE)
        self.resize(WIN_WIDTH, WIN_HEIGHT)

        self.image = None  # original (BGR), read once on load
        self.gray = None  # grayscale copy used by every detector
        self.results = {}  # detector name -> edge map

        self.original_panel = ImagePanel("Original")
        self.sobel_panel = ImagePanel("Sobel")
        self.canny_panel = ImagePanel("Canny")

        panels = QHBoxLayout()
        panels.addWidget(self.original_panel)
        panels.addWidget(self.sobel_panel)
        panels.addWidget(self.canny_panel)

        load_img_button = QPushButton("Load image")
        load_img_button.clicked.connect(self.load_image)

        sobel_button = QPushButton("Sobel Operator")
        sobel_button.clicked.connect(self.apply_sobel)

        canny_button = QPushButton("Canny Operator")
        canny_button.clicked.connect(self.apply_canny)

        pidinet_button = QPushButton("PiDiNet Operator")
        pidinet_button.clicked.connect(self.apply_pidinet)
        pidinet_button.setEnabled(False)  # enable once PiDiNet is wired in
        pidinet_button.setToolTip("Coming soon")

        buttons = QHBoxLayout()
        buttons.addWidget(load_img_button)
        buttons.addWidget(sobel_button)
        buttons.addWidget(canny_button)
        buttons.addWidget(pidinet_button)

        central = QWidget()
        root = QVBoxLayout(central)
        root.addLayout(panels, stretch=1)
        root.addLayout(buttons)
        self.setCentralWidget(central)

    @Slot()
    def apply_sobel(self):
        if self.gray is None:
            return
        edges = sobel_edges(self.gray)
        self.results["sobel"] = edges
        self.sobel_panel.set_image(edges)

    @Slot()
    def apply_canny(self):
        if self.gray is None:
            return
        edges = canny_edges(self.gray)
        self.results["canny"] = edges
        self.canny_panel.set_image(edges)

    @Slot()
    def apply_pidinet(self):
        pass

    @Slot()
    def load_image(self):
        """
        Load image once, keep a grayscale copy, and show it in the Original panel
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All files (*)"
        )
        if not path:
            return
        # imdecode + fromfile instead of imread: handles non-ASCII Windows paths
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        self.image = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.results = {}

        self.original_panel.set_image(self.image)
        self.sobel_panel.clear()
        self.canny_panel.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
