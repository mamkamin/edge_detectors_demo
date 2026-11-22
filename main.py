from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap, QImage
import cv2
import numpy as np
import sys
import time
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

from pidinet_detector import PiDiNetDetector

WIN_TITLE = "Edge detectors Demo"
WIN_WIDTH = 1200
WIN_HEIGHT = 700
PANEL_MIN_SIZE = 300

SOBEL_KSIZE = 3
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150


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
    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=SOBEL_KSIZE)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=SOBEL_KSIZE)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def canny_edges(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(
        blur,
        threshold1=CANNY_LOW_THRESHOLD,
        threshold2=CANNY_HIGH_THRESHOLD,
        apertureSize=3,
        L2gradient=False,
    )


def timed(detect, image):
    """
    Run a detector and return (edge map, elapsed time in ms)
    """
    start = time.perf_counter()
    edges = detect(image)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return edges, elapsed_ms


class ImagePanel(QWidget):
    """
    A titled image area with a caption underneath (detector settings and timing)
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

    def set_caption(self, text):
        self.caption_label.setText(text)

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
        self.pidinet = None  # PiDiNetDetector, loaded on first use (slow to load)

        self.original_panel = ImagePanel("Original")
        self.sobel_panel = ImagePanel("Sobel")
        self.canny_panel = ImagePanel("Canny")
        self.pidinet_panel = ImagePanel("PiDiNet")

        panels = QHBoxLayout()
        panels.addWidget(self.original_panel)
        panels.addWidget(self.sobel_panel)
        panels.addWidget(self.canny_panel)
        panels.addWidget(self.pidinet_panel)

        load_img_button = QPushButton("Load image")
        load_img_button.clicked.connect(self.load_image)

        sobel_button = QPushButton("Sobel Operator")
        sobel_button.clicked.connect(self.apply_sobel)

        canny_button = QPushButton("Canny Operator")
        canny_button.clicked.connect(self.apply_canny)

        pidinet_button = QPushButton("PiDiNet Operator")
        pidinet_button.clicked.connect(self.apply_pidinet)

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
        edges, ms = timed(sobel_edges, self.gray)
        self.results["sobel"] = edges
        self.sobel_panel.set_image(edges)
        # Sobel has no threshold: it shows the normalized gradient magnitude
        self.sobel_panel.set_caption(
            f"Kernel {SOBEL_KSIZE}x{SOBEL_KSIZE}, no threshold  |  {ms:.1f} ms"
        )

    @Slot()
    def apply_canny(self):
        if self.gray is None:
            return
        edges, ms = timed(canny_edges, self.gray)
        self.results["canny"] = edges
        self.canny_panel.set_image(edges)
        self.canny_panel.set_caption(
            f"Thresholds {CANNY_LOW_THRESHOLD} / {CANNY_HIGH_THRESHOLD}  |  {ms:.1f} ms"
        )

    @Slot()
    def apply_pidinet(self):
        if self.image is None:
            return
        if self.pidinet is None:
            self.pidinet = PiDiNetDetector()
        # PiDiNet is a CNN trained on color images, so it takes the BGR original
        edges, ms = timed(self.pidinet.detect, self.image)
        self.results["pidinet"] = edges
        self.pidinet_panel.set_image(edges)
        # PiDiNet has no threshold: it shows the raw edge probability map
        device = self.pidinet.device.type.upper()
        self.pidinet_panel.set_caption(f"No threshold, {device}  |  {ms:.1f} ms")

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
        self.pidinet_panel.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
