"""
Capture a screenshot of the edge detectors demo for a given image.

Usage:
    python tools/capture_screenshot.py IMAGE_PATH
    python tools/capture_screenshot.py IMAGE_PATH --repeat 10

With --repeat N, also times each detector N more times after warm-up and
prints the median ms (same timed() and detector functions the captions use).

If a tiling window manager forces the window to another size, render it
offscreen instead (same widgets, exact 1600x600):
    QT_QPA_PLATFORM=offscreen python tools/capture_screenshot.py IMAGE_PATH
"""

import argparse
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from main import MainWindow, canny_edges, sobel_edges, timed  # noqa: E402

SCREENSHOT_DIR = ROOT / "screenshots"
WINDOW_SIZE = (1600, 600)
SETTLE_MS = 500


def load_image(window, path):
    """
    Same as MainWindow.load_image(), but with the path given instead of a dialog
    """
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {path}")
    window.image = img
    window.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    window.results = {}

    window.original_panel.set_image(window.image)
    window.sobel_panel.clear()
    window.canny_panel.clear()
    window.pidinet_panel.clear()


def run_detectors(window):
    window.apply_sobel()
    window.apply_canny()
    window.apply_pidinet()


def median_timings(window, repeat):
    """
    Time each detector `repeat` times on the loaded image (after warm-up)
    """
    detectors = {
        "Sobel": (sobel_edges, window.gray),
        "Canny": (canny_edges, window.gray),
        "PiDiNet": (window.pidinet.detect, window.image),
    }
    for name, (detect, image) in detectors.items():
        times = [timed(detect, image)[1] for _ in range(repeat)]
        runs = ", ".join(f"{t:.1f}" for t in times)
        print(f"{name:8s} median {statistics.median(times):.1f} ms  (runs: {runs})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="path to the input image")
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="also time each detector N times and print the median ms",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    out_path = SCREENSHOT_DIR / f"{image_path.stem}_app.png"

    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(*WINDOW_SIZE)
    window.show()
    app.processEvents()

    load_image(window, image_path)
    run_detectors(window)  # warm-up (first-run timings are not representative)
    run_detectors(window)  # steady-state timings shown in the captions

    for _ in range(5):
        app.processEvents()

    def capture():
        app.processEvents()
        size = window.size()
        if (size.width(), size.height()) != WINDOW_SIZE:
            print(
                f"Warning: window is {size.width()}x{size.height()}, "
                f"not {WINDOW_SIZE[0]}x{WINDOW_SIZE[1]} (window manager?)",
                file=sys.stderr,
            )
        SCREENSHOT_DIR.mkdir(exist_ok=True)
        if not window.grab().save(str(out_path), "PNG"):
            print(f"Failed to save screenshot to {out_path}", file=sys.stderr)
        else:
            print(f"Saved {out_path.relative_to(ROOT)}")
        print(f"Sobel:   {window.sobel_panel.caption_label.text()}")
        print(f"Canny:   {window.canny_panel.caption_label.text()}")
        print(f"PiDiNet: {window.pidinet_panel.caption_label.text()}")

        if args.repeat > 0:
            print(f"\nMedian over {args.repeat} runs on {image_path.name}:")
            median_timings(window, args.repeat)

        app.quit()

    def enforce_size():
        # Tiling window managers re-place new windows (e.g. half-screen) shortly
        # after show(), overriding the resize above; resize again once they have
        window.resize(*WINDOW_SIZE)
        for _ in range(5):
            app.processEvents()
        QTimer.singleShot(SETTLE_MS, capture)

    QTimer.singleShot(SETTLE_MS, enforce_size)
    app.exec()


if __name__ == "__main__":
    main()
