"""
PiDiNet edge detection, wrapping the official implementation in ./pidinet
(https://github.com/hellozhuo/pidinet).
"""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from pidinet.models import pidinet_converted
from pidinet.models.convert_pidinet import convert_pidinet

CHECKPOINT_PATH = (
    Path(__file__).parent / "pidinet" / "trained_models" / "table5_pidinet.pth"
)

# table5_pidinet was trained with: --config carv4 --sa --dil
MODEL_ARGS = SimpleNamespace(config="carv4", sa=True, dil=True)

# ImageNet statistics, the same normalization PiDiNet was trained with
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PiDiNetDetector:
    """
    Loads the pretrained PiDiNet once and runs it on BGR images.

    Uses the "converted" model: the pixel difference convolutions are folded
    into plain convolutions, which gives identical output but runs faster.
    """

    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path):
        model = pidinet_converted(MODEL_ARGS)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # Checkpoint was saved from nn.DataParallel, so keys start with "module."
        state_dict = {
            key.removeprefix("module."): value
            for key, value in checkpoint["state_dict"].items()
        }
        model.load_state_dict(convert_pidinet(state_dict, MODEL_ARGS.config))

        return model.to(self.device).eval()

    def _to_tensor(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - MEAN) / STD
        chw = normalized.transpose(2, 0, 1)  # HWC -> CHW
        return torch.from_numpy(chw).unsqueeze(0).to(self.device)  # add batch dim

    @torch.no_grad()
    def detect(self, bgr):
        """
        Return an 8-bit edge map (white edges on black) for a BGR image
        """
        outputs = self.model(self._to_tensor(bgr))
        # The model returns 4 side outputs plus the fused map; the fused one is last
        edge_prob = outputs[-1].squeeze().cpu().numpy()
        return (edge_prob * 255).astype(np.uint8)
