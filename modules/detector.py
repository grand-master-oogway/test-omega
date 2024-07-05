from __future__ import annotations

import logging
import os
import platform
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from numpy import ndarray

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors

from libs.utils.augmentations import letterbox
from libs.models.common import DetectMultiBackend
from libs.utils.general import (
    LOGGER,
    Profile,
    check_img_size,
    cv2,
    non_max_suppression,
    scale_boxes,
)
from libs.utils.torch_utils import select_device


class Detector:
    def __init__(self, frames: List, ids: List, debug: bool = True):
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self.frames = frames
        self.ids = ids

        self.weights = "yolov5n.pt"  # model path or triton URL
        self.data = "data/coco128.yaml"  # dataset.yaml path
        self.imgsz = (640, 640)  # inference size (height, width)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        view_img = False  # show results
        self.classes = 0  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference

        """load & prepare model"""

        self.device = select_device("")
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else 1, 3, *imgsz))  # warmup

    def detect_person_bbox(self) -> List:
        self._logger.debug('run Detector')

        im = _prepare_im(self, self.frames)

        bbox = []
        for image in im:
            pred = self.model(image, augment=False, visualize=False)

            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                       max_det=self.max_det)
            bbox.append(pred[0])

        return bbox


def _prepare_im(self, frames: List) -> ndarray:
    im = [letterbox(np.asarray(frame), 640, stride=32, auto=True)[0] for frame in frames]
    im = [image.transpose((2, 0, 1))[::-1] for image in im]
    im = [np.ascontiguousarray(image) for image in im]
    im = [torch.from_numpy(image).to(self.model.device) for image in im]
    im = [image.half() if self.model.fp16 else image.float() for image in im]
    im = np.asarray([image / 255 for image in im])
    return im
