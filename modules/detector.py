from __future__ import annotations

import logging
import os
import platform
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

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
    def __init__(self, debug: bool = True):
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self.weights = "yolov5n.pt"  # model path or triton URL
        self.data = "data/coco128.yaml"  # dataset.yaml path
        self.imgsz = (640, 640)  # inference size (height, width)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
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

    def detect_person_bbox(self, images: List[ndarray]) -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
        self._logger.debug('run Detector')
        im = _prepare_im(self, images)

        bbox = []
        class_id = []
        for image in im:
            pred = self.model(image, augment=False, visualize=False)

            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                       max_det=self.max_det)
            bbox.append([pred[0][i][0:4] for i in range(len(pred[0]))])
            class_id.append([pred[0][i][-1] for i in range(len(pred[0]))])

        return bbox, class_id


def _prepare_im(self, images: List) -> List[ndarray]:
    im_to_detect = images.copy()
    im_to_detect = [letterbox(np.asarray(im), 640, stride=32, auto=True)[0] for im in im_to_detect]
    im_to_detect = [im.transpose((2, 0, 1))[::-1] for im in im_to_detect]
    im_to_detect = [np.ascontiguousarray(im) for im in im_to_detect]
    im_to_detect = [torch.from_numpy(im).to(self.model.device) for im in im_to_detect]
    im_to_detect = [im.half() if self.model.fp16 else im.float() for im in im_to_detect]
    im_to_detect = np.asarray([im / 255 for im in im_to_detect])
    if len(im_to_detect.shape) == 3:
        im_to_detect = im_to_detect[None]
    return im_to_detect
