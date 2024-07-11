from __future__ import annotations

import os
import sys
import torch
import logging
import numpy as np
from pathlib import Path
from numpy import ndarray
from typing import List, Tuple
from objects import DetectedObject

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from libs.utils.augmentations import letterbox
from libs.models.common import DetectMultiBackend
from libs.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes
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
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
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

    def get_bbox(self, images: List[ndarray], ids: List) -> List[List[DetectedObject]]:
        self._logger.debug('run Detector')
        im = self._prepare_im(images)

        pred = self.model(im, augment=False, visualize=False)

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)

        bboxes: List[List[DetectedObject]] = []

        for i, det in enumerate(pred):
            detected_obj: List[DetectedObject] = []
            im0 = images[i]
            if len(det):
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for j in range(det.size()[0]):
                    x1, y1, x2, y2 = self._convert_to_orig_shape((det[j, :4].view(1, 4) / gn).view(-1).tolist(),
                                                                 im0.shape[:2])
                    confidence = round(det[j][-2], 2)
                    cls = det[j][-1]
                    c = int(cls)
                    label = self.names[c]
                    centroid = [(x1 + x2) // 2.0, (y1 + y2) // 2.0]

                    detected_obj.append(
                        DetectedObject(bbox=[x1, y1, x2, y2], class_name=label, conf=confidence, centroid=centroid,
                                       source_id=ids[i]))

            bboxes.append(detected_obj)
        return bboxes

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

    def _convert_to_orig_shape(self, bbox: List[float], orig_shape: Tuple[int, int]) -> Tuple[
        float, float, float, float]:
        height, width = orig_shape
        x1, y1, x2, y2 = bbox
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height
        return x1, y1, x2, y2
