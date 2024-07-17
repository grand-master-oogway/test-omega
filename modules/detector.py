from __future__ import annotations

import os
import sys
import torch
import logging
import numpy as np
from torch import Tensor
from pathlib import Path
from typing import List, Tuple, Union
from objects.model_config import ModelConfig

from objects import DetectedObject

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from libs.utils.augmentations import letterbox
from libs.models.common import DetectMultiBackend
from libs.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes
)
from libs.utils.torch_utils import select_device


class Detector:
    def __init__(self, model_config: ModelConfig, debug: bool = True):
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self.weights = model_config.weights
        self.imgsz = model_config.imgsz
        self.conf_thres = model_config.conf_thres
        self.iou_thres = model_config.iou_thres
        self.max_det = model_config.max_det
        self.classes = model_config.classes
        self.agnostic_nms = model_config.agnostic_nms
        self.half = model_config.half
        self.dnn = model_config.dnn
        self.device = select_device("cpu")
        self.model = DetectMultiBackend(weights=self.weights, device=self.device, dnn=self.dnn)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt

    def get_bbox(self, images: List, ids: List) -> Union[List[List[DetectedObject]], None]:
        self._logger.debug('run Detector')
        if len(np.asarray(images).shape) == 1:
            return None
        images_to_detect = self._prepare_im(images)
        pred = self.model(images_to_detect)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)

        bboxes: List[List[DetectedObject]] = []

        for i, det in enumerate(pred):
            detected_obj: List[DetectedObject] = []
            im0 = images[i]
            if len(det):
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                det[:, :4] = scale_boxes(images_to_detect.shape[2:], det[:, :4], im0.shape).round()

                for j in range(det.size()[0]):
                    x1, y1, x2, y2 = self._convert_to_orig_shape((det[j, :4].view(1, 4) / gn).view(-1).tolist(),
                                                                 im0.shape[:2])
                    confidence = round(float(det[j][-2]), 2)
                    cls = det[j][-1]
                    c = int(cls)
                    label = self.names[c]
                    centroid = [int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)]

                    detected_obj.append(
                        DetectedObject(bbox=[x1, y1, x2, y2], class_name=label, conf=confidence, centroid=centroid,
                                       source_id=ids[i]))

            bboxes.append(detected_obj)
        return bboxes

    def _prepare_im(self, images: List) -> Tensor:
        im_to_detect = images.copy()
        im_to_detect = np.stack(
            [letterbox(np.asarray(im), self.imgsz, stride=self.stride, auto=False)[0] for im in im_to_detect], 0)
        im_to_detect = im_to_detect[..., ::-1].transpose((0, 3, 1, 2))
        im_to_detect = np.ascontiguousarray(im_to_detect)
        im_to_detect = torch.from_numpy(im_to_detect).to(self.device).half()
        im_to_detect = im_to_detect.float() / 255
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
