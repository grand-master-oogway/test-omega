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
    """
    Class for detecting objects
    """
    def __init__(self, model_config: ModelConfig, debug: bool = True):

        """
        Constructor for Detector class.

        Args:
            model_config (ModelConfig): Configuration object for the model.
            debug (bool): Flag for enabling debug logging. Default is True.
        """
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self._logger.debug('initialize Detector')

        self.weights: str = model_config.weights
        self.imgsz: Tuple[int, int] = model_config.imgsz
        self.conf_thres: float = model_config.conf_thres
        self.iou_thres: float = model_config.iou_thres
        self.max_det: int = model_config.max_det
        self.classes: Union[int, None] = model_config.classes
        self.agnostic_nms: bool = model_config.agnostic_nms
        self.half: bool = model_config.half
        self.dnn: bool = model_config.dnn
        self.device: torch.device = select_device("cpu")
        self.model: DetectMultiBackend = DetectMultiBackend(weights=self.weights, device=self.device, dnn=self.dnn)
        self.stride: int = self.model.stride
        self.names: List[str] = self.model.names
        self.pt: bool = self.model.pt

    def get_bbox(self, images: List[np.ndarray], ids: List[int]) -> Union[List[List[DetectedObject]], None]:
        """
        Get bounding boxes for the given images.

        Args:
            images (List[np.ndarray]): List of images to process.
            ids (List[int]): List of IDs corresponding to the images.

        Returns:
            Union[List[List[DetectedObject]], None]: List of lists containing detected objects.
        """
        if len(np.asarray(images).shape) == 1:
            return None

        images_to_detect: Tensor = self._prepare_im(images)
        pred: List[Tensor] = self.model(images_to_detect)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)

        bboxes: List[List[DetectedObject]] = []

        for i, det in enumerate(pred):
            detected_objects: List[DetectedObject] = []
            im0: np.ndarray = images[i]
            if len(det):
                gn: Tensor = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                det[:, :4] = scale_boxes(images_to_detect.shape[2:], det[:, :4], im0.shape).round()

                for j in range(det.size()[0]):
                    x1, y1, x2, y2 = self._convert_to_orig_shape((det[j, :4].view(1, 4) / gn).view(-1).tolist(),
                                                                 im0.shape[:2])
                    confidence: float = round(float(det[j][-2]), 2)
                    cls: Tensor = det[j][-1]
                    c: int = int(cls)
                    label: str = self.names[c]
                    centroid: List[int] = [int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)]

                    detected_objects.append(
                        DetectedObject(bbox=np.asarray([x1, y1, x2, y2], dtype=int), class_name=label, conf=confidence,
                                       centroid=centroid, source_id=ids[i]))

            bboxes.append(detected_objects)
        return bboxes

    def _prepare_im(self, images: List[np.ndarray]) -> Tensor:
        """
        Prepare images for detection.

        Args:
            images (List[np.ndarray]): List of images to process.

        Returns:
            Tensor: Tensor of prepared images.
        """
        im_to_detect: np.ndarray = np.stack(
            [letterbox(np.asarray(im), self.imgsz, stride=self.stride, auto=False)[0] for im in images], 0)
        im_to_detect = im_to_detect[..., ::-1].transpose((0, 3, 1, 2))
        im_to_detect = np.ascontiguousarray(im_to_detect)
        im_to_detect = torch.from_numpy(im_to_detect).to(self.device)
        if self.half:
            im_to_detect = im_to_detect.half()  # Convert to FP16
        else:
            im_to_detect = im_to_detect.float()  # Convert to FP32
        im_to_detect /= 255.0  # Normalize to [0, 1]

        if len(im_to_detect.shape) == 3:
            im_to_detect = im_to_detect.unsqueeze(0)

        return im_to_detect

    def _convert_to_orig_shape(self, bbox: List[float], orig_shape: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """
        Convert bounding box to original image shape.

        Args:
            bbox (List[float]): Bounding box coordinates.
            orig_shape (Tuple[int, int]): Original image shape.

        Returns:
            Tuple[float, float, float, float]: Converted bounding box coordinates.
        """
        height: int
        width: int
        height, width = orig_shape
        x1: float
        y1: float
        x2: float
        y2: float
        x1, y1, x2, y2 = bbox
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height
        return x1, y1, x2, y2
