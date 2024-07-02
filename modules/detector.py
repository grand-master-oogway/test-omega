from __future__ import annotations

import logging
import os
import platform
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors

from modules.utils.augmentations import letterbox
from models.common import DetectMultiBackend
from modules.utils.general import (
    LOGGER,
    Profile,
    check_img_size,
    cv2,
    non_max_suppression,
    scale_boxes,
)
from modules.utils.torch_utils import select_device


class Detector:
    weights = "yolov5n.pt"  # model path or triton URL
    data = "data/coco128.yaml"  # dataset.yaml path
    imgsz = (640, 640)  # inference size (height, width)
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    device = ""  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img = False  # show results
    save_txt = False  # save results to *.txt
    save_csv = False  # save results in CSV format
    save_conf = False  # save confidences in --save-txt labels
    save_crop = False  # save cropped prediction boxes
    nosave = False  # do not save images/videos
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    augment = False  # augmented inference
    visualize = False  # visualize features
    update = False  # update all models
    project = ROOT / "runs/detect"  # save results to project/name
    name = "exp"  # save results to project/name
    exist_ok = False  # existing project/name ok, do not increment
    line_thickness = 3  # bounding box thickness (pixels)
    hide_labels = False  # hide labels
    hide_conf = False  # hide confidences
    half = False  # use FP16 half-precision inference
    dnn = False  # use OpenCV DNN for ONNX inference
    vid_stride = 1  # video frame-rate stride
    auto = True

    def __init__(self, frames: List, ids: List, debug: bool = True):
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self.frames = frames
        self.ids = ids

    def run(self) -> None:
        self._logger.debug('run Detector')

        # save_img = not self.nosave and not self.source.endswith(".txt")  # save inference images

        """load model"""
        device = select_device(self.device)
        model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        bs = 1
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

        for frame, _id in zip(self.frames, self.ids):
            if isinstance(frame, np.ndarray):
                im0s = frame
                im = letterbox(frame, 640, stride=32, auto=True)[0]  # padded resize
                im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                im = np.ascontiguousarray(im)  # contiguous

                with dt[0]:
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    if model.xml and im.shape[0] > 1:
                        ims = torch.chunk(im, im.shape[0], 0)

                # Inference
                with dt[1]:
                    pred = model(im, augment=self.augment, visualize=self.visualize)
                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                               max_det=self.max_det)

                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1

                    im0, frame = im0s.copy(), _id

                    imc = im0.copy()
                    annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)


                            label = None if self.hide_labels else (names[c] if self.hide_conf else f"{names[c]} {conf:.2f}")


                            print('xyxy -> ', xyxy)
                            print('label -> ', label)
                            print('id -> ', _id)

                            annotator.box_label(xyxy, label, color=colors(c, True))

                    # Stream results
                    im0 = annotator.result()
                    # print("im0 -->", im0)
                    return im0
                    # if view_img:
                    #     if platform.system() == "Linux" and p not in windows:
                    #         windows.append(p)
                    #         cv2.namedWindow(str(p),
                    #                         cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    #         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    #     cv2.imshow(str(p), im0)
                    #     cv2.waitKey(1)  # 1 millisecond
