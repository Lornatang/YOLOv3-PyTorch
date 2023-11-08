# Copyright 2023 AlphaBetter Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import random
import warnings
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.backends import cudnn

from yolov3_pytorch.data.datasets import LoadImages, LoadStreams
from yolov3_pytorch.models.darknet import Darknet
from yolov3_pytorch.models.utils import load_state_dict
from yolov3_pytorch.utils.common import load_class_names_from_file, scale_coords, xyxy2xywh
from yolov3_pytorch.utils.nms import non_max_suppression
from yolov3_pytorch.utils.plots import plot_one_box


class Inferencer:
    def __init__(self, opts: argparse.Namespace):
        self.inputs = opts.inputs
        self.output = opts.output
        self.model_config_path = opts.model_config_path
        self.img_size = opts.img_size
        self.gray = opts.gray

        self.class_names = load_class_names_from_file(opts.class_names_path)
        self.num_classes = len(self.class_names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.class_names))]

        self.model_weights_path = opts.model_weights_path
        self.half = opts.half
        self.fuse = opts.fuse
        self.show_image = opts.show_image
        self.save_txt = opts.save_txt
        self.fourcc = opts.fourcc
        self.conf_thresh = opts.conf_thresh
        self.iou_thresh = opts.iou_thresh
        self.augment = opts.augment
        self.filter_classes = opts.filter_classes
        self.agnostic_nms = opts.agnostic_nms
        self.device = opts.device

        if self.device == "cpu":
            self.half = False
            self.device = torch.device("cpu")
        elif self.device == "gpu":
            if not torch.cuda.is_available():
                warnings.warn("CUDA is not available. Using CPU.")
                self.half = False
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda")
        else:
            raise ValueError(f"'{self.device}' is not supported.")

        # Load model and data
        if self.inputs.startswith("rtsp") or self.inputs.startswith("http"):
            self.detect_video = True
            self.detect_image = False
            self.save_image = False
            cudnn.benchmark = True
            self.dataset = LoadStreams(self.inputs, self.img_size, self.gray)
        else:
            self.detect_video = False
            self.detect_image = True
            self.save_image = True
            cudnn.benchmark = False
            self.dataset = LoadImages(self.inputs, self.img_size, self.gray)
        self.model = self.build_model()

        os.makedirs(self.output, exist_ok=True)

    def build_model(self) -> nn.Module:
        # Create model
        model = Darknet(self.model_config_path,
                        self.img_size,
                        self.gray,
                        False,
                        False)
        model = model.to(self.device)

        model.num_classes = self.num_classes

        # Load model weights
        if self.model_weights_path.endswith(".pth.tar"):
            state_dict = torch.load(self.model_weights_path, map_location=self.device)["state_dict"]
            model = load_state_dict(model, state_dict)
        elif self.model_weights_path.endswith(".weights"):
            model.load_darknet_weights(self.model_weights_path)
        else:
            raise ValueError(f"'{self.model_weights_path}' is not supported.")
        print(f"Loaded `{self.model_weights_path}` models weights successfully.")

        if self.half:
            model.half()

        if self.fuse:
            model.fuse()

        return model

    def predict(self) -> None:
        self.model.eval()

        for path, img, raw_img, video_capture in self.dataset:
            img = img.to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            with torch.no_grad():
                output, _ = self.model(img, self.augment)

            # For NMS
            if self.half:
                output = output.float()

            # Apply NMS
            output = non_max_suppression(output,
                                         self.conf_thresh,
                                         self.iou_thresh,
                                         False,
                                         self.filter_classes,
                                         self.agnostic_nms)

            # Process detections
            for detect_index, detect_result in enumerate(output):
                if self.detect_video:
                    path, results, raw_frame = path[detect_index], f"{detect_index}: ", raw_img[detect_index].copy()
                elif self.detect_image:
                    path, results, raw_frame = path, "", raw_img
                else:
                    path, results, raw_frame = path, "", raw_img

                save_path = str(Path(self.output) / Path(path).name)
                results += f"{img.shape[2]}x{img.shape[3]} "
                gn = torch.tensor(raw_frame.shape)[[1, 0, 1, 0]]
                if detect_result is not None and len(detect_result):
                    # Rescale boxes from image_size to raw_frame size
                    detect_result[:, :4] = scale_coords(img.shape[2:],
                                                        detect_result[:, :4],
                                                        raw_frame.shape).round()

                    # Print results
                    for c in detect_result[:, -1].unique():
                        number = (detect_result[:, -1] == c).sum()  # detections per class
                        results += f"{number} {self.class_names[int(c)]}, "

                    # Write results
                    for *xyxy, confidence, classes in reversed(detect_result):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(save_path[:save_path.rfind(".")] + ".txt", "a") as file:
                                file.write(("%g " * 5 + "\n") % (classes, *xywh))  # label format

                        if self.save_image or self.show_image:  # Add bbox to image
                            label = f"{self.class_names[int(classes)]} {confidence:.2f}"
                            plot_one_box(xyxy, raw_frame, label=label, color=self.colors[int(classes)])

                # Print result
                print(results)

                # Stream results
                if self.show_image:
                    cv2.imshow(path, raw_frame)
                    if cv2.waitKey(1) == ord("q"):
                        raise StopIteration

                # Save results (image with detections)
                if self.save_image:
                    if self.dataset.mode == "images":
                        cv2.imwrite(save_path, raw_frame)
                    else:
                        vid_writer.release()
                        fps = video_capture.get(cv2.CAP_PROP_FPS)
                        w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*self.fourcc), fps, (w, h))
                        vid_writer.write(raw_frame)
