# Copyright 2020 Lorna Authors. All Rights Reserved.
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
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.backends import cudnn

import model
from dataset import LoadImages, LoadStreams
from utils import make_directory, non_max_suppression, plot_one_box, scale_coords, xyxy2xywh


def main(args):
    # Detect result save address
    detect_result_dir = os.path.join("results", "detect", args.detect_results_name)
    make_directory(detect_result_dir)

    # Load data
    if args.inputs.startswith("rtsp") or args.inputs.startswith("http"):
        detect_video = True
        detect_image = False
        save_image = False
        cudnn.benchmark = True
        dataset = LoadStreams(args.inputs, args.image_size, args.gray)
    else:
        detect_video = False
        detect_image = True
        save_image = True
        cudnn.benchmark = False
        dataset = LoadImages(args.inputs, args.image_size, args.gray)

    # Load class names
    with open(args.names_file_path, "r") as f:
        names = f.read().split('\n')
    names = list(filter(None, names))
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Build model
    device = choice_device(args.device)
    yolo_model = build_model(args.model_arch_name,
                             args.image_size,
                             False,
                             args.model_weights_path,
                             device,
                             args.half,
                             args.fuse)

    detect(yolo_model,
           dataset,
           args.half,
           names,
           colors,
           detect_video,
           detect_image,
           args.show_image,
           save_image,
           args.save_txt,
           args.fourcc,
           detect_result_dir,
           args.conf_threshold,
           args.iou_threshold,
           args.augment,
           args.filter_classes,
           args.agnostic_nms,
           device)


def choice_device(device: str = "cpu") -> torch.device:
    # Select model processing equipment type
    if device == "cuda":
        device = torch.device("cuda", 0)
    elif device[:4] == "cuda":
        device = torch.device(device)
    else:
        device = torch.device("cpu")

    return device


def build_model(
        model_arch_name: str,
        test_image_size: int | tuple[int, int],
        gray: bool = False,
        model_weights_path: str = None,
        device: torch.device = torch.device("cpu"),
        half: bool = False,
        fuse: bool = False,
) -> nn.Module:
    """Initialize YOLO model

    Args:
        model_arch_name (str): Model architecture name
        test_image_size (int | tuple[int, int]): Test image size
        gray (bool, optional): Whether to use grayscale images. Default: ``False``.
        model_weights_path (str, optional): Model weights path. Default: ``None``.
        device (torch.device, optional): Model processing equipment. Default: ``torch.device("cpu")``.
        half (bool, optional): Whether to use half precision. Default: ``False``.
        fuse (bool, optional): Whether to fuse model. Default: ``False``.

    Returns:
        yolo_model (nn.Module): YOLO model

    """
    # Create model
    yolo_model = model.__dict__[model_arch_name](image_size=test_image_size, gray=gray)
    # Load the pre-trained model weights
    if model_weights_path.endswith(".pth.tar"):
        yolo_model = model.load_torch_weights(yolo_model, model_weights_path)
    elif model_weights_path.endswith(".weights"):
        model.load_darknet_weights(yolo_model, model_weights_path)
    else:
        raise "The model weights path is not correct."
    print(f"Loaded `{model_weights_path}` pretrained model weights successfully.")

    yolo_model = yolo_model.to(device=device)

    if half:
        yolo_model.half()

    if fuse:
        yolo_model.fuse()

    return yolo_model


def detect(
        yolo_model: nn.Module,
        dataset: LoadStreams or LoadImages,
        half: bool = False,
        names: list[str] = None,
        colors: list[list[int]] = None,
        detect_video: bool = False,
        detect_image: bool = False,
        show_image: bool = False,
        save_image: bool = False,
        save_txt: bool = False,
        fourcc: str = "mp4v",
        detect_results_dir: str = None,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        augment: bool = False,
        filter_classes: list[int] = None,
        agnostic_nms: bool = False,
        device: torch.device = torch.device("cpu"),
) -> None:
    """Detect

    Args:
        yolo_model (nn.Module): YOLO model
        dataset (LoadStreams or LoadImages): Dataset
        half (bool, optional): Whether to use half precision. Default: ``False``.
        names (list[str], optional): Class names. Default: ``None``.
        colors (list[list[int]], optional): Class colors. Default: ``None``.
        detect_video (bool, optional): Whether to detect video. Default: ``False``.
        detect_image (bool, optional): Whether to detect image. Default: ``False``.
        show_image (bool, optional): Whether to show image. Default: ``False``.
        save_image (bool, optional): Whether to save image. Default: ``False``.
        save_txt (bool, optional): Whether to save txt. Default: ``False``.
        fourcc (str, optional): Video codec. Default: ``"mp4v"``.
        detect_results_dir (str, optional): Detect result directory. Default: ``None``.
        conf_threshold (float, optional): Confidence threshold. Default: ``0.001``.
        iou_threshold (float, optional): IoU threshold. Default: ``0.5``.
        augment (bool, optional): Whether to use data augmentation. Default: ``False``.
        filter_classes (list[int], optional): Filter classes. Default: ``None``.
        agnostic_nms (bool, optional): Whether to use agnostic nms. Default: ``False``.
        device (torch.device, optional): Model processing equipment. Default: ``torch.device("cpu")``.

    Returns:

    """
    yolo_model.eval()

    for input_path, image, raw_image, video_capture in dataset:
        image = image.to(device)
        image = image.half() if half else image.float()
        image /= 255.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        # Inference
        with torch.no_grad():
            output = yolo_model(image, augment=augment)[0]

        # For NMS
        if half:
            output = output.float()

        # Apply NMS
        output = non_max_suppression(output,
                                     conf_threshold,
                                     iou_threshold,
                                     False,
                                     filter_classes,
                                     agnostic_nms)

        # Process detections
        for detect_index, detect_result in enumerate(output):
            if detect_video:
                path, results, raw_frame = input_path[detect_index], f"{detect_index}: ", raw_image[detect_index].copy()
            elif detect_image:
                path, results, raw_frame = input_path, "", raw_image
            else:
                path, results, raw_frame = input_path, "", raw_image

            save_path = str(Path(detect_results_dir) / Path(path).name)
            results += f"{image.shape[2]}x{image.shape[3]} "
            gn = torch.tensor(raw_frame.shape)[[1, 0, 1, 0]]
            if detect_result is not None and len(detect_result):
                # Rescale boxes from image_size to raw_frame size
                detect_result[:, :4] = scale_coords(image.shape[2:],
                                                    detect_result[:, :4],
                                                    raw_frame.shape).round()

                # Print results
                for c in detect_result[:, -1].unique():
                    n = (detect_result[:, -1] == c).sum()  # detections per class
                    results += f"{n} {names[int(c)]}, "

                # Write results
                for *xyxy, confidence, classes in reversed(detect_result):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind(".")] + ".txt", "a") as file:
                            file.write(("%g " * 5 + "\n") % (classes, *xywh))  # label format

                    if save_image or show_image:  # Add bbox to image
                        label = f"{names[int(classes)]} {confidence:.2f}"
                        plot_one_box(xyxy, raw_frame, label=label, color=colors[int(classes)])

            # Print result
            print(results)

            # Stream results
            if show_image:
                cv2.imshow(path, raw_frame)
                if cv2.waitKey(1) == ord("q"):
                    raise StopIteration

            # Save results (image with detections)
            if save_image:
                if dataset.mode == "images":
                    cv2.imwrite(save_path, raw_frame)
                else:
                    fps = video_capture.get(cv2.CAP_PROP_FPS)
                    w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(raw_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detect_results_name", type=str, default="coco",
                        help="detect results name")
    parser.add_argument("--inputs", type=str, default="./data/examples",
                        help="Input source. Default: ``./data/examples``.")
    parser.add_argument("--names_file_path", type=str, default="./data/coco.names",
                        help="Types of objects detected. Default: ``./data/coco.names``.")
    parser.add_argument("--model_arch_name", type=str, default="yolov3_coco",
                        help="model arch name. Default: ``yolov3_coco``.")
    parser.add_argument("--image_size", type=int or tuple, default=608,
                        help="Image size. Default: 608.")
    parser.add_argument("--gray", type=bool, default=False,
                        help="Whether to use gray image. Default: ``False``.")
    parser.add_argument("--model_weights_path", type=str,
                        default="./results/pretrained_models/YOLOv3-COCO.weights",
                        help="Model file weight path. Default: ``./results/pretrained_models/YOLOv3-COCO.weights``.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device. Default: ``cpu``.")
    parser.add_argument("--half", action="store_true", help="Half precision FP16 inference.")
    parser.add_argument("--fuse", action="store_true", help="Fuse Conv2d + BatchNorm2d layers.")
    parser.add_argument("--show_image", action="store_true", help="Show image.")
    parser.add_argument("--save_txt", action="store_true", help="Save results to *.txt.")
    parser.add_argument("--fourcc", type=str, default="mp4v",
                        help="output video codec (verify ffmpeg support). Default: ``mp4v``.")
    parser.add_argument("--conf_threshold", type=float, default=0.3,
                        help="Object confidence threshold. Default: 0.3.")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IOU threshold for NMS. Default: 0.5.")
    parser.add_argument("--augment", action="store_true",
                        help="augmented inference")
    parser.add_argument("--filter_classes", nargs="+", type=int,
                        help="Filter by class")
    parser.add_argument("--agnostic_nms", action="store_true",
                        help="Class-agnostic NMS")
    args = parser.parse_args()

    main(args)
