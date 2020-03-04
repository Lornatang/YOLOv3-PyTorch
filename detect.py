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
import shutil
import time
from pathlib import Path

import cv2
import torch.backends.cudnn as cudnn
import torch.onnx

from models import Darknet
from models import ONNX_EXPORT
from models import load_darknet_weights
from utils import LoadImages
from utils import LoadStreams
from utils import apply_classifier
from utils import load_classes
from utils import load_classifier
from utils import non_max_suppression
from utils import plot_one_box
from utils import scale_coords
from utils import select_device


def detect(save_img=False):
    # (320, 192) or (416, 256) or (608, 352) for (height, width)
    image_size = (608, 352) if ONNX_EXPORT else args.image_size
    out, source, weights, view_img, save_txt = args.output, args.source, args.weights, args.view_img, args.save_txt
    camera = source == '0' or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(device='cpu' if ONNX_EXPORT else args.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(args.cfg, image_size)

    # Load weights
    if weights.endswith('.pth'):
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        model_classifier = load_classifier(name='resnet101', n=2)  # initialize
        model_classifier.load_state_dict(
            torch.load('weights/resnet101.pth', map_location=device)['model'])  # load weights
        model_classifier.to(device)
        model_classifier.eval()
    else:
        model_classifier = None

    # Migrate the model to the specified device
    model.to(device)
    # set eval model mode
    model.eval()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        image = torch.zeros((1, 3) + image_size)  # (1, 3, 608, 352)
        filename = args.weights.replace(args.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, tuple(image), filename, verbose=False, opset_version=11)

        # Validate exported model
        import onnx
        model = onnx.load(filename)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Set Dataloader
    video_path, video_writer = None, None
    if camera:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=image_size)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=image_size)

    # Get names and colors
    names = load_classes(args.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    start_time = time.time()
    for image_path, image, im0s, video_capture in dataset:
        t = time.time()
        image = torch.from_numpy(image).to(device)
        image = image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        # Inference
        predict = model(image)[0]

        # Apply NMS
        predict = non_max_suppression(predict, args.confidence_threshold, args.iou_threshold, classes=args.classes,
                                      agnostic=args.agnostic_nms)

        # Apply Classifier
        if classify:
            predict = apply_classifier(predict, model_classifier, image, im0s)

        # Process detections
        for i, det in enumerate(predict):  # detections per image
            if camera:  # batch_size >= 1
                p, s, im0 = image_path[i], f'{i}: ', im0s[i]
            else:
                p, s, im0 = image_path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += f'{image.shape[2]}*{image.shape[3]} '  # get image size
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(image.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Print time (inference + NMS)
            print(f"{s}Done. {time.time() - t:.3f}s")

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if video_path != save_path:  # new video
                        video_path = save_path
                        if isinstance(video_writer, cv2.VideoWriter):
                            video_writer.release()  # release previous video writer

                        fps = video_capture.get(cv2.CAP_PROP_FPS)
                        w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*args.fourcc), fps, (w, h))
                    video_writer.write(im0)

    print('Done. (%.3fs)' % (time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.pth', help='weights path')
    parser.add_argument('--source', type=str, default='data/examples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--image-size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--confidence-threshold', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    args = parser.parse_args()
    print(args)

    with torch.no_grad():
        detect()
