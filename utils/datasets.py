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
import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread
import warnings

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.utils import xyxy2xywh, xywh2xyxy

help_url = "https://github.com/Lornatang/YOLOv3-PyTorch#train-on-custom-dataset"
image_formats = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".dng"]
video_formats = [".mov", ".avi", ".mp4"]

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def exif_size(image):
    """ Returns exif-corrected PIL size.

    Args:
        image (PngImageFile): Image matrix data.

    Returns:
        Size after image processing (width, height).
    """
    size = image.size
    try:
        rotation = dict(image._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            size = (size[1], size[0])
        elif rotation == 8:  # rotation 90
            size = (size[1], size[0])
    except RuntimeWarning:
        pass

    return size


class LoadImages:
    """ Use only in the inference phase"""

    def __init__(self, path, image_size=416):
        """ Load the pictures in the directory and convert them to the corresponding format.

        Args:
            path (str): Source address of the file.
            image_size (int): Size of loaded pictures. default:416
        """
        path = str(Path(path))
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.*")))
        elif os.path.isfile(path):
            files = [path]

        images = [x for x in files if os.path.splitext(x)[-1].lower() in image_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in video_formats]
        image_num, video_num = len(images), len(videos)

        self.image_size = image_size
        self.files = images + videos
        self.files_num = image_num + video_num
        self.video_flag = [False] * image_num + [True] * video_num
        self.mode = "images"
        if any(videos):
            self.new_video(videos[0])
        else:
            self.capture = None
        assert self.files_num > 0, "No images or videos found in `" + path + "`"

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.files_num:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            ret_val, raw_image = self.capture.read()
            if not ret_val:
                self.count += 1
                self.capture.release()
                if self.count == self.files_num:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, raw_image = self.capture.read()

            self.frame += 1
            print(f"video {self.count + 1}/{self.files_num} ({self.frame}/{self.frames_num}) {path}: ", end='')

        else:
            # Read image
            self.count += 1
            raw_image = cv2.imread(path)  # BGR
            assert raw_image is not None, "Image Not Found `" + path + "`"
            print(f"image {self.count}/{self.files_num} {path}: ", end='')

        # Padded resize
        image = letterbox(raw_image, new_shape=self.image_size)[0]

        # BGR convert to RGB (3x416x416)
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)

        return path, image, raw_image, self.capture

    def new_video(self, path):
        self.frame = 0
        self.capture = cv2.VideoCapture(path)
        self.frames_num = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.files_num


class LoadCamera:
    """ Use only in the inference phase"""

    def __init__(self, pipe=0, image_size=416):
        self.image_size = image_size

        if pipe == "0":
            pipe = 0  # local camera

        self.pipe = pipe
        self.capture = cv2.VideoCapture(pipe)  # video capture object
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord("q"):
            self.capture.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            retval, raw_image = self.capture.read()
            raw_image = cv2.flip(raw_image, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.capture.grab()
                if n % 30 == 0:  # skip frames
                    retval, raw_image = self.capture.retrieve()
                    if retval:
                        break

        assert retval, f"Camera Error {self.pipe}"
        image_path = "webcam.jpg"
        print(f"webcam {self.count}: ", end='')

        # Padded resize
        image = letterbox(raw_image, new_shape=self.image_size)[0]

        # BGR convert to RGB (3x416x416)
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)

        return image_path, image, raw_image, None

    def __len__(self):
        return 0


class LoadStreams:
    """ For reading camera and network data"""

    def __init__(self, sources="streams.txt", image_size=608):
        self.mode = "images"
        self.image_size = image_size

        if os.path.isfile(sources):
            with open(sources, "r") as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.images = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f"{i + 1}/{n}: {s}... ", end='')

            cap = cv2.VideoCapture(0 if s == "0" else s)
            assert cap.isOpened(), "Failed to open %s" % s

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.images[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f"Success ({width}*{height} at {fps:.2f}FPS).")
            thread.start()
        print("")  # newline

        # check for common shapes
        s = np.stack([letterbox(x, new_shape=self.image_size)[0].shape for x in self.images], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            warnings.warn("WARNING: Different stream shapes detected. "
                          "For optimal performance supply similarly-shaped streams.")

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        num = 0
        while cap.isOpened():
            num += 1
            cap.grab()
            if num == 4:  # read every 4th frame
                _, self.images[index] = cap.retrieve()
                num = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        raw_image = self.images.copy()
        if cv2.waitKey(1) == ord("q"):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        image = [letterbox(x, new_shape=self.image_size, auto=self.rect, interp=cv2.INTER_LINEAR)[0] for x in raw_image]

        # Stack
        image = np.stack(image, 0)

        # Convert
        image = image[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        image = np.ascontiguousarray(image)

        return self.sources, image, raw_image, None

    def __len__(self):
        return 0


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, image_size=416, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_labels=True, cache_images=False, single_cls=False):
        path = str(Path(path))  # os-agnostic
        assert os.path.isfile(path), f"File not found {path}. See {help_url}"
        with open(path, "r") as f:
            self.image_files = [x.replace("/", os.sep) for x in f.read().splitlines()  # os-agnostic
                                if os.path.splitext(x)[-1].lower() in image_formats]

        image_files_num = len(self.image_files)
        assert image_files_num > 0, f"No images found in {path}. See {help_url}"
        batch_index = np.floor(np.arange(image_files_num) / batch_size).astype(np.int)
        batch_num = batch_index[-1] + 1

        self.image_files_num = image_files_num
        self.batch = batch_index  # batch index of image
        self.img_size = image_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)

        # Define labels
        self.label_files = [x.replace("images", "labels").replace(os.path.splitext(x)[-1], ".txt")
                            for x in self.image_files]

        if self.rect:
            sp = path.replace(".txt", ".shapes")  # shapefile path
            try:
                with open(sp, "r") as f:  # read existing shapefile
                    s = [x.split() for x in f.read().splitlines()]
                    assert len(s) == self.image_files_num, "Shapefile out of sync"
            except:
                s = [exif_size(Image.open(f)) for f in tqdm(self.image_files, desc="Reading image shapes")]
                np.savetxt(sp, s, fmt="%g")  # overwrites existing (if any)

            # Sort by aspect ratio
            s = np.array(s, dtype=np.float64)
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            i = ar.argsort()
            self.image_files = [self.image_files[i] for i in i]
            self.label_files = [self.label_files[i] for i in i]
            self.shapes = s[i]  # wh
            ar = ar[i]

            # Set training image shapes
            shapes = [[1, 1]] * batch_num
            for i in range(batch_num):
                ari = ar[batch_index == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * image_size / 32.).astype(np.int) * 32

        # Preload labels (required for weighted CE training)
        self.images = [None] * self.image_files_num
        self.labels = [None] * self.image_files_num
        if cache_labels or image_weights:  # cache labels for faster training
            self.labels = [np.zeros((0, 5))] * self.image_files_num
            extract_bounding_boxes = False
            create_datasubset = False
            pbar = tqdm(self.label_files, desc="Caching labels")
            nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
            for i, file in enumerate(pbar):
                try:
                    with open(file, "r") as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except:
                    nm += 1  # print("missing labels for image %s" % self.img_files[i])  # file missing
                    continue

                if l.shape[0]:
                    assert l.shape[1] == 5, "> 5 label columns: %s" % file
                    assert (l >= 0).all(), "negative labels: %s" % file
                    assert (l[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels: %s" % file
                    if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                        nd += 1  # print("WARNING: duplicate rows in %s" % self.label_files[i])  # duplicate rows
                    if single_cls:
                        l[:, 0] = 0  # force dataset into single-class mode
                    self.labels[i] = l
                    nf += 1  # file found

                    # Create subdataset (a smaller dataset)
                    if create_datasubset and ns < 1E4:
                        if ns == 0:
                            create_folder(path="./datasubset")
                            os.makedirs("./datasubset/images")
                        exclude_classes = 43
                        if exclude_classes not in l[:, 0]:
                            ns += 1
                            # shutil.copy(src=self.img_files[i], dst="./datasubset/images/")  # copy image
                            with open("./datasubset/images.txt", "a") as f:
                                f.write(self.image_files[i] + "\n")

                    # Extract object detection boxes for a second stage classifier
                    if extract_bounding_boxes:
                        p = Path(self.image_files[i])
                        img = cv2.imread(str(p))
                        h, w = img.shape[:2]
                        for j, x in enumerate(l):
                            f = "%s%sclassifier%s%g_%g_%s" % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                            if not os.path.exists(Path(f).parent):
                                os.makedirs(Path(f).parent)  # make new output folder

                            b = x[1:] * [w, h, w, h]  # box
                            b[2:] = b[2:].max()  # rectangle to square
                            b[2:] = b[2:] * 1.3 + 30  # pad
                            b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                            b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                            b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                            assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), "Failure extracting classifier boxes"
                else:
                    ne += 1  # print("empty labels for image %s" % self.img_files[i])  # file empty
                    # os.system("rm "%s" "%s"" % (self.img_files[i], self.label_files[i]))  # remove

                pbar.desc = "Caching labels (%g found, %g missing, %g empty, %g duplicate, for %g images)" % (
                    nf, nm, ne, nd, self.image_files_num)
            assert nf > 0, "No labels found. See %s" % help_url

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        if cache_images:  # if training
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.image_files)), desc="Caching images")
            self.img_hw0, self.img_hw = [None] * self.image_files_num, [None] * self.image_files_num
            for i in pbar:  # max 10k images
                self.images[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
                gb += self.images[i].nbytes
                pbar.desc = "Caching images (%.1fGB)" % (gb / 1E9)

        # Detect corrupted images https://medium.com/joelthchao/programmatically-detect-corrupted-image-8c1b2006c3d3
        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io  # conda install -c conda-forge scikit-image
            for file in tqdm(self.image_files, desc="Detecting corrupted images"):
                try:
                    _ = io.imread(file)
                except IOError:
                    print("Corrupted image detected: %s" % file)

    def __len__(self):
        return len(self.image_files)

    # def __iter__(self):
    #     self.count = -1
    #     print("ran dataset iter")
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        img_path = self.image_files[index]
        label_path = self.label_files[index]

        hyp = self.hyp
        if self.mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            if os.path.isfile(label_path):
                x = self.labels[index]
                if x is None:  # labels not preloaded
                    with open(label_path, "r") as f:
                        x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)

                if x.size > 0:
                    # Normalized xywh to pixel xyxy format
                    labels = x.copy()
                    labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                    labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                    labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                    labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            # Augment imagespace
            if not self.mosaic:
                img, labels = random_affine(img, labels,
                                            degrees=hyp["degrees"],
                                            translate=hyp["translate"],
                                            scale=hyp["scale"],
                                            shear=hyp["shear"])

            # Augment colorspace
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, img_path, shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    image = self.images[index]
    if image is None:  # not cached
        img_path = self.image_files[index]
        image = cv2.imread(img_path)  # BGR
        assert image is not None, "Image Not Found " + img_path
        h0, w0 = image.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r < 1 or (self.augment and (r != 1)):  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_LINEAR if self.augment else cv2.INTER_AREA  # LINEAR for training, AREA for testing
            image = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return image, (h0, w0), image.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.images[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    x = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * x).clip(None, 255).astype(np.uint8)
    np.clip(img_hsv[:, :, 0], None, 179, out=img_hsv[:, :, 0])  # inplace hue clip (0 - 179 deg)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def load_mosaic(self, index):
    # loads images in a mosaic

    labels4 = []
    s = self.img_size
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    img4 = np.zeros((s * 2, s * 2, 3), dtype=np.uint8) + 128  # base image with 4 tiles
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Load labels
        label_path = self.label_files[index]
        if os.path.isfile(label_path):
            x = self.labels[index]
            if x is None:  # labels not preloaded
                with open(label_path, "r") as f:
                    x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)

            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
            else:
                labels = np.zeros((0, 5), dtype=np.float32)
            labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

    # Augment
    img4, labels4 = random_affine(img4, labels4,
                                  degrees=self.hyp["degrees"] * 1,
                                  translate=self.hyp["translate"] * 1,
                                  scale=self.hyp["scale"] * 1,
                                  shear=self.hyp["shear"] * 1,
                                  border=-s // 2)  # border to remove

    return img4, labels4


def letterbox(img, new_shape=(416, 416), color=(128, 128, 128),
              auto=True, scaleFill=False, scaleup=True, interp=cv2.INTER_AREA):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=interp)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))

    if targets is None:  # targets = [cls, xyxy]
        targets = []
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    changed = (border != 0) or (M != np.eye(3)).any()
    if changed:
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_AREA, borderValue=(128, 128, 128))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def cutout(image, labels):
    # Reference: https://arxiv.org/abs/1708.04552
    # Reference: https://github.com/hysts/pytorch_cutout/blob/master/dataloader.py
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def reduce_img_size(path="../data/sm4/images", img_size=1024):  # from utils.datasets import *; reduce_img_size()
    # creates a new ./images_reduced folder with reduced size images of maximum size img_size
    path_new = path + "_reduced"  # reduced images path
    create_folder(path_new)
    for f in tqdm(glob.glob("%s/*.*" % path)):
        try:
            img = cv2.imread(f)
            h, w = img.shape[:2]
            r = img_size / max(h, w)  # size ratio
            if r < 1.0:
                img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)  # _LINEAR fastest
            fnew = f.replace(path, path_new)  # .replace(Path(f).suffix, ".jpg")
            cv2.imwrite(fnew, img)
        except:
            print("WARNING: image failure %s" % f)


def convert_images2bmp():  # from utils.datasets import *; convert_images2bmp()
    # Save images
    formats = [x.lower() for x in image_formats] + [x.upper() for x in image_formats]
    for path in ["../data/sm4/images", "../data/sm4/background"]:
        create_folder(path + "bmp")
        for ext in formats:
            for f in tqdm(glob.glob("%s/*%s" % (path, ext)), desc="Converting %s" % ext):
                cv2.imwrite(f.replace(ext.lower(), ".bmp").replace(path, path + "bmp"), cv2.imread(f))

    # Save labels
    for file in ["../data/sm4/out_train.txt", "../data/sm4/out_test.txt"]:
        with open(file, "r") as f:
            lines = f.read()
            lines = lines.replace("/images", "/imagesbmp")
            lines = lines.replace("/background", "/backgroundbmp")
        for ext in formats:
            lines = lines.replace(ext, ".bmp")
        with open(file.replace(".txt", "bmp.txt"), "w") as f:
            f.write(lines)


def recursive_dataset2bmp(dataset="../data/sm4_bmp"):  # from utils.datasets import *; recursive_dataset2bmp()
    # Converts dataset to bmp (for faster training)
    formats = [x.lower() for x in image_formats] + [x.upper() for x in image_formats]
    for a, b, files in os.walk(dataset):
        for file in tqdm(files, desc=a):
            p = a + "/" + file
            s = Path(file).suffix
            if s == ".txt":  # replace text
                with open(p, "r") as f:
                    lines = f.read()
                for f in formats:
                    lines = lines.replace(f, ".bmp")
                with open(p, "w") as f:
                    f.write(lines)
            elif s in formats:  # replace image
                cv2.imwrite(p.replace(s, ".bmp"), cv2.imread(p))
                if s != ".bmp":
                    os.system("rm `%s`" % p)


def imagelist2folder(path="data/coco_64img.txt"):  # from utils.datasets import *; imagelist2folder()
    # Copies all the images in a text file (list of images) into a folder
    create_folder(path[:-4])
    with open(path, "r") as f:
        for line in f.read().splitlines():
            os.system("cp `%s` %s" % (line, path[:-4]))
            print(line)


def create_folder(path="./new_folder"):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
