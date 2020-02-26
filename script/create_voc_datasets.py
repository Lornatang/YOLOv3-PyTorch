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

# Source: https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
import os
import xml.etree.ElementTree

sets = [("2007", "train"), ("2007", "val"), ("2007", "test")]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(year, image_id):
    in_file = open(f"VOCdevkit/VOC{year}/Annotations/{image_id}.xml")
    out_file = open(f"VOCdevkit/VOC{year}/labels/{image_id}.txt", "w")
    tree = xml.etree.ElementTree.parse(in_file)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        cls = obj.find("name").text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find("bndbox")
        b = (float(xmlbox.find("xmin").text), float(xmlbox.find("xmax").text), float(xmlbox.find("ymin").text),
             float(xmlbox.find("ymax").text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + "\n")


current_path = os.getcwd()

for years, image_set in sets:
    if not os.path.exists(f"VOCdevkit/VOC{years}/labels/"):
        os.makedirs(f"VOCdevkit/VOC{years}/labels/")
    image_indexs = open(f"VOCdevkit/VOC{years}/ImageSets/Main/{image_set}.txt").read().strip().split()
    list_file = open(f"{years}_{image_set}.txt", "w")
    for image_index in image_indexs:
        list_file.write(f"{current_path}/VOCdevkit/VOC{years}/JPEGImages/{image_index}.jpg\n")
        convert_annotation(years, image_index)
    list_file.close()
