import argparse
import os
import os.path as osp
import shutil
import time
from tqdm import tqdm
from pathlib import Path

import yaml

try:
    import orjson as json
except ModuleNotFoundError:
    print("install orjson package makes read json file more quickly! ---->  \033[91mpip install orjson\033[0m")
    import json


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, required=True, help='input dataset dir')
    parser.add_argument('--trainimg-dirname', type=str, default="train2017",
                        help='train image directory name under the dataset directory, default is train2017')
    parser.add_argument('--valimg-dirname', type=str, default="val2017",
                        help='train image directory name under the dataset directory, default is val2017')
    parser.add_argument('--trainjson-filename', type=str, default="instances_train2017.json",
                        help='train label .json filename under the annotations directory, default is instances_train2017.json')
    parser.add_argument('--valjson-filename', type=str, default="instances_val2017.json",
                        help='val label .json filename under the annotations directory, default is instances_val2017.json')
    parser.add_argument('--output-dir', type=str, default=None, help='new dataset dir, it will copy image, not recommend !!')
    parser.add_argument('--classes', type=str, nargs='+', default=None, help='The reserved class name to filter dataset, default nothing to do')
    parser.add_argument('--save_empty', action='store_true', help='whether save the empty data')
    opt = parser.parse_args()
    return opt


# ltwh2xywh 并归一化
def ltwh2xywh_normalize(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def coco2yolo(json_file: str, labels_dir: str, classes: list = None, save_empty=False):
    print(f"开始读取 {osp.abspath(json_file)} ......")
    t1 = time.time()
    with open(json_file, 'r') as f:
        data = json.loads(f.read())
    t2 = time.time() - t1
    print(f"cost: {t2}")

    id_map = {}
    names = {}

    # 解析目标类别，也就是 categories 字段，并将类别写入文件 classes.txt 中，存放在label_dir的同级目录中
    data_dir = Path(labels_dir).parent.as_posix()
    for i, category in enumerate(data['categories']):
        id_map[category['id']] = i
        names[i] = category['name']
    if classes is None:
        classes = list(names.values())
    with open(osp.join(data_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
        f.writelines([x + '\n' for x in classes])
    print(f"generate classes.txt under the {data_dir} Success!!")

    ann_yolo_categorys = {}
    for ann in tqdm(data['annotations'], desc=f'preprocessing the label file annotations: {json_file}'):
        img_id = ann['image_id']
        if img_id not in ann_yolo_categorys:
            ann_yolo_categorys[img_id] = []
        ann_yolo_categorys[img_id].append(ann)

    img_filenames = []
    for img in tqdm(data['images'], total=len(data['images']), desc="convert data...."):

        # 解析 images 字段，分别取出图片文件名、图片的宽和高、图片id
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]

        # label文件名，与对应图片名只有后缀名不一样
        label_filename = osp.splitext(filename)[0] + ".txt"
        label_file = osp.join(labels_dir, label_filename)
        anns = ann_yolo_categorys.get(img_id, [])
        content_lines = []
        for ann in anns:
            box = ltwh2xywh_normalize((img_width, img_height), ann["bbox"])  # 在线的ogc服务作为影像的来源，然后
            name = names[id_map[ann["category_id"]]]
            if name in classes:
                one_line = "%s %s %s %s %s\n" % (classes.index(name), box[0], box[1], box[2], box[3])
                content_lines.append(one_line)
        if len(content_lines) or save_empty:
            # 将图片的标签写入到文件中
            with open(label_file, 'w', encoding='utf-8') as f:
                f.writelines(content_lines)
            img_filenames.append(filename)

    return img_filenames, names


def save_yolo_data_config(dataset_dir, names):
    # Save dataset.yaml
    d = {'path': osp.abspath(dataset_dir),
         'train': "images/train",
         'val': "images/val",
         'test': None,
         'nc': len(names.keys()),  # yolov5 later
         'names': names}  # dictionary

    with open(osp.join(dataset_dir, Path(dataset_dir).with_suffix('.yaml').name), 'w', encoding='utf-8') as f:
        yaml.dump(d, f, sort_keys=False)
    print(f"generate yolo yaml file under the {dataset_dir} Success!!")


def save_txt(save_path, files):
    files = [x + '\n' for x in files]
    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines(files)
    print(f"generate {save_path} Success!!")


def cp_images(img_filenames, o_dir, d_dir, tag='train'):
    dst_img_files = []
    for img_filename in tqdm(img_filenames, total=len(img_filenames), desc=f"copy {tag} image...."):
        o_img_file = osp.join(o_dir, img_filename)
        d_img_file = osp.join(d_dir, img_filename)
        shutil.copy(o_img_file, d_img_file)
        dst_img_files.append(d_img_file)
    return dst_img_files


def mv_images(img_filenames, o_dir, d_dir, tag='train'):
    dst_img_files = []
    for img_filename in tqdm(img_filenames, total=len(img_filenames), desc=f"move {tag} image...."):
        o_img_file = osp.join(o_dir, img_filename)
        d_img_file = osp.join(d_dir, img_filename)
        shutil.move(o_img_file, d_img_file)
        dst_img_files.append(d_img_file)
    return dst_img_files


def new_dataset(output_dir, o_train_img_dir, o_val_img_dir, train_json_file, val_json_file, classes, save_empty):
    # 删除old，创建输出路径的文件夹
    if osp.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    # 转成绝对路径
    output_dir = osp.abspath(output_dir)

    # train
    train_dir = osp.join(output_dir, 'train')
    d_train_img_dir = osp.join(train_dir, 'images')
    d_train_label_dir = osp.join(train_dir, 'labels')
    os.makedirs(d_train_img_dir)
    os.makedirs(d_train_label_dir)
    # convert train label
    print(f"start to convert train annotation file {train_json_file}.....")
    train_img_filenames, names = coco2yolo(train_json_file, d_train_label_dir, classes, save_empty)
    print(f"convert {train_json_file} Success!! ")
    # copy train image
    train_dst_img_files = cp_images(train_img_filenames, o_train_img_dir, d_train_img_dir, tag='train')

    # val
    val_dir = osp.join(output_dir, 'val')
    d_val_img_dir = osp.join(val_dir, 'images')
    d_val_label_dir = osp.join(val_dir, 'labels')
    os.makedirs(d_val_img_dir)
    os.makedirs(d_val_label_dir)
    # convert val label
    print(f"start to convert val annotation file {val_json_file}")
    val_img_filenames, names = coco2yolo(val_json_file, d_val_label_dir, classes, save_empty)
    print(f"convert {val_json_file} Success!! ")
    # copy val image
    val_dst_img_files = cp_images(val_img_filenames, o_val_img_dir, d_val_img_dir, tag='val')

    # Save dataset.yaml
    save_yolo_data_config(output_dir, names)

    # save train.txt, val.txt for image abs path
    # train.txt
    save_txt(osp.join(output_dir, 'train.txt'), train_dst_img_files)
    # val.txt
    save_txt(osp.join(output_dir, 'val.txt'), val_dst_img_files)


def just_convertlabel(dataset_dir, o_train_img_dir, o_val_img_dir, train_json_file, val_json_file, classes, save_empty):
    # 转成绝对路径
    dataset_dir = osp.abspath(dataset_dir)
    # train
    train_dir = osp.join(dataset_dir, 'train')
    d_train_img_dir = osp.join(train_dir, 'images')
    d_train_label_dir = osp.join(train_dir, "labels")
    os.makedirs(d_train_img_dir, exist_ok=True)
    os.makedirs(d_train_label_dir, exist_ok=True)
    # convert train label
    print(f"start to convert train annotation file {train_json_file}.....")
    train_img_filenames, names = coco2yolo(train_json_file, d_train_label_dir, classes, save_empty)
    print(f"convert {train_json_file} Success!! ")
    # move train images
    train_dst_img_files = mv_images(train_img_filenames, o_train_img_dir, d_train_img_dir, tag='train')

    # val
    val_dir = osp.join(dataset_dir, 'val')
    d_val_img_dir = osp.join(val_dir, 'images')
    d_val_label_dir = osp.join(val_dir, "labels")
    os.makedirs(d_val_img_dir, exist_ok=True)
    os.makedirs(d_val_label_dir, exist_ok=True)
    # convert train label
    print(f"start to convert val annotation file {val_json_file}")
    val_img_filenames, names = coco2yolo(val_json_file, d_val_label_dir, classes, save_empty)
    print(f"convert {val_json_file} Success!! ")
    # move val images
    val_dst_img_files = mv_images(val_img_filenames, o_val_img_dir, d_val_img_dir, tag='val')

    # Save dataset.yaml
    save_yolo_data_config(dataset_dir, names)


def run(dataset_dir, train_img_dirname, val_img_dirname, train_json_filename, val_json_filename, classes, output_dir=None, save_empty=False):
    train_json_file = osp.join(dataset_dir, "annotations", train_json_filename)
    val_json_file = osp.join(dataset_dir, "annotations", val_json_filename)
    train_img_dir = osp.join(dataset_dir, train_img_dirname)
    val_img_dir = osp.join(dataset_dir, val_img_dirname)
    assert osp.exists(train_json_file), f"{train_json_file} not exists, please make sure your input param trainjson_filename is correct"
    assert osp.exists(val_json_file), f"{val_json_file} not exists, please make sure your input param valjson_filename is correct"
    assert osp.exists(train_img_dir), f"{train_img_dir} not exists, please make sure your trainimg_dirname is correct"
    assert osp.exists(val_img_dir), f"{val_img_dir} not exists, please make sure your valimg_dirname is correct"
    if output_dir is None:
        just_convertlabel(dataset_dir, train_img_dir, val_img_dir, train_json_file, val_json_file, classes, save_empty)
    else:
        new_dataset(output_dir, train_img_dir, val_img_dir, train_json_file, val_json_file, classes, save_empty)


if __name__ == '__main__':
    opt = parse_opt()
    run(dataset_dir=opt.dataset_dir,
        train_img_dirname=opt.trainimg_dirname,
        val_img_dirname=opt.valimg_dirname,
        train_json_filename=opt.trainjson_filename,
        val_json_filename=opt.valjson_filename,
        classes=opt.classes,
        output_dir=opt.output_dir,
        save_empty=opt.save_empty)

    # write train.txt and val.txt
    train_dir = "./images/train"
    val_dir = "./images/valid"
    train_txt = "./train.txt"
    val_txt = "./val.txt"
    for file in os.listdir(train_dir):
        with open(train_txt, "a") as f:
            f.write(f"./images/train/{file}\n")
    for file in os.listdir(val_dir):
        with open(val_txt, "a") as f:
            f.write(f"./images/valid/{file}\n")
