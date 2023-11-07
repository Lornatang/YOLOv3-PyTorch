import os 
import json
from tqdm import tqdm
import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument('--json_path', default='./instances_val2017.json',type=str, help="input: coco format(json)")
parser.add_argument('--save_path', default='./labels', type=str, help="specify where to save the output dir of labels")
arg = parser.parse_args()
 
def convert(size, box):
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
    return (x, y, w, h)
 
if __name__ == '__main__':
    json_file =   arg.json_path # COCO Object Instance 类型的标注
    ana_txt_save_path = arg.save_path  # 保存的路径
 
    data = json.load(open(json_file, 'r'))
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)
    
    id_map = {}
    with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
        # 写入classes.txt
        for i, category in enumerate(data['categories']): 
            f.write(f"{category['name']}\n") 
            id_map[category['id']] = i
 
    anns = {}
    for ann in data['annotations']:
        imgid = ann['image_id']
        anns.setdefault(imgid, []).append(ann)
  
    for img in tqdm(data['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
        f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
 
        ann_img = anns.get(img_id, [])
        for ann in ann_img:
            box = convert((img_width, img_height), ann["bbox"])
            f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
        f_txt.close()

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
