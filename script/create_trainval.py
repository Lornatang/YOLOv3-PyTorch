import os
import random

xml_path = "Annotations"
txt_path = "ImageSets/Main"

xml_list = os.listdir(xml_path)
xml_total = len(xml_list)
train_total = int(xml_total * 0.8)
train_sample = random.sample(range(xml_total), train_total)

train = open(txt_path + "/train.txt", "w")
val = open(txt_path + "/val.txt", "w")

for i in range(xml_total):
    name = xml_list[i][:-4] + "\n"
    if i in train_sample:
        train.write(name)
    else:
        val.write(name)

train.close()
val.close()
