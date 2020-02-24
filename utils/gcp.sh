#!/usr/bin/env bash

# New VM
rm -rf sample_data yolov3
git clone https://github.com/ultralytics/yolov3
# sudo apt-get install zip
#git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . --user && cd .. && rm -rf apex
sudo conda install -yc conda-forge scikit-image pycocotools
python3 -c "from yolov3.utils.google_utils import gdrive_download; gdrive_download('193Zp_ye-3qXMonR1nZj3YyxMtQkMy50k','coco2014.zip')"
python3 -c "from yolov3.utils.google_utils import gdrive_download; gdrive_download('1C3HewOG9akA3y456SZLBJZfNDPkBwAto','knife.zip')"
python3 -c "from yolov3.utils.google_utils import gdrive_download; gdrive_download('13g3LqdpkNE8sPosVJT6KFXlfoMypzRP4','sm4.zip')"
sudo shutdown

# Re-clone
rm -rf yolov3  # Warning: remove existing
git clone https://github.com/ultralytics/yolov3 # master
bash yolov3/data/get_coco2017.sh
# git clone -b test --depth 1 https://github.com/ultralytics/yolov3 test  # branch
cd yolov3
python3 test.py --weights ultralytics68.pt --task benchmark

# Mount local SSD
lsblk
sudo mkfs.ext4 -F /dev/nvme0n1
sudo mkdir -p /mnt/disks/nvme0n1
sudo mount /dev/nvme0n1 /mnt/disks/nvme0n1
sudo chmod a+w /mnt/disks/nvme0n1
cp -r coco /mnt/disks/nvme0n1

# Train
python3 train.py

# Resume
python3 train.py --resume

# Detect
python3 detect.py

# Test
python3 test.py --save-json

# Kill All
t=ultralytics/yolov3:v240
docker kill $(docker ps -a -q --filter ancestor=$t)
t=ultralytics/yolov3:v208
docker kill $(docker ps -a -q --filter ancestor=$t)

# Evolve wer
sudo -s
t=ultralytics/yolov3:v206
docker kill $(docker ps -a -q --filter ancestor=$t)
for i in 0 1 2 3 0 1 2 3
do
  docker pull $t && docker run -d --gpus all --ipc=host -v "$(pwd)"/data:/usr/src/data $t bash utils/evolve.sh $i
  sleep 180
done

# Evolve athena
sudo -s
t=ultralytics/yolov3:v208
docker kill $(docker ps -a -q --filter ancestor=$t)
for i in 0 1
do
  docker pull $t && docker run --gpus all -d --ipc=host -v "$(pwd)"/out:/usr/src/out $t bash utils/evolve.sh $i
  sleep 120
done

# Evolve coco
sudo -s
t=ultralytics/yolov3:evolve
# docker kill $(docker ps -a -q --filter ancestor=$t)
for i in 0 1 6 7
do
  docker pull $t && docker run --gpus all -d --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t bash utils/evolve.sh $i
  sleep 30
done


t=ultralytics/yolov3:evolve && docker pull $t && docker run --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t bash utils/evolve.sh 2


# Git pull
git pull https://github.com/ultralytics/yolov3  # master
git pull https://github.com/ultralytics/yolov3 test  # branch

# Test Darknet training
python3 test.py --weights ../darknet/backup/yolov3.backup

# Copy last.pt TO bucket
gsutil cp yolov3/weights/last1gpu.pt gs://ultralytics

# Copy last.pt FROM bucket
gsutil cp gs://ultralytics/last.pt yolov3/weights/last.pt
wget https://storage.googleapis.com/ultralytics/yolov3/last_v1_0.pt -O weights/last_v1_0.pt
wget https://storage.googleapis.com/ultralytics/yolov3/best_v1_0.pt -O weights/best_v1_0.pt

# Reproduce tutorials
rm results*.txt  # WARNING: removes existing results
python3 train.py --nosave --data data/coco_1img.data && mv results.txt results0r_1img.txt
python3 train.py --nosave --data data/coco_10img.data && mv results.txt results0r_10img.txt
python3 train.py --nosave --data data/coco_100img.data && mv results.txt results0r_100img.txt
# python3 train.py --nosave --data data/coco_100img.data --transfer && mv results.txt results3_100imgTL.txt
python3 -c "from utils import utils; utils.plot_results()"
# gsutil cp results*.txt gs://ultralytics
gsutil cp results.png gs://ultralytics
sudo shutdown

# Reproduce mAP
python3 test.py --save-json --img 608
python3 test.py --save-json --img 416
python3 test.py --save-json --img 320
sudo shutdown

# Benchmark script
git clone https://github.com/ultralytics/yolov3  # clone our repo
git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . --user && cd .. && rm -rf apex  # install nvidia apex
python3 -c "from yolov3.utils.google_utils import gdrive_download; gdrive_download('1HaXkef9z6y5l4vUnCYgdmEAj61c6bfWO','coco.zip')"  # download coco dataset (20GB)
cd yolov3 && clear && python3 train.py --epochs 1  # run benchmark (~30 min)

# Unit tests
python3 detect.py  # detect 2 persons, 1 tie
python3 test.py --data data/coco_32img.data  # test mAP = 0.8
python3 train.py --data data/coco_32img.data --epochs 5 --nosave  # train 5 epochs
python3 train.py --data data/coco_1cls.data --epochs 5 --nosave  # train 5 epochs
python3 train.py --data data/coco_1img.data --epochs 5 --nosave  # train 5 epochs

# AlexyAB Darknet
gsutil cp -r gs://sm6/supermarket2 .  # dataset from bucket
rm -rf darknet && git clone https://github.com/AlexeyAB/darknet && cd darknet && wget -c https://pjreddie.com/media/files/darknet53.conv.74  # sudo apt install libopencv-dev && make
./darknet detector calc_anchors data/coco_img64.data -num_of_clusters 9 -width 320 -height 320  # kmeans anchor calculation
./darknet detector train ../supermarket2/supermarket2.data ../yolo_v3_spp_pan_scale.cfg darknet53.conv.74 -map -dont_show # train spp
./darknet detector train ../yolov3/data/coco.data ../yolov3-spp.cfg darknet53.conv.74 -map -dont_show # train spp coco

#Docker
sudo docker kill "$(sudo docker ps -q)"
sudo docker pull ultralytics/yolov3:v0
sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco ultralytics/yolov3:v0


t=ultralytics/yolov3:v70 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 32 --accum 2 --pre --bucket yolov4 --name 70 --device 0 --multi
t=ultralytics/yolov3:v73 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 73 --device 5 --cfg cfg/yolov3s.cfg
t=ultralytics/yolov3:v74 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 74 --device 0 --cfg cfg/yolov3s.cfg
t=ultralytics/yolov3:v75 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 75 --device 7 --cfg cfg/yolov3s.cfg
t=ultralytics/yolov3:v76 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 76 --device 0 --cfg cfg/yolov3-spp.cfg

t=ultralytics/yolov3:v79 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 79 --device 5
t=ultralytics/yolov3:v80 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 80 --device 0
t=ultralytics/yolov3:v81 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 81 --device 7
t=ultralytics/yolov3:v82 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 82 --device 0 --cfg cfg/yolov3s.cfg

t=ultralytics/yolov3:v83 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --bucket yolov4 --name 83 --device 6 --multi --nosave
t=ultralytics/yolov3:v84 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --bucket yolov4 --name 84 --device 0 --multi
t=ultralytics/yolov3:v85 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --bucket yolov4 --name 85 --device 0 --multi
t=ultralytics/yolov3:v86 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --bucket yolov4 --name 86 --device 1 --multi
t=ultralytics/yolov3:v87 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --bucket yolov4 --name 87 --device 2 --multi
t=ultralytics/yolov3:v88 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --bucket yolov4 --name 88 --device 3 --multi
t=ultralytics/yolov3:v89 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 89 --device 1
t=ultralytics/yolov3:v90 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 90 --device 0 --cfg cfg/yolov3-spp-matrix.cfg
t=ultralytics/yolov3:v91 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 91 --device 0 --cfg cfg/yolov3-spp-matrix.cfg

t=ultralytics/yolov3:v92 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 92 --device 0
t=ultralytics/yolov3:v93 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 27 --batch 16 --accum 4 --pre --bucket yolov4 --name 93 --device 0 --cfg cfg/yolov3-spp-matrix.cfg


#SM4
t=ultralytics/yolov3:v96 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'ultralytics68.pt' --epochs 1000 --img 320 --batch 32 --accum 2 --pre --bucket yolov4 --name 96 --device 0 --multi --cfg cfg/yolov3-spp-3cls.cfg --data ../data/sm4/out.data --nosave
t=ultralytics/yolov3:v97 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'ultralytics68.pt' --epochs 1000 --img 320 --batch 32 --accum 2 --pre --bucket yolov4 --name 97 --device 4 --multi --cfg cfg/yolov3-spp-3cls.cfg --data ../data/sm4/out.data --nosave
t=ultralytics/yolov3:v98 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'ultralytics68.pt' --epochs 1000 --img 320 --batch 16 --accum 4 --pre --bucket yolov4 --name 98 --device 5 --multi --cfg cfg/yolov3-spp-3cls.cfg --data ../data/sm4/out.data --nosave
t=ultralytics/yolov3:v113 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 16 --accum 4 --pre --bucket yolov4 --name 101 --device 7 --multi --nosave

t=ultralytics/yolov3:v102 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'yolov3-tiny.pt' --epochs 1000 --img 320 --batch 64 --accum 1 --pre --bucket yolov4 --name 102 --device 0 --cfg cfg/yolov3-tiny-3cls.cfg --data ../data/sm4/out.data --nosave --cache
t=ultralytics/yolov3:v103 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'yolov3-tiny.pt' --epochs 500 --img 320 --batch 64 --accum 1 --pre --bucket yolov4 --name 103 --device 0 --cfg cfg/yolov3-tiny-3cls.cfg --data ../data/sm4/out.data --nosave --cache
t=ultralytics/yolov3:v104 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'yolov3-tiny.pt' --epochs 500 --img 320 --batch 64 --accum 1 --pre --bucket yolov4 --name 104 --device 0 --cfg cfg/yolov3-tiny-3cls.cfg --data ../data/sm4/out.data --nosave --cache
t=ultralytics/yolov3:v105 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'yolov3-tiny.pt' --epochs 500 --img 320 --batch 64 --accum 1 --pre --bucket yolov4 --name 105 --device 0 --cfg cfg/yolov3-tiny-3cls.cfg --data ../data/sm4/out.data --nosave --cache
t=ultralytics/yolov3:v106 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/data:/usr/src/data $t python3 train.py --weights 'yolov3-tiny.pt' --epochs 500 --img 320 --batch 64 --accum 1 --pre --bucket yolov4 --name 106 --device 0 --cfg cfg/yolov3-tiny-3cls-sm4.cfg --data ../data/sm4/out.data --nosave --cache
t=ultralytics/yolov3:v107 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 107 --device 5 --nosave --cfg cfg/yolov3-spp3.cfg
t=ultralytics/yolov3:v108 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 108 --device 7 --nosave

t=ultralytics/yolov3:v109 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --bucket yolov4 --name 109 --device 4 --multi --nosave
t=ultralytics/yolov3:v110 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --bucket yolov4 --name 110 --device 3 --multi --nosave

t=ultralytics/yolov3:v83 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 111 --device 0
t=ultralytics/yolov3:v112 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 112 --device 1 --nosave
t=ultralytics/yolov3:v113 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 113 --device 2 --nosave
t=ultralytics/yolov3:v114 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 114 --device 2 --nosave
t=ultralytics/yolov3:v113 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 115 --device 5 --nosave  --cfg cfg/yolov3-spp3.cfg
t=ultralytics/yolov3:v116 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 116 --device 1 --nosave

t=ultralytics/yolov3:v83 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 16 --accum 4 --epochs 27 --pre --bucket yolov4 --name 117 --device 0 --nosave --multi
t=ultralytics/yolov3:v118 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 16 --accum 4 --epochs 27 --pre --bucket yolov4 --name 118 --device 5 --nosave --multi
t=ultralytics/yolov3:v119 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 119 --device 1 --nosave
t=ultralytics/yolov3:v120 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 120 --device 2 --nosave
t=ultralytics/yolov3:v121 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 121 --device 0 --nosave --cfg cfg/csresnext50-panet-spp.cfg
t=ultralytics/yolov3:v122 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 273 --pre --bucket yolov4 --name 122 --device 2 --nosave
t=ultralytics/yolov3:v123 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 273 --pre --bucket yolov4 --name 123 --device 5 --nosave

t=ultralytics/yolov3:v124 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 124 --device 0 --nosave --cfg yolov3-tiny
t=ultralytics/yolov3:v124 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 125 --device 1 --nosave --cfg yolov3-tiny2
t=ultralytics/yolov3:v124 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 126 --device 1 --nosave --cfg yolov3-tiny3
t=ultralytics/yolov3:v127 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 127 --device 0 --nosave --cfg yolov3-tiny4
t=ultralytics/yolov3:v124 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 64 --accum 1 --epochs 273 --pre --bucket yolov4 --name 128 --device 1 --nosave --cfg yolov3-tiny2 --multi
t=ultralytics/yolov3:v129 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 64 --accum 1 --epochs 273 --pre --bucket yolov4 --name 129 --device 0 --nosave --cfg yolov3-tiny2

t=ultralytics/yolov3:v130 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 130 --device 0 --nosave
t=ultralytics/yolov3:v133 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 22 --accum 3 --epochs 250 --pre --bucket yolov4 --name 131 --device 0 --nosave --multi
t=ultralytics/yolov3:v130 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 132 --device 0 --nosave --data coco2014.data
t=ultralytics/yolov3:v133 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 22 --accum 3 --epochs 27 --pre --bucket yolov4 --name 133 --device 0 --nosave --multi
t=ultralytics/yolov3:v134 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 134 --device 0 --nosave --data coco2014.data

t=ultralytics/yolov3:v135 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 24 --accum 3 --epochs 270 --pre --bucket yolov4 --name 135 --device 0 --nosave --multi --data coco2014.data
t=ultralytics/yolov3:v136 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 24 --accum 3 --epochs 270 --pre --bucket yolov4 --name 136 --device 0 --nosave --multi --data coco2014.data

t=ultralytics/yolov3:v137 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 137 --device 7 --nosave --data coco2014.data
t=ultralytics/yolov3:v137 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --bucket yolov4 --name 138 --device 6 --nosave --data coco2014.data

t=ultralytics/yolov3:v140 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 140 --device 1 --nosave --data coco2014.data --arc uBCE
t=ultralytics/yolov3:v141 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 141 --device 0 --nosave --data coco2014.data --arc uBCE
t=ultralytics/yolov3:v142 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --batch 32 --accum 2 --epochs 27 --pre --bucket yolov4 --name 142 --device 1 --nosave --data coco2014.data --arc uBCE

t=ultralytics/yolov3:v146 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 146 --device 0 --nosave --data coco2014.data
t=ultralytics/yolov3:v147 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 147 --device 1 --nosave --data coco2014.data
t=ultralytics/yolov3:v148 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 148 --device 2 --nosave --data coco2014.data
t=ultralytics/yolov3:v149 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 149 --device 3 --nosave --data coco2014.data
t=ultralytics/yolov3:v150 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 150 --device 4 --nosave --data coco2014.data
t=ultralytics/yolov3:v151 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 151 --device 5 --nosave --data coco2014.data
t=ultralytics/yolov3:v152 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 152 --device 6 --nosave --data coco2014.data
t=ultralytics/yolov3:v153 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 153 --device 7 --nosave --data coco2014.data

t=ultralytics/yolov3:v154 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 154 --device 0 --nosave --data coco2014.data
t=ultralytics/yolov3:v155 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 155 --device 0 --nosave --data coco2014.data --arc defaultpw

t=ultralytics/yolov3:v156 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 156 --device 5 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v157 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 157 --device 6 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v158 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 158 --device 7 --nosave --data coco2014.data --arc defaultpw

t=ultralytics/yolov3:v159 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 159 --device 0 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v160 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 160 --device 1 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v161 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 161 --device 2 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v162 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 162 --device 3 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v163 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 163 --device 4 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v164 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 164 --device 5 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v165 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 165 --device 6 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v166 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 166 --device 6 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v167 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 167 --device 7 --nosave --data coco2014.data --arc defaultpw

t=ultralytics/yolov3:v168 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 168 --device 5 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v169 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 169 --device 6 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v170 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 170 --device 7 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v171 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 171 --device 4 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v172 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 172 --device 3 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v173 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 173 --device 2 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v174 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 174 --device 1 --nosave --data coco2014.data --arc defaultpw
t=ultralytics/yolov3:v175 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 320 --batch 64 --accum 1 --epochs 27 --pre --bucket yolov4 --name 175 --device 0 --nosave --data coco2014.data --arc defaultpw

t=ultralytics/yolov3:v177 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 416 --batch 22 --accum 3 --epochs 273 --pre --bucket yolov4 --name 177 --device 0 --nosave --data coco2014.data --multi
t=ultralytics/yolov3:v178 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 416 --batch 22 --accum 3 --epochs 273 --pre --bucket yolov4 --name 178 --device 0 --nosave --data coco2014.data --multi
t=ultralytics/yolov3:v179 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --img 416 --batch 22 --accum 3 --epochs 273 --pre --bucket yolov4 --name 179 --device 0 --nosave --data coco2014.data --multi --cfg yolov3s-18a.cfg

t=ultralytics/yolov3:v143 && sudo docker build -t $t . && sudo docker push $t

t=ultralytics/yolov3:v179 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name 179
t=ultralytics/yolov3:v180 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name 180
t=ultralytics/yolov3:v183 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name 181 --cfg yolov3s9a-640.cfg
t=ultralytics/yolov3:v183 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name 182 --cfg yolov3s9a-320-640.cfg
t=ultralytics/yolov3:v183 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name 183 --cfg yolov3s15a-640.cfg
t=ultralytics/yolov3:v183 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name 184 --cfg yolov3s15a-320-640.cfg

t=ultralytics/yolov3:v185 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name 185
t=ultralytics/yolov3:v186 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name 186
n=187 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n
t=ultralytics/yolov3:v189 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name 188 --cfg yolov3s15a-320-640.cfg
n=190 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n
n=191 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n
n=192 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n

n=193 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n
n=194 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n
n=195 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n
n=196 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 10 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n

n=197 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 273 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n
n=198 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 640 --epochs 273 --batch 22 --accum 3 --weights '' --arc defaultpw --pre --multi --bucket yolov4 --name $n


# athena
n=199 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 100 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --bucket ultralytics/athena --name $n --device 0
n=200 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 100 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --bucket ultralytics/athena --name $n --device 6
n=207 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 100 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --bucket ultralytics/athena --name $n --device 7
n=208 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 0
n=211 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 100 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 0 --bucket ult/athena --name $n --cfg yolov3-spp-1cls.cfg
n=212 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 100 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 0 --bucket ult/athena --name $n --cfg yolov3-spp-1cls.cfg
n=213 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 100 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 0 --bucket ult/athena --name $n --cfg yolov3-spp-1cls.cfg
n=214 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 100 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 0 --bucket ult/athena --name $n --cfg yolov3-spp-1cls.cfg
n=215 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 100 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 0 --bucket ult/athena --name $n --cfg yolov3-spp-1cls.cfg
n=217 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all --ipc=host -it -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 100 --batch 8 --accum 8 --weights ultralytics68.pt --arc default --pre --multi --device 6 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=219 && t=ultralytics/yolov3:v215 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights ultralytics68.pt --arc default --pre --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=220 && t=ultralytics/yolov3:v215 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 20 --batch 8 --accum 8 --weights ultralytics68.pt --arc default --pre --multi --device 1 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=221 && t=ultralytics/yolov3:v215 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 30 --batch 8 --accum 8 --weights ultralytics68.pt --arc default --pre --multi --device 2 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=222 && t=ultralytics/yolov3:v215 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 40 --batch 8 --accum 8 --weights ultralytics68.pt --arc default --pre --multi --device 3 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=223 && t=ultralytics/yolov3:v215 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=224 && t=ultralytics/yolov3:v215 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 20 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 1 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=225 && t=ultralytics/yolov3:v215 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 30 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=226 && t=ultralytics/yolov3:v215 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 40 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --pre --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=227 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=228 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 20 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=229 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 20 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=240 && t=ultralytics/yolov3:v240 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg --var 0
n=241 && t=ultralytics/yolov3:v240 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --multi --device 1 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg --var 1
n=242 && t=ultralytics/yolov3:v240 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --multi --device 2 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg --var 3
n=243 && t=ultralytics/yolov3:v240 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --multi --device 3 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg --var 5
n=244 && t=ultralytics/yolov3:v240 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --multi --device 4 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg --var 7
n=245 && t=ultralytics/yolov3:v245 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights '' --arc defaultpw --multi --device 5 --bucket ult/athena --name $n --nosave --cfg yolov3-1cls.cfg
n=246 && t=ultralytics/yolov3:v245 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights '' --arc defaultpw --multi --device 6 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=247 && t=ultralytics/yolov3:v245 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights '' --arc defaultpw --multi --device 7 --bucket ult/athena --name $n --nosave --cfg yolov3-spp3-1cls.cfg
n=248 && t=ultralytics/yolov3:v245 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights darknet53.conv.74 --arc defaultpw --multi --device 5 --bucket ult/athena --name $n --nosave --cfg yolov3-1cls.cfg
n=249 && t=ultralytics/yolov3:v245 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights darknet53.conv.74 --arc defaultpw --multi --device 6 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=250 && t=ultralytics/yolov3:v245 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights darknet53.conv.74 --arc defaultpw --multi --device 7 --bucket ult/athena --name $n --nosave --cfg yolov3-spp3-1cls.cfg
n=251 && t=ultralytics/yolov3:v240 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --multi --device 3 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg --var 9
n=252 && t=ultralytics/yolov3:v240 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights ultralytics68.pt --arc defaultpw --multi --device 4 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg --var 100
n=253 && t=ultralytics/yolov3:v245 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 60 --batch 8 --accum 8 --weights darknet53.conv.74 --arc defaultpw --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-1cls.cfg
n=254 && t=ultralytics/yolov3:v245 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 60 --batch 8 --accum 8 --weights darknet53.conv.74 --arc defaultpw --multi --device 1 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=255 && t=ultralytics/yolov3:v245 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 60 --batch 8 --accum 8 --weights darknet53.conv.74 --arc defaultpw --multi --device 2 --bucket ult/athena --name $n --nosave --cfg yolov3-spp3-1cls.cfg


# wer
n=201 && t=ultralytics/yolov3:v201 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 1000 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --pre --multi --bucket ult/wer --name $n --device 0 --cfg yolov3-tiny-3cls.cfg
n=202 && t=ultralytics/yolov3:v201 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 1000 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --pre --multi --bucket ult/wer --name $n --device 1 --cfg yolov3-tiny-3cls-sm4.cfg
n=203 && t=ultralytics/yolov3:v201 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 1000 --batch 64 --accum 1 --weights '' --arc defaultpw --pre --multi --bucket ult/wer --name $n --device 2 --cfg yolov3-tiny-3cls-sm4.cfg
n=204 && t=ultralytics/yolov3:v202 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 1000 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --pre --multi --bucket ult/wer --name $n --device 3 --cfg yolov3-tiny-3cls-sm4.cfg
n=205 && t=ultralytics/yolov3:v202 && sudo docker pull $t && sudo docker run -it --gpus all -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 1000 --batch 64 --accum 1 --weights '' --arc defaultpw --pre --multi --bucket ult/wer --name $n --device 4 --cfg yolov3-tiny-3cls-sm4.cfg
n=206 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --pre --multi --notest --nosave --cache --device 0 --cfg yolov3-tiny-3cls.cfg
n=209 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 1000 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --pre --multi --bucket ult/wer --name $n --nosave --cache --device 3 --cfg yolov3-tiny-3cls.cfg
n=210 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 1000 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --pre --multi --bucket ult/wer --name $n --nosave --cache --device 1 --cfg yolov3-tiny-3cls.cfg
n=216 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 1000 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --pre --multi --bucket ult/wer --name $n --nosave --cache --device 0 --cfg yolov3-tiny-3cls.cfg
n=218 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all --ipc=host -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 1000 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc default --pre --multi --bucket ult/wer --name $n --nosave --cache --device 7 --cfg yolov3-tiny-3cls.cfg
n=230 && t=ultralytics/athena:v$n && sudo docker pull $t && sudo docker run --gpus all --ipc=host -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --multi --bucket ult/wer --name $n --nosave --cache --device 0 --cfg yolov3-tiny-1cls.cfg --single
n=231 && t=ultralytics/athena:v$n && sudo docker pull $t && sudo docker run --gpus all --ipc=host -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --multi --bucket ult/wer --name $n --nosave --cache --device 1 --cfg yolov3-tiny-1cls.cfg --single
n=232 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all --ipc=host -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --multi --bucket ult/wer --name $n --nosave --cache --device 0 --cfg yolov3-tiny-1cls.cfg --single
n=233 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all --ipc=host -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --multi --bucket ult/wer --name $n --nosave --cache --device 0 --cfg yolov3-tiny-1cls.cfg --single
n=234 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run --gpus all --ipc=host -it -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 416 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --multi --bucket ult/wer --name $n --nosave --cache --device 0 --cfg yolov3-tiny-1cls.cfg --single
n=235 && t=ultralytics/yolov3:v206 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --multi 1.2 --bucket ult/wer --name $n --nosave --cache --device 0 --cfg yolov3-tiny-1cls.cfg --single
n=236 && t=ultralytics/yolov3:v206 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --multi 1.4 --bucket ult/wer --name $n --nosave --cache --device 0 --cfg yolov3-tiny-1cls.cfg --single
n=237 && t=ultralytics/yolov3:v206 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --multi 1.6 --bucket ult/wer --name $n --nosave --device 1 --cfg yolov3-tiny-1cls.cfg --single
n=238 && t=ultralytics/yolov3:v206 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --multi 1.8 --bucket ult/wer --name $n --nosave --cache --device 0 --cfg yolov3-tiny-1cls.cfg --single
n=239 && t=ultralytics/yolov3:v206 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --multi 2.0 --bucket ult/wer --name $n --nosave --cache --device 0 --cfg yolov3-tiny-1cls.cfg --single
n=256 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 500 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --multi --bucket ult/wer --name $n --nosave --cache --device 6 --cfg yolov3-tiny-1cls.cfg --single
n=257 && t=ultralytics/yolov3:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 500 --batch 64 --accum 1 --weights yolov3-tiny.pt --arc defaultpw --multi --bucket ult/wer --name $n --nosave --cache --device 7 --cfg yolov3-tiny-1cls.cfg --single --adam


#coco
n=2 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --nosave --bucket ult/coco --name $n --device 0 --multi
n=3 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 384 --epochs 27 --batch 32 --accum 2 --weights '' --device 1 --cfg yolov3.cfg --nosave --bucket ult/coco --name $n
n=4 && t=ultralytics/coco:v3 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 384 --epochs 27 --batch 32 --accum 2 --weights '' --device 2 --cfg yolov3-spp.cfg --nosave --bucket ult/coco --name $n
n=5 && t=ultralytics/coco:v3 && sudo docker pull $t && sudo docker run -d --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 384 --epochs 27 --batch 32 --accum 2 --weights '' --device 3 --cfg yolov3-spp3.cfg --nosave --bucket ult/coco --name $n
n=6 && t=ultralytics/coco:v3 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 384 --epochs 27 --batch 32 --accum 2 --weights '' --device 0 --cfg yolov4.cfg --nosave --bucket ult/coco --name $n
n=7 && t=ultralytics/coco:v3 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 384 --epochs 27 --batch 32 --accum 2 --weights '' --device 1 --cfg yolov4s.cfg --nosave --bucket ult/coco --name $n
n=8 && t=ultralytics/coco:v8 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 384 --epochs 27 --batch 32 --accum 2 --weights '' --device 0 --cfg yolov4.cfg --nosave --bucket ult/coco --name $n
n=9 && t=ultralytics/coco:v9 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 20 --accum 3 --weights '' --device 0 --cfg yolov4a.cfg --nosave --bucket ult/coco --name $n --multi
n=10 && t=ultralytics/coco:v9 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 20 --accum 3 --weights '' --device 0 --cfg yolov4b.cfg --nosave --bucket ult/coco --name $n --multi
n=11 && t=ultralytics/coco:v9 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 20 --accum 3 --weights '' --device 0 --cfg yolov4c.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=12 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 20 --accum 3 --weights '' --device 0 --cfg yolov3-spp.cfg --nosave --bucket ult/coco --name $n --multi
n=13 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 20 --accum 3 --weights '' --device 0 --cfg yolov3-spp.cfg --nosave --bucket ult/coco --name $n --multi
n=14 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 20 --accum 3 --weights '' --device 0 --cfg yolov3-spp.cfg --nosave --bucket ult/coco --name $n --multi
n=15 && t=ultralytics/coco:v14 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov3-spp.cfg --nosave --bucket ult/coco --name $n --multi
n=16 && t=ultralytics/coco:v9 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov4a.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=17 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov4d.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=18 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov4a.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=19 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov4e.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=20 && t=ultralytics/coco:v14 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 512 608 --epochs 27 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov3-spp.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=21 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov3-sppe.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=22 && t=ultralytics/coco:v14 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 608 608 --epochs 27 --batch 12 --accum 6 --weights '' --device 0 --cfg yolov3-spp.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=23 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=24 && t=ultralytics/coco:v24 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 512 608 --epochs 27 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=25 && t=ultralytics/coco:v24 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 608 608 --epochs 27 --batch 12 --accum 6 --weights '' --device 0 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=26 && t=ultralytics/coco:v24 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 512 608 --epochs 27 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov3-spp.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=27 && t=ultralytics/coco:v24 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 512 608 --epochs 273 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov3-spp.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=28 && t=ultralytics/coco:v24 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 608 608 --epochs 273 --batch 12 --accum 6 --weights '' --device 0 --cfg yolov3-spp.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=29 && t=ultralytics/coco:v24 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 512 608 --epochs 27 --batch 15 --accum 4 --weights '' --device 0 --cfg yolov4a.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=30 && t=ultralytics/coco:v24 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 608 608 --epochs 273 --batch 12 --accum 6 --weights '' --device 0 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=31 && t=ultralytics/coco:v31 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov3-sppf.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=32 && t=ultralytics/coco:v31 && sudo docker pull $t && sudo docker run --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov3-sppg.cfg --nosave --bucket ult/coco --name $n --multi
n=33 && t=ultralytics/coco:v33 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 608 608 --epochs 273 --batch 12 --accum 6 --weights '' --device 0 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=34 && t=ultralytics/coco:v34 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 608 608 --epochs 273 --batch 12 --accum 6 --weights '' --device 0 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=35 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi
n=36 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 16 --accum 4 --weights '' --device 1 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi
n=37 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 16 --accum 4 --weights '' --device 2 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi
n=38 && t=ultralytics/coco:v35 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 512 608 --epochs 27 --batch 10 --accum 8 --weights '' --device 3 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi
n=39 && t=ultralytics/coco:v35 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 512 608 --epochs 27 --batch 10 --accum 6 --weights '' --device 4 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi
n=40 && t=ultralytics/coco:v35 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 512 608 --epochs 27 --batch 8 --accum 8 --weights '' --device 5 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi
n=41 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 16 --accum 4 --weights '' --device 6 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi
n=42 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 16 --accum 4 --weights '' --device 7 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi
n=45 && t=ultralytics/coco:v45 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 512 608 --epochs 27 --batch 8 --accum 16 --weights '' --device 2 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi
n=46 && t=ultralytics/coco:v45 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 512 608 --epochs 27 --batch 8 --accum 4 --weights '' --device 6 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi
n=47 && t=ultralytics/coco:v45 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 512 608 --epochs 27 --batch 8 --accum 2 --weights '' --device 7 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi
n=48 && t=ultralytics/coco:v45 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 512 608 --epochs 27 --batch 8 --accum 8 --weights '' --device 3 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi
n=49 && t=ultralytics/coco:v45 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 512 608 --epochs 27 --batch 8 --accum 32 --weights '' --device 4 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi
n=50 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 512 608 --epochs 27 --batch 8 --accum 8 --weights '' --device 3 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi
n=51 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 512 608 --epochs 27 --batch 8 --accum 8 --weights '' --device 4 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi
n=52 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 512 608 --epochs 27 --batch 8 --accum 8 --weights '' --device 2 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi
n=53 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 608 608 --epochs 27 --batch 8 --accum 8 --weights '' --device 5 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n
n=54 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --nosave --bucket ult/coco --name $n --device 0 --multi
n=55 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --weights '' --epochs 273 --batch 16 --accum 4 --pre --nosave --bucket ult/coco --name $n --device 2 --multi
n=56 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 608 608 --epochs 273 --batch 16 --accum 4 --weights '' --device 3 --cfg yolov3-spp.cfg --nosave --bucket ult/coco --name $n --multi
n=57 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 608 608 --epochs 273 --batch 16 --accum 4 --weights '' --device 4 --cfg yolov3-spp.cfg --nosave --bucket ult/coco --name $n --multi
n=58 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 608 608 --epochs 273 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov3-spp.cfg --nosave --bucket ult/coco --name $n --multi
n=59 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 608 608 --epochs 273 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov3-spp.cfg --nosave --bucket ult/coco --name $n --multi
n=60 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 608 608 --epochs 273 --batch 16 --accum 4 --weights '' --device 1 --cfg yolov3-spp.cfg --nosave --bucket ult/coco --name $n --multi
n=61 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 512 608 --epochs 273 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi
n=62 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 512 608 --epochs 273 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov3-sppa.cfg --nosave --bucket ult/coco --name $n --multi
n=63 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 273 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov3-spp.cfg --nosave --bucket ult/coco --name $n --multi
n=64 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 16 --accum 4 --weights '' --device 1 --cfg yolov3-spp.cfg --nosave --bucket ult/coco --name $n --multi
n=65 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov3-spp.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=66 && t=ultralytics/coco:v65 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 8 --accum 8 --weights '' --device 0 --cfg darknet53-bifpn3.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=67 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 15 --accum 4 --weights '' --device 0 --cfg csdarknet53-bifpn-optimal.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=68 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 273 --batch 16 --accum 4 --weights '' --device 0 --cfg yolov3-spp.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=69 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 15 --accum 4 --weights '' --device 0 --cfg csdarknet53-bifpn-optimal.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown
n=70 && t=ultralytics/coco:v69 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --data coco2014.data --img-size 416 608 --epochs 27 --batch 15 --accum 4 --weights '' --device 0 --cfg csresnext50-bifpn-optimal.cfg --nosave --bucket ult/coco --name $n --multi && sudo shutdown


# athena
n=32 && t=ultralytics/athena:v32 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 100 --batch 8 --accum 8 --weights ultralytics68.pt --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=33 && t=ultralytics/athena:v33 && sudo docker pull $t && sudo docker run --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 10 --batch 8 --accum 8 --weights ultralytics68.pt --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=34 && t=ultralytics/athena:v33 && sudo docker pull $t && sudo docker run --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 20 --batch 8 --accum 8 --weights ultralytics68.pt --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg
n=35 && t=ultralytics/athena:v33 && sudo docker pull $t && sudo docker run --gpus all --ipc=host -v "$(pwd)"/out:/usr/src/out $t python3 train.py --data ../out/data.data --img-size 608 --epochs 30 --batch 8 --accum 8 --weights ultralytics68.pt --multi --device 0 --bucket ult/athena --name $n --nosave --cfg yolov3-spp-1cls.cfg && sudo shutdown


# wer
n=18 && t=ultralytics/wer:v18 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.conv.15 --multi --bucket ult/wer --name $n --nosave --cache --device 0 --cfg yolov3-tiny-1cls.cfg --single --adam
n=19 && t=ultralytics/wer:v18 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.conv.15 --multi --bucket ult/wer --name $n --nosave --cache --device 1 --cfg yolov3-tiny-3l-1cls.cfg --single --adam
n=20 && t=ultralytics/wer:v18 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.conv.15 --multi --bucket ult/wer --name $n --nosave --cache --device 2 --cfg yolov3-tiny-prnc-1cls.cfg --single --adam
n=21 && t=ultralytics/wer:v18 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.conv.15 --multi --bucket ult/wer --name $n --nosave --cache --device 3 --cfg yolov3-tiny-prn-1cls.cfg --single --adam
n=22 && t=ultralytics/wer:v18 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights '' --multi --bucket ult/wer --name $n --nosave --cache --device 0 --cfg yolov3-tiny-3l-1cls.cfg --single --adam
n=23 && t=ultralytics/wer:v18 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights '' --multi --bucket ult/wer --name $n --nosave --cache --device 1 --cfg yolov3-tinyr-3l-1cls.cfg --single --adam
n=24 && t=ultralytics/wer:v24 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights '' --multi --bucket ult/wer --name $n --nosave --cache --device 3 --cfg yolov3-tiny-3l-1cls.cfg --single --adam
n=25 && t=ultralytics/wer:v25 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 100 --batch 64 --accum 1 --weights yolov3-tiny.pt --multi --bucket ult/wer --name $n --nosave --cache --device 2 --cfg yolov3-tiny3-1cls.cfg --single --adam
n=26 && t=ultralytics/wer:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/data:/usr/src/data $t python3 train.py --data ../data/sm4/out.data --img-size 320 --epochs 1000 --batch 64 --accum 1 --weights yolov3-tiny.pt --multi --bucket ult/wer --name $n --nosave --cache --device 0 --cfg yolov3-tiny3-1cls.cfg --single --adam
