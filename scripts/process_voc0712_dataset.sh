# Download voc datasets
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar ./VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar ./VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar ./VOCtrainval_11-May-2012.tar

# Decompress Ali Tianchi Street View Character Detection Dataset
tar -xvf ./VOCtrainval_06-Nov-2007.tar -C ./
tar -xvf ./VOCtest_06-Nov-2007.tar -C ./
tar -xvf ./VOCtrainval_11-May-2012.tar -C ./

# Extract the tag information in the Json file and form a character detection dataset
mkdir -p ./images/train
mkdir -p ./images/test
mkdir -p ./labels/train
mkdir -p ./labels/test

python3 voc2yolo.py

# Delete residual files in the process of making datasets
rm -rf ./2007_test.txt
# shellcheck disable=SC2035
rm -rf ./2007_train.txt
# shellcheck disable=SC2035
rm -rf ./2007_val.txt
# shellcheck disable=SC2035
rm -rf ./2012_train.txt
# shellcheck disable=SC2035
rm -rf ./2012_val.txt
# shellcheck disable=SC2035
rm -rf ./*.tar
rm -rf ./VOCdevkit

mkdir -p ../data/voc0712

mv ./images ../data/voc0712
mv ./labels ../data/voc0712
mv ./train.txt ../data/voc0712
mv ./test.txt ../data/voc0712
