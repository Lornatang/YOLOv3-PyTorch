# Download coco2017 datasets
wget http://images.cocodataset.org/zips/train2017.zip ./train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip ./val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip ./annotations_trainval2017.zip

# Unzip the dataset
unzip -e train2017.zip
unzip -e val2017.zip
unzip -e annotations_trainval2017.zip

# Extract the tag information in the Json file and form a character detection dataset
mkdir -p ./images/train
mkdir -p ./images/valid
mkdir -p ./labels/train
mkdir -p ./labels/valid

find train2017/ -name "*.jpg" -exec cp -r {} ./images/train/ \; 
find val2017/ -name "*.jpg" -exec cp -r {} ./images/valid/ \; 

python3 coco2yolo.py --json_path annotations/instances_train2017.json --save_path ./labels/train
python3 coco2yolo.py --json_path annotations/instances_val2017.json --save_path ./labels/valid

cat val.txt > test.txt

rm ./labels/train/classes.txt
rm ./labels/valid/classes.txt

# shellcheck disable=SC2035
rm *.zip
# shellcheck disable=SC2035
rm *.json
rm -rf train2017
rm -rf val2017

mkdir ../data/coco2017
mv ./images ../data/coco2017/images
mv ./labels ../data/coco2017/labels
# shellcheck disable=SC2035
mv *.txt ../data/coco2017/
