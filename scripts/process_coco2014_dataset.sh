# Download coco2014 datasets
wget http://images.cocodataset.org/zips/train2014.zip ./train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip ./val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip ./annotations_trainval2014.zip

# Unzip the dataset
unzip -e train2014.zip
unzip -e val2014.zip
unzip -e annotations_trainval2014.zip

# Extract the tag information in the Json file and form a character detection dataset
mkdir -p ./images/train
mkdir -p ./images/valid
mkdir -p ./labels/train
mkdir -p ./labels/valid

find train2014/ -name "*.jpg" -exec cp -r {} ./images/train/ \; 
find val2014/ -name "*.jpg" -exec cp -r {} ./images/valid/ \; 

python3 coco2yolo.py --json_path annotations/instances_train2014.json --save_path ./labels/train
python3 coco2yolo.py --json_path annotations/instances_val2014.json --save_path ./labels/valid

rm ./labels/train/classes.txt
rm ./labels/valid/classes.txt

# shellcheck disable=SC2035
rm *.zip
# shellcheck disable=SC2035
rm *.json
rm -rf train2014
rm -rf val2014
