# Usage

## Step1: Download datasets

- [Google Driver](https://drive.google.com/drive/folders/1xuXk-uvAe-F2m6oxbOQB3DFM573GPN57?usp=share_link)
- [Baidu Driver](https://pan.baidu.com/s/1UsLQvMLbm1uhv-tYTL2q-w?pwd=llot)

## Step2: Prepare the dataset in the following format

```text
- VOC0712
    - process_voc0712_dataset.sh
    - voc_label.py
    - VOCtest_06-Nov-2007.tar
    - VOCtrainval_06-Nov-2007.tar
    - VOCtrainval_11-May-2012.tar
```

## Step3: Preprocess the train dataset

```bash
bash process_voc_datasets.sh
```

## Step4: Check that the final dataset directory schema is completely correct

```text
- VOC0712
    - images
        - train
        - test
    - labels
        - train
        - test
```

