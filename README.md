## RM-LFPN-YOLO for Rebar Counting

## Required environment
torch==1.13.1 cuda11.6

## File download 
weight
webgage：https://pan.baidu.com/s/1VNGmzciW8eOKSWq_Umfmog?pwd=rzh2
password：rzh2

datasets
webgage：https://pan.baidu.com/s/1GOy91bjVEB9f5KIV-sUbEQ?pwd=psq0 
password：psq0

## Training
1. Preparation of datasets 
**You need to download the dataset before training, unzip it and put it in the root directory.**  

2. Processing of datasets   
python voc_annotation.py
Modify annotation_mode=2 in voc_annotation.py and run voc_annotation.py to generate 2007_train.txt and 2007_val.txt in the root directory.   

3. Training 
python train.py 

## Evaluation 
python get_map.py 
get the evaluation results, which are saved in the map_out folder.(Modify model_path as well as classes_path inside yolo.py.)
python summary.py 
get the number of parameters and calculations for the model.
## Predict   
python predict.py
(You need to go to yolo.py and change the model_path and classes_path.)
**Settings inside predict.py allow for fps testing and video video detection.** 

## Reference
github:https://github.com/bubbliiiing/mobilenet-yolov4-pytorch
